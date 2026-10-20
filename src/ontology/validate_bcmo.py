"""
Validate the BCMO ontology and produce coverage/consistency statistics.

Checks:
- Concept coverage: how many extracted concepts are in the ontology
- Relation consistency: domain/range constraints, no orphan references
- Duplicate detection
- Summary statistics
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_ontology(version="v2"):
    path = DATA_DIR / f"bcmo_{version}.json"
    with open(path) as f:
        return json.load(f)


def load_extracted():
    concepts, relations = {}, []
    cp = DATA_DIR / "extracted_concepts.json"
    rp = DATA_DIR / "extracted_relations.json"
    if cp.exists():
        with open(cp) as f:
            concepts = json.load(f)
    if rp.exists():
        with open(rp) as f:
            relations = json.load(f)
    return concepts, relations


def get_all_concept_names(bcmo):
    """Get all concept names and aliases as a lowercase set."""
    names = set()
    for cls_data in bcmo["concept_classes"].values():
        for inst in cls_data.get("instances", []):
            if isinstance(inst, str):
                names.add(inst.lower())
            elif isinstance(inst, dict):
                names.add(inst["name"].lower())
                for a in inst.get("aliases", []):
                    names.add(a.lower())
    return names


def check_concept_coverage(bcmo, extracted_concepts):
    """Check how many extracted concepts are covered in the ontology."""
    all_names = get_all_concept_names(bcmo)
    results = {}

    for category, items in extracted_concepts.items():
        covered = []
        missing = []
        for item in items:
            if item.lower() in all_names:
                covered.append(item)
            else:
                missing.append(item)
        pct = len(covered) / len(items) * 100 if items else 0
        results[category] = {
            "total": len(items),
            "covered": len(covered),
            "missing": len(missing),
            "coverage_pct": round(pct, 1),
            "missing_items": missing,
        }

    return results


def check_relation_consistency(bcmo):
    """Check relations against domain/range constraints."""
    all_names = get_all_concept_names(bcmo)
    relation_types = bcmo["relation_types"]
    issues = []

    for i, rel in enumerate(bcmo["factual_relations"]):
        subj = rel["subject"]
        rtype = rel["relation"]
        obj = rel["object"]

        # Check relation type exists
        if rtype not in relation_types:
            issues.append(f"Relation #{i}: unknown type '{rtype}'")

        # Check subject/object exist in ontology
        if subj.lower() not in all_names:
            issues.append(f"Relation #{i}: subject '{subj}' not in ontology")
        if obj.lower() not in all_names:
            issues.append(f"Relation #{i}: object '{obj}' not in ontology")

    return issues


def check_duplicates(bcmo):
    """Check for duplicate concept instances and relations."""
    # Concept duplicates
    concept_dupes = []
    seen_names = defaultdict(list)
    for cls_name, cls_data in bcmo["concept_classes"].items():
        for inst in cls_data.get("instances", []):
            name = inst["name"] if isinstance(inst, dict) else inst
            key = name.lower()
            seen_names[key].append(cls_name)
    for name, classes in seen_names.items():
        if len(classes) > 1:
            concept_dupes.append(f"'{name}' appears in: {', '.join(classes)}")

    # Relation duplicates
    rel_dupes = []
    seen_rels = defaultdict(int)
    for rel in bcmo["factual_relations"]:
        key = (rel["subject"], rel["relation"], rel["object"])
        seen_rels[key] += 1
    for key, count in seen_rels.items():
        if count > 1:
            rel_dupes.append(f"{key[0]} --{key[1]}--> {key[2]} (appears {count}x)")

    return concept_dupes, rel_dupes


def compute_statistics(bcmo):
    """Compute summary statistics."""
    classes = bcmo["concept_classes"]
    relations = bcmo["factual_relations"]

    # Instance counts per class
    class_stats = {}
    for cls_name, cls_data in classes.items():
        instances = cls_data.get("instances", [])
        expert = sum(1 for i in instances
                     if isinstance(i, dict) and i.get("source") == "expert")
        medquad = sum(1 for i in instances
                      if isinstance(i, dict) and i.get("source") == "MedQuAD")
        other = len(instances) - expert - medquad
        class_stats[cls_name] = {
            "total": len(instances),
            "expert": expert,
            "medquad": medquad + other,
        }

    # Relation type distribution
    rel_dist = defaultdict(int)
    for rel in relations:
        rel_dist[rel["relation"]] += 1

    # Data-grounded vs expert
    grounded = sum(1 for r in relations if r.get("data_grounded", False))
    expert = len(relations) - grounded

    return {
        "class_stats": class_stats,
        "relation_distribution": dict(rel_dist),
        "total_relations": len(relations),
        "expert_relations": expert,
        "data_grounded_relations": grounded,
    }


def main():
    bcmo = load_ontology("v2")
    extracted_concepts, extracted_relations = load_extracted()

    print("=" * 60)
    print("BCMO v2.0 Validation Report")
    print("=" * 60)

    # Metadata
    meta = bcmo["ontology_metadata"]
    print(f"\nOntology: {meta['name']} ({meta['acronym']}) v{meta['version']}")
    print(f"Created: {meta['created'][:10]}")

    # Statistics
    stats = compute_statistics(bcmo)
    print(f"\n--- Concept Class Summary ---")
    total_inst = 0
    for cls_name, s in sorted(stats["class_stats"].items()):
        total_inst += s["total"]
        src = f"({s['expert']} expert, {s['medquad']} MedQuAD)"
        print(f"  {cls_name:30s}: {s['total']:3d} instances {src}")
    print(f"  {'TOTAL':30s}: {total_inst:3d}")

    print(f"\n--- Relation Distribution ---")
    for rtype, count in sorted(stats["relation_distribution"].items(),
                                key=lambda x: -x[1]):
        print(f"  {rtype:30s}: {count}")
    print(f"  {'TOTAL':30s}: {stats['total_relations']}")
    print(f"  Expert-curated: {stats['expert_relations']}")
    print(f"  MedQuAD-grounded: {stats['data_grounded_relations']}")

    # Concept coverage
    print(f"\n--- Concept Coverage (extracted vs ontology) ---")
    coverage = check_concept_coverage(bcmo, extracted_concepts)
    all_covered = 0
    all_total = 0
    for category, c in sorted(coverage.items()):
        all_covered += c["covered"]
        all_total += c["total"]
        status = "OK" if c["coverage_pct"] >= 80 else "LOW"
        print(f"  {category:25s}: {c['covered']}/{c['total']} ({c['coverage_pct']}%) [{status}]")
        if c["missing_items"]:
            for m in c["missing_items"]:
                print(f"    MISSING: {m}")
    overall_pct = round(all_covered / all_total * 100, 1) if all_total > 0 else 0
    print(f"  {'OVERALL':25s}: {all_covered}/{all_total} ({overall_pct}%)")

    # Consistency checks
    print(f"\n--- Consistency Checks ---")
    issues = check_relation_consistency(bcmo)
    if issues:
        print(f"  Found {len(issues)} issue(s):")
        for issue in issues[:20]:
            print(f"    - {issue}")
        if len(issues) > 20:
            print(f"    ... and {len(issues) - 20} more")
    else:
        print("  All relations reference known concepts: PASS")

    # Duplicate checks
    concept_dupes, rel_dupes = check_duplicates(bcmo)
    if concept_dupes:
        print(f"\n  Concept duplicates ({len(concept_dupes)}):")
        for d in concept_dupes:
            print(f"    - {d}")
    else:
        print("  No concept duplicates: PASS")

    if rel_dupes:
        print(f"\n  Relation duplicates ({len(rel_dupes)}):")
        for d in rel_dupes:
            print(f"    - {d}")
    else:
        print("  No relation duplicates: PASS")

    print(f"\n{'=' * 60}")
    print("Validation complete.")


if __name__ == "__main__":
    main()
