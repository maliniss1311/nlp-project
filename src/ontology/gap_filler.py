"""
Ontology gap filler for BCMO.

Closes the loop between fact-checking and the ontology: claims that cannot be
verified because BCMO is missing a concept, an alias, or a relation are logged
as gaps, matched against MedQuAD evidence, proposed as ontology additions,
reviewed by a human, and merged into a new BCMO version with provenance.

Gap types:
    alias     - entity is a surface form of an existing concept (e.g. "alcohol")
    concept   - entity has no counterpart in BCMO (e.g. "vitamin C")
    relation  - both entities are known but the triple is not recorded

Usage:
    python src/ontology/gap_filler.py detect  [--claims FILE]   # log gaps from claims
    python src/ontology/gap_filler.py propose                   # attach evidence + proposals
    python src/ontology/gap_filler.py review  [--approve ID ...] [--reject ID ...] [--approve-evidenced]
    python src/ontology/gap_filler.py merge   [--version 2.1]   # write data/bcmo_v<version>.json
    python src/ontology/gap_filler.py report  [--version 2.1]   # coverage before vs after
    python src/ontology/gap_filler.py demo                      # run the full loop on sample claims

--claims FILE is a JSON list of {"subject", "relation", "object", "negated"?, "source_text"?}
or a plain-text file of claim sentences (one per line, run through ClaimExtractor).
"""

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ontology.bcmo_queries import BCMO, DATA_DIR
from pipeline.claim_extractor import Claim, ClaimExtractor
from pipeline.entity_linker import EntityLinker, _normalize

GAPS_PATH = DATA_DIR / "ontology_gaps.json"
MEDQUAD_PATH = DATA_DIR / "medquad_breast_cancer.json"

NEGATION_CUES = [
    "no evidence", "not been shown", "has not been", "have not been", "does not",
    "do not", "did not", "is not", "are not", "no link", "no association",
    "not linked", "not associated", "no proof", "not proven", "myth", "unlikely",
]

# Words in a sentence that signal the relation being asserted.
RELATION_CUES = {
    "treats": ["treat", "therapy", "used for", "given"],
    "prevents": ["prevent", "lower", "reduce"],
    "increases_risk_of": ["increase", "risk", "higher", "linked", "cause"],
    "decreases_risk_of": ["decrease", "lower", "reduce", "protect"],
    "causes": ["cause", "lead to", "result in", "linked"],
    "detects": ["detect", "find", "diagnose", "screen", "check"],
    "has_symptom": ["symptom", "sign"],
    "has_side_effect": ["side effect", "cause", "may"],
    "has_biomarker": ["receptor", "positive", "negative", "marker"],
    "contraindicated_for": ["should not", "not recommended", "avoid"],
    "associated_with": ["associated", "linked", "related"],
    "reduces_recurrence_risk": ["recurrence", "come back", "reduce"],
    "improves_survival": ["survival", "live longer", "improve"],
}

# Keyword heuristics for suggesting a class for brand-new concepts.
CLASS_HINTS = [
    ("Medication", ["mab", "nib", "ib ", "drug", "pill", "vitamin", "supplement"]),
    ("Treatment", ["therapy", "surgery", "ectomy", "treatment"]),
    ("DiagnosticTest", ["scan", "test", "biopsy", "mammo", "mri", "ultrasound"]),
    ("Symptom", ["pain", "lump", "swelling", "discharge", "rash"]),
    ("SideEffect", ["loss", "nausea", "fatigue", "toxicity", "neuropathy"]),
    ("Biomarker", ["receptor", "gene", "her2", "brca", "ki-67"]),
]


# --------------------------------------------------------------------------- #
# Gap store
# --------------------------------------------------------------------------- #

def _gap_id(kind, key):
    return kind[:3] + "-" + hashlib.md5(key.encode()).hexdigest()[:8]


def load_gaps():
    if GAPS_PATH.exists():
        with open(GAPS_PATH) as f:
            return json.load(f)
    return {"created": datetime.now().isoformat(), "gaps": {}}


def save_gaps(store):
    store["updated"] = datetime.now().isoformat()
    with open(GAPS_PATH, "w") as f:
        json.dump(store, f, indent=2)


def _record(store, kind, key, payload, example):
    gid = _gap_id(kind, key)
    gap = store["gaps"].get(gid)
    if gap is None:
        gap = {"id": gid, "type": kind, "status": "open", "count": 0,
               "examples": [], **payload}
        store["gaps"][gid] = gap
    gap["count"] += 1
    if example and example not in gap["examples"] and len(gap["examples"]) < 5:
        gap["examples"].append(example)
    return gap


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def _tokens(text):
    return set(t for t in re.split(r"[\s/-]+", _normalize(text)) if t)


def _alias_candidate(entity, linker):
    """Find an existing concept whose name/alias contains every token of `entity`."""
    ent = _tokens(entity)
    if not ent:
        return None
    best = None
    for stored_norm, canonical, cls in linker.all_names:
        stored = _tokens(stored_norm)
        if ent <= stored:
            # Prefer the tightest containing name
            if best is None or len(stored) < best[0]:
                best = (len(stored), canonical, cls)
    return (best[1], best[2]) if best else None


def _misinfo_match(entity, bcmo):
    ent = _tokens(entity)
    for item in bcmo.misinfo:
        claim_tokens = _tokens(item["claim"])
        # allow plural forms ("bras" vs "bra")
        claim_tokens |= {t.rstrip("s") for t in claim_tokens}
        if ent and ent <= claim_tokens:
            return item
    return None


def detect_entity_gap(entity, linker, bcmo, store, example):
    linked = linker.link(entity)
    if linked.match_type != "unlinked":
        return linked.canonical_name
    norm = _normalize(entity)
    cand = _alias_candidate(entity, linker)
    if cand:
        _record(store, "alias", norm, {
            "entity": entity, "target_concept": cand[0], "target_class": cand[1],
        }, example)
        return None
    _record(store, "concept", norm, {"entity": entity}, example)
    return None


def detect_gaps(claims, bcmo=None, store=None):
    """Run claims through linking + verification and log every gap."""
    bcmo = bcmo or BCMO()
    linker = EntityLinker(bcmo)
    store = store or load_gaps()
    results = []
    for claim in claims:
        example = claim.source_text or repr(claim)
        subj = detect_entity_gap(claim.subject, linker, bcmo, store, example)
        obj = detect_entity_gap(claim.object, linker, bcmo, store, example)

        verdict = None
        if subj and obj:
            verdict, _ = bcmo.verify_claim(subj, claim.relation, obj)
            if verdict == "unsupported":
                key = f"{subj.lower()}|{claim.relation}|{obj.lower()}"
                _record(store, "relation", key, {
                    "subject": subj, "relation": claim.relation, "object": obj,
                }, example)
        else:
            # Relation can't be checked until the entities exist; log it so it is
            # re-examined after alias/concept gaps are filled.
            key = f"{claim.subject.lower()}|{claim.relation}|{claim.object.lower()}"
            _record(store, "relation", key, {
                "subject": subj or claim.subject, "relation": claim.relation,
                "object": obj or claim.object, "blocked_by_entity_gap": True,
            }, example)
        results.append((claim, verdict or "unsupported"))
    save_gaps(store)
    return results


# --------------------------------------------------------------------------- #
# Evidence + proposals
# --------------------------------------------------------------------------- #

def _load_medquad():
    with open(MEDQUAD_PATH) as f:
        return json.load(f)


def _sentences(text):
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\s+-\s+", text) if len(s.strip()) > 15]


def _surface_forms(name, bcmo):
    forms = {name}
    inst = bcmo.get_instance(name)
    if inst:
        forms.update(inst.get("aliases", []))
    # Also use the head word(s) of multi-word names, e.g. "Alcohol Consumption" -> "alcohol"
    words = [w for w in re.split(r"[\s/]+", name) if len(w) > 3]
    if len(words) > 1:
        forms.add(words[0])
    return {f.lower() for f in forms if len(f) > 1}


def _contains(sentence, forms):
    s = sentence.lower()
    return any(re.search(r"(?<!\w)" + re.escape(f) + r"(?!\w)", s) for f in forms)


def find_evidence(subject, relation, obj, bcmo, docs, limit=5):
    """Return MedQuAD sentences mentioning both entities, with polarity."""
    sf, of = _surface_forms(subject, bcmo), _surface_forms(obj, bcmo)
    cues = RELATION_CUES.get(relation, [])
    hits = []
    for doc in docs:
        for qa in doc.get("qa_pairs", []):
            for sent in _sentences(qa.get("answer", "")):
                if not (_contains(sent, sf) and _contains(sent, of)):
                    continue
                low = sent.lower()
                if cues and not any(c in low for c in cues):
                    continue
                hits.append({
                    "sentence": sent[:400],
                    "negated": any(c in low for c in NEGATION_CUES),
                    "source_doc": doc["id"], "source_qa": qa["qid"],
                    "source": doc.get("source"), "url": doc.get("url"),
                })
                if len(hits) >= limit:
                    return hits
    return hits


def _suggest_class(entity):
    e = " " + entity.lower() + " "
    for cls, hints in CLASS_HINTS:
        if any(h in e for h in hints):
            return cls
    return None


def propose(store=None, bcmo=None):
    """Attach evidence and a concrete ontology edit to every open gap."""
    store = store or load_gaps()
    bcmo = bcmo or BCMO()
    linker = EntityLinker(bcmo)
    docs = _load_medquad()

    for gap in store["gaps"].values():
        if gap["status"] not in ("open", "proposed"):
            continue

        if gap["type"] == "alias":
            gap["proposal"] = {
                "action": "add_alias", "concept": gap["target_concept"],
                "class": gap["target_class"], "alias": gap["entity"],
            }
            gap["evidence_label"] = "lexical"

        elif gap["type"] == "concept":
            misinfo = _misinfo_match(gap["entity"], bcmo)
            if misinfo:
                gap["proposal"] = {
                    "action": "add_concept", "class": "MisinformationTopic",
                    "name": gap["entity"].title(),
                    "misinformation_claim": misinfo["claim"],
                    "source": misinfo.get("source"),
                }
                gap["evidence_label"] = "known_misinformation"
            else:
                gap["proposal"] = {
                    "action": "add_concept", "class": _suggest_class(gap["entity"]),
                    "name": gap["entity"].title(),
                    "needs": "class confirmation and UMLS CUI / NCI code",
                }
                gap["evidence_label"] = "unverifiable"

        elif gap["type"] == "relation":
            # Re-resolve entities in case alias/concept gaps were since merged.
            subj = linker.link(gap["subject"])
            obj = linker.link(gap["object"])
            if subj.match_type == "unlinked" or obj.match_type == "unlinked":
                gap["proposal"] = None
                gap["evidence_label"] = "blocked_by_entity_gap"
                continue
            gap["subject"], gap["object"] = subj.canonical_name, obj.canonical_name
            gap.pop("blocked_by_entity_gap", None)

            if bcmo.verify_claim(gap["subject"], gap["relation"], gap["object"])[0] != "unsupported":
                gap["status"] = "resolved"
                gap["evidence_label"] = "already_in_ontology"
                continue

            # A claim about a known myth topic is contradicted by the curated debunk.
            myth = _misinfo_match(gap["subject"], bcmo) if subj.concept_class == "MisinformationTopic" else None
            if myth and myth.get("verdict") == "false":
                gap["evidence"] = [{"sentence": myth["explanation"], "negated": True,
                                    "source": myth.get("source")}]
                gap["evidence_label"] = "contradicted"
                gap["proposal"] = {
                    "action": "add_relation", "subject": gap["subject"],
                    "relation": gap["relation"], "object": gap["object"], "negated": True,
                    "source": myth.get("source"), "source_doc": None, "source_qa": None,
                }
                gap["status"] = "proposed"
                continue

            ev = find_evidence(gap["subject"], gap["relation"], gap["object"], bcmo, docs)
            gap["evidence"] = ev
            pos = [e for e in ev if not e["negated"]]
            neg = [e for e in ev if e["negated"]]
            if pos and len(pos) >= len(neg):
                label, chosen, negated = "supported", pos[0], False
            elif neg:
                label, chosen, negated = "contradicted", neg[0], True
            else:
                label, chosen, negated = "unverifiable", None, False
            gap["evidence_label"] = label
            gap["proposal"] = None if chosen is None else {
                "action": "add_relation", "subject": gap["subject"],
                "relation": gap["relation"], "object": gap["object"],
                "negated": negated, "source_doc": chosen["source_doc"],
                "source_qa": chosen["source_qa"],
            }

        if gap["status"] == "open":
            gap["status"] = "proposed"

    save_gaps(store)
    return store


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #

def review(approve=(), reject=(), approve_evidenced=False, store=None):
    store = store or load_gaps()
    for gid in approve:
        if gid in store["gaps"]:
            store["gaps"][gid]["status"] = "approved"
    for gid in reject:
        if gid in store["gaps"]:
            store["gaps"][gid]["status"] = "rejected"
    if approve_evidenced:
        for gap in store["gaps"].values():
            if (gap["status"] == "proposed" and gap.get("proposal")
                    and gap.get("evidence_label") in ("lexical", "supported",
                                                      "contradicted", "known_misinformation")):
                gap["status"] = "approved"
    save_gaps(store)
    return store


def print_gaps(store):
    order = {"approved": 0, "proposed": 1, "open": 2, "merged": 3, "resolved": 4, "rejected": 5}
    gaps = sorted(store["gaps"].values(), key=lambda g: (order.get(g["status"], 9), -g["count"]))
    print(f"{'ID':<14}{'TYPE':<10}{'STATUS':<10}{'N':>3}  {'EVIDENCE':<22}DETAIL")
    for g in gaps:
        if g["type"] == "relation":
            detail = f"({g['subject']}, {g['relation']}, {g['object']})"
        elif g["type"] == "alias":
            detail = f"'{g['entity']}' -> {g['target_concept']} [{g['target_class']}]"
        else:
            cls = (g.get("proposal") or {}).get("class") or "?"
            detail = f"'{g['entity']}' -> new {cls}"
        print(f"{g['id']:<14}{g['type']:<10}{g['status']:<10}{g['count']:>3}  "
              f"{g.get('evidence_label', '-'):<22}{detail}")
        for ev in (g.get("evidence") or [])[:1]:
            print(f"{'':<39}evidence [{ev.get('source_qa') or ev.get('source')}]: {ev['sentence'][:110]}")


# --------------------------------------------------------------------------- #
# Merge into a new ontology version
# --------------------------------------------------------------------------- #

def _find_instance(data, cls, name):
    for i, inst in enumerate(data["concept_classes"].get(cls, {}).get("instances", [])):
        n = inst if isinstance(inst, str) else inst["name"]
        if n.lower() == name.lower():
            return i, inst
    return None, None


def merge(version="2.1", base=None, store=None):
    store = store or load_gaps()
    bcmo = BCMO(base) if base else BCMO()
    data = copy.deepcopy(bcmo.data)
    now = datetime.now().isoformat()
    added = {"add_alias": 0, "add_concept": 0, "add_relation": 0}

    for gap in store["gaps"].values():
        if gap["status"] != "approved" or not gap.get("proposal"):
            continue
        p = gap["proposal"]
        prov = {"gap_id": gap["id"], "added": now, "evidence": gap.get("evidence_label")}

        if p["action"] == "add_alias":
            idx, inst = _find_instance(data, p["class"], p["concept"])
            if inst is None:
                continue
            if isinstance(inst, str):
                inst = {"name": inst}
                data["concept_classes"][p["class"]]["instances"][idx] = inst
            aliases = inst.setdefault("aliases", [])
            if p["alias"].lower() not in (a.lower() for a in aliases):
                aliases.append(p["alias"])
                inst.setdefault("gap_filler_aliases", []).append({"alias": p["alias"], **prov})

        elif p["action"] == "add_concept":
            cls = p.get("class")
            if not cls:
                print(f"  skip {gap['id']}: concept '{p['name']}' has no class assigned")
                continue
            classes = data["concept_classes"]
            if cls not in classes:
                classes[cls] = {"description": "Added by ontology gap filler", "instances": []}
            if _find_instance(data, cls, p["name"])[1] is None:
                inst = {"name": p["name"], "aliases": [gap["entity"]],
                        "source": "gap_filler", "provenance": prov}
                if p.get("misinformation_claim"):
                    inst["misinformation_claim"] = p["misinformation_claim"]
                    inst["reference"] = p.get("source")
                classes[cls]["instances"].append(inst)

        elif p["action"] == "add_relation":
            data["factual_relations"].append({
                "subject": p["subject"], "relation": p["relation"], "object": p["object"],
                "negated": p["negated"], "source": p.get("source") or "MedQuAD (gap filler)",
                "source_doc": p["source_doc"], "source_qa": p["source_qa"],
                "data_grounded": p["source_doc"] is not None, "provenance": prov,
            })

        added[p["action"]] += 1
        gap["status"] = "merged"
        gap["merged_into"] = f"bcmo_v{version}"

    meta = data["ontology_metadata"]
    meta["version"] = version
    meta["created"] = now
    meta["derived_from"] = bcmo.data["ontology_metadata"]["version"]
    meta["concept_class_count"] = len(data["concept_classes"])
    meta["total_concept_instances"] = sum(
        len(c.get("instances", [])) for c in data["concept_classes"].values())
    meta["total_factual_relations"] = len(data["factual_relations"])
    meta["data_grounded_relations"] = sum(1 for r in data["factual_relations"] if r.get("data_grounded"))
    prior = meta.get("gap_filler_additions", {}) if meta["derived_from"] == version else {}
    meta["gap_filler_additions"] = {k: prior.get(k, 0) + v for k, v in added.items()}

    out = DATA_DIR / f"bcmo_v{version}.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    save_gaps(store)
    print(f"Wrote {out.relative_to(DATA_DIR.parent)}: {added}")
    return out


# --------------------------------------------------------------------------- #
# Coverage report
# --------------------------------------------------------------------------- #

def coverage(claims, bcmo):
    linker = EntityLinker(bcmo)
    ents = [e for c in claims for e in (c.subject, c.object)]
    linked = sum(1 for e in ents if linker.link(e).match_type != "unlinked")
    verdicts = {}
    for c in claims:
        s, o = linker.link(c.subject), linker.link(c.object)
        if "unlinked" in (s.match_type, o.match_type):
            v = "unsupported"
        else:
            v, _ = bcmo.verify_claim(s.canonical_name, c.relation, o.canonical_name)
        verdicts[v] = verdicts.get(v, 0) + 1
    decided = len(claims) - verdicts.get("unsupported", 0)
    return {"entities_linked": f"{linked}/{len(ents)}",
            "claims_with_verdict": f"{decided}/{len(claims)}", "verdicts": verdicts}


def report(claims, version="2.1"):
    before = coverage(claims, BCMO(DATA_DIR / "bcmo_v2.json"))
    after_path = DATA_DIR / f"bcmo_v{version}.json"
    print(f"{'':<22}{'v2.0':<26}v{version}")
    if not after_path.exists():
        print(f"(no {after_path.name} yet -- run merge)")
        after = {k: "-" for k in before}
    else:
        after = coverage(claims, BCMO(after_path))
    for k in before:
        print(f"{k:<22}{str(before[k]):<26}{after[k]}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

SAMPLE_CLAIMS = [
    Claim("alcohol", "increases_risk_of", "Breast Cancer",
          source_text="Drinking alcohol has nothing to do with breast cancer."),
    Claim("exercise", "decreases_risk_of", "Breast Cancer",
          source_text="Exercise lowers your chance of getting breast cancer."),
    Claim("Trastuzumab", "treats", "HER2-positive",
          source_text="Herceptin is used for HER2-positive tumors."),
    Claim("vitamin C", "treats", "Breast Cancer",
          source_text="High-dose vitamin C cures breast cancer."),
    Claim("underwire bra", "causes", "Breast Cancer",
          source_text="Wearing an underwire bra causes breast cancer."),
    Claim("Obesity", "increases_risk_of", "Breast Cancer",
          source_text="Being overweight raises breast cancer risk."),
    Claim("Mammography", "detects", "Breast Cancer",
          source_text="Mammograms can find breast cancer early."),
    Claim("Radiation Therapy", "treats", "Ductal Carcinoma In Situ",
          source_text="Radiation is used after lumpectomy for DCIS."),
]


def load_claims(path):
    if path is None:
        return SAMPLE_CLAIMS
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            return [Claim(**{k: v for k, v in c.items() if k in Claim.__dataclass_fields__})
                    for c in json.load(f)]
    extractor = ClaimExtractor()
    claims = []
    for line in p.read_text().splitlines():
        if line.strip():
            claims.extend(extractor.extract(line))
    return claims


def main():
    ap = argparse.ArgumentParser(description="BCMO ontology gap filler")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("detect", "report", "demo"):
        s = sub.add_parser(name)
        s.add_argument("--claims", help="JSON claim list or text file of claim sentences")
        if name != "detect":
            s.add_argument("--version", default="2.1")
    sub.add_parser("propose")
    r = sub.add_parser("review")
    r.add_argument("--approve", nargs="*", default=[])
    r.add_argument("--reject", nargs="*", default=[])
    r.add_argument("--approve-evidenced", action="store_true",
                   help="approve every proposal backed by lexical, MedQuAD or misinformation evidence")
    m = sub.add_parser("merge")
    m.add_argument("--version", default="2.1")
    sub.add_parser("list")
    args = ap.parse_args()

    if args.cmd == "detect":
        results = detect_gaps(load_claims(args.claims))
        print(f"Checked {len(results)} claims; gaps logged to {GAPS_PATH.name}")
        print_gaps(load_gaps())
    elif args.cmd == "propose":
        print_gaps(propose())
    elif args.cmd == "review":
        print_gaps(review(args.approve, args.reject, args.approve_evidenced))
    elif args.cmd == "list":
        print_gaps(load_gaps())
    elif args.cmd == "merge":
        merge(args.version)
    elif args.cmd == "report":
        report(load_claims(args.claims), args.version)
    elif args.cmd == "demo":
        claims = load_claims(args.claims)
        if GAPS_PATH.exists():
            GAPS_PATH.unlink()
        print("== 1. detect"); detect_gaps(claims); print_gaps(load_gaps())
        print("\n== 2. propose (MedQuAD evidence)"); print_gaps(propose())
        print("\n== 3. review (auto-approve evidenced)"); review(approve_evidenced=True)
        print("\n== 4. merge"); merge(args.version)
        # Second pass: relations that were blocked by entity gaps can now be checked.
        print("\n== 5. re-propose blocked relations against new version")
        store = load_gaps()
        new = BCMO(DATA_DIR / f"bcmo_v{args.version}.json")
        propose(store, new); review(approve_evidenced=True, store=store)
        merge(args.version, base=DATA_DIR / f"bcmo_v{args.version}.json", store=store)
        print_gaps(load_gaps())
        print("\n== 6. coverage"); report(claims, args.version)


if __name__ == "__main__":
    main()
