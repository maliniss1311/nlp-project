"""
Query utilities for the Breast Cancer Misinformation Ontology (BCMO).

Provides functions to look up concepts, verify claims against the ontology,
and find related entities -- used by the claim verification pipeline.
"""

import json
from pathlib import Path
from collections import defaultdict


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class BCMO:
    """Interface for querying the Breast Cancer Misinformation Ontology."""

    def __init__(self, path=None):
        path = path or (DATA_DIR / "bcmo_v2.json")
        if not path.exists():
            path = DATA_DIR / "bcmo.json"
        with open(path) as f:
            self.data = json.load(f)

        self.concepts = self.data["concept_classes"]
        self.relations = self.data["relation_types"]
        self.modifiers = self.data["modifiers"]
        self.facts = self.data["factual_relations"]
        self.misinfo = self.data["common_misinformation"]

        # Build lookup indexes
        self._name_index = {}  # lowercase name -> (class, instance)
        self._alias_index = {}  # lowercase alias -> (class, instance_name)
        self._build_indexes()

    def _build_indexes(self):
        for cls_name, cls_data in self.concepts.items():
            for inst in cls_data.get("instances", []):
                if isinstance(inst, str):
                    self._name_index[inst.lower()] = (cls_name, inst)
                elif isinstance(inst, dict):
                    name = inst["name"]
                    self._name_index[name.lower()] = (cls_name, name)
                    for alias in inst.get("aliases", []):
                        self._alias_index[alias.lower()] = (cls_name, name)

    def lookup(self, term):
        """Find a concept by name or alias. Returns (class_name, instance_name) or None."""
        t = term.lower().strip()
        if t in self._name_index:
            return self._name_index[t]
        if t in self._alias_index:
            return self._alias_index[t]
        # Partial match
        for key, val in self._name_index.items():
            if t in key or key in t:
                return val
        for key, val in self._alias_index.items():
            if t in key or key in t:
                return val
        return None

    def get_instance(self, name):
        """Get full instance data for a concept by name."""
        for cls_data in self.concepts.values():
            for inst in cls_data.get("instances", []):
                if isinstance(inst, dict) and inst["name"].lower() == name.lower():
                    return inst
                elif isinstance(inst, str) and inst.lower() == name.lower():
                    return {"name": inst}
        return None

    def find_relations(self, subject=None, relation=None, obj=None):
        """Query factual relations by subject, relation type, and/or object."""
        results = []
        for fact in self.facts:
            if subject and fact["subject"].lower() != subject.lower():
                continue
            if relation and fact["relation"] != relation:
                continue
            if obj and fact["object"].lower() != obj.lower():
                continue
            results.append(fact)
        return results

    def get_treatments_for(self, condition):
        """Find all treatments for a given condition/subtype."""
        return self.find_relations(relation="treats", obj=condition)

    def get_side_effects_of(self, treatment):
        """Find all side effects of a given treatment."""
        return self.find_relations(subject=treatment, relation="has_side_effect")

    def get_risk_factors_for(self, condition="Breast Cancer"):
        """Find all risk factors for a condition."""
        return self.find_relations(relation="increases_risk_of", obj=condition)

    def get_contraindications(self, treatment=None, condition=None):
        """Find contraindications for a treatment or condition."""
        return self.find_relations(
            subject=treatment, relation="contraindicated_for", obj=condition
        )

    def verify_claim(self, subject, relation, obj):
        """Verify a claim triple against the ontology.
        
        Returns one of:
            - "supported": claim matches a factual relation
            - "contradicted": ontology contains evidence against the claim
            - "unsupported": no relevant information found
            - "misleading": partially true but missing important context
        """
        # Direct match
        direct = self.find_relations(subject=subject, relation=relation, obj=obj)
        if direct:
            for fact in direct:
                if fact.get("negated"):
                    return "contradicted", fact
            return "supported", direct[0]

        # Check for negation of the claim
        negated_relations = {
            "treats": "contraindicated_for",
            "contraindicated_for": "treats",
            "increases_risk_of": "decreases_risk_of",
            "decreases_risk_of": "increases_risk_of",
        }
        if relation in negated_relations:
            counter = self.find_relations(
                subject=subject, relation=negated_relations[relation], obj=obj
            )
            if counter:
                return "contradicted", counter[0]

        # Check if subject/object exist in ontology but relation doesn't
        subj_exists = self.lookup(subject)
        obj_exists = self.lookup(obj)
        if subj_exists and obj_exists:
            # Entities known, but this specific relation not recorded
            related = self.find_relations(subject=subject)
            if related:
                return "unsupported", {
                    "note": f"'{subject}' has known relations but not '{relation}' with '{obj}'"
                }

        return "unsupported", None

    def check_misinformation(self, text):
        """Check if text matches known misinformation patterns."""
        text_lower = text.lower()
        matches = []
        for item in self.misinfo:
            claim_words = item["claim"].lower().split()
            match_count = sum(1 for w in claim_words if w in text_lower)
            if match_count / len(claim_words) > 0.5:
                matches.append(item)
        return matches

    def get_all_concepts_flat(self):
        """Return a flat list of all concept names for entity linking."""
        names = set()
        for cls_data in self.concepts.values():
            for inst in cls_data.get("instances", []):
                if isinstance(inst, str):
                    names.add(inst)
                elif isinstance(inst, dict):
                    names.add(inst["name"])
                    for alias in inst.get("aliases", []):
                        names.add(alias)
        return sorted(names)

    def summary(self):
        """Print a summary of the ontology."""
        meta = self.data.get("ontology_metadata", {})
        total = sum(len(v.get("instances", [])) for v in self.concepts.values())
        version = meta.get("version", "?")
        print(f"BCMO v{version}")
        print(f"  Concept classes: {len(self.concepts)}")
        print(f"  Total instances: {total}")
        print(f"  Relation types: {len(self.relations)}")
        print(f"  Factual relations: {len(self.facts)}")
        grounded = sum(1 for f in self.facts if f.get("data_grounded"))
        expert = len(self.facts) - grounded
        print(f"    Expert-curated: {expert}")
        print(f"    MedQuAD-grounded: {grounded}")
        print(f"  Known misinformation: {len(self.misinfo)}")
        print(f"  Modifiers: {len(self.modifiers)}")

    def get_provenance(self, subject, relation, obj):
        """Get source provenance for a specific relation triple."""
        for fact in self.facts:
            if (fact["subject"].lower() == subject.lower()
                    and fact["relation"] == relation
                    and fact["object"].lower() == obj.lower()):
                return {
                    "data_grounded": fact.get("data_grounded", False),
                    "source": fact.get("source", "N/A"),
                    "source_doc": fact.get("source_doc", "N/A"),
                    "source_qa": fact.get("source_qa", "N/A"),
                }
        return None


def demo():
    """Demonstrate BCMO query capabilities."""
    bcmo = BCMO()
    bcmo.summary()

    print("\n--- Concept Lookup ---")
    for term in ["trastuzumab", "HER2", "TNBC", "chemo", "BRCA1", "mammogram"]:
        result = bcmo.lookup(term)
        print(f"  '{term}' -> {result}")

    print("\n--- Treatments for HER2-Positive Breast Cancer ---")
    for r in bcmo.get_treatments_for("HER2-Positive Breast Cancer"):
        print(f"  {r['subject']} (strength: {r.get('strength', 'N/A')})")

    print("\n--- Side Effects of Chemotherapy ---")
    for r in bcmo.get_side_effects_of("Chemotherapy"):
        print(f"  {r['object']} (frequency: {r.get('strength', 'N/A')})")

    print("\n--- Claim Verification Examples ---")
    claims = [
        ("Trastuzumab", "treats", "HER2-Positive Breast Cancer"),
        ("Endocrine Therapy", "treats", "Triple-Negative Breast Cancer"),
        ("Chemotherapy", "treats", "HER2-Positive Breast Cancer"),
        ("Vitamin C", "treats", "Breast Cancer"),
    ]
    for subj, rel, obj in claims:
        verdict, evidence = bcmo.verify_claim(subj, rel, obj)
        note = f" -- {evidence.get('note', '')}" if evidence and isinstance(evidence, dict) and evidence.get('note') else ""
        print(f"  ({subj}, {rel}, {obj}) -> {verdict}{note}")

    print("\n--- Misinformation Check ---")
    test_texts = [
        "Wearing underwire bras causes breast cancer",
        "Chemotherapy is the standard treatment for early breast cancer",
        "If no one in my family had breast cancer I won't get it",
    ]
    for text in test_texts:
        matches = bcmo.check_misinformation(text)
        if matches:
            for m in matches:
                print(f"  '{text[:50]}...' -> {m['verdict']}: {m['explanation'][:80]}...")
        else:
            print(f"  '{text[:50]}...' -> No known misinformation match")

    print("\n--- Provenance Tracking ---")
    provenance_tests = [
        ("Trastuzumab", "treats", "HER2-Positive Breast Cancer"),
        ("Chemotherapy", "treats", "Breast Cancer"),
        ("Family History", "increases_risk_of", "Breast Cancer"),
    ]
    for subj, rel, obj in provenance_tests:
        p = bcmo.get_provenance(subj, rel, obj)
        if p:
            src_type = "MedQuAD" if p["data_grounded"] else "Expert"
            src = p.get("source", p.get("source_doc", "N/A"))
            print(f"  ({subj}, {rel}, {obj}) -> [{src_type}] source: {src}")
        else:
            print(f"  ({subj}, {rel}, {obj}) -> No provenance found")


if __name__ == "__main__":
    demo()
