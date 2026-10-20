"""
Entity Linker: Map extracted entity mentions to BCMO ontology concepts.

Takes raw entity strings from claim extraction and resolves them to
canonical BCMO concept names using exact matching, alias matching,
and fuzzy string similarity.
"""

import re
from typing import Optional, Tuple, List
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ontology.bcmo_queries import BCMO


@dataclass
class LinkedEntity:
    original: str
    canonical_name: str
    concept_class: str
    confidence: float  # 0.0 to 1.0
    match_type: str  # "exact", "alias", "fuzzy", "unlinked"

    def __repr__(self):
        return f"{self.original} -> {self.canonical_name} [{self.concept_class}] ({self.match_type}, {self.confidence:.2f})"


def _normalize(text):
    """Normalize text for matching."""
    t = text.lower().strip()
    t = re.sub(r'[^\w\s/-]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def _levenshtein_ratio(s1, s2):
    """Compute similarity ratio between two strings (0.0 to 1.0)."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    distance = matrix[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


class EntityLinker:
    """Link entity mentions to BCMO ontology concepts."""

    def __init__(self, bcmo: BCMO = None, fuzzy_threshold: float = 0.75):
        self.bcmo = bcmo or BCMO()
        self.fuzzy_threshold = fuzzy_threshold
        self._build_concept_map()

    def _build_concept_map(self):
        """Build a mapping of all concept names/aliases for matching."""
        self.concept_map = {}  # normalized_name -> (canonical_name, class_name)
        self.all_names = []

        for cls_name, cls_data in self.bcmo.concepts.items():
            for inst in cls_data.get("instances", []):
                if isinstance(inst, str):
                    norm = _normalize(inst)
                    self.concept_map[norm] = (inst, cls_name)
                    self.all_names.append((norm, inst, cls_name))
                elif isinstance(inst, dict):
                    name = inst["name"]
                    norm = _normalize(name)
                    self.concept_map[norm] = (name, cls_name)
                    self.all_names.append((norm, name, cls_name))
                    for alias in inst.get("aliases", []):
                        alias_norm = _normalize(alias)
                        self.concept_map[alias_norm] = (name, cls_name)
                        self.all_names.append((alias_norm, name, cls_name))

    def link(self, entity: str) -> LinkedEntity:
        """Link a single entity mention to a BCMO concept."""
        norm = _normalize(entity)

        # Exact match
        if norm in self.concept_map:
            canonical, cls = self.concept_map[norm]
            return LinkedEntity(
                original=entity, canonical_name=canonical,
                concept_class=cls, confidence=1.0, match_type="exact",
            )

        # Substring match (entity is part of a concept name or vice versa)
        for stored_norm, canonical, cls in self.all_names:
            if norm in stored_norm or stored_norm in norm:
                overlap = min(len(norm), len(stored_norm)) / max(len(norm), len(stored_norm))
                if overlap > 0.5:
                    return LinkedEntity(
                        original=entity, canonical_name=canonical,
                        concept_class=cls, confidence=0.85, match_type="alias",
                    )

        # Fuzzy match
        best_score = 0.0
        best_match = None
        for stored_norm, canonical, cls in self.all_names:
            score = _levenshtein_ratio(norm, stored_norm)
            if score > best_score:
                best_score = score
                best_match = (canonical, cls)

        if best_score >= self.fuzzy_threshold and best_match:
            return LinkedEntity(
                original=entity, canonical_name=best_match[0],
                concept_class=best_match[1], confidence=best_score,
                match_type="fuzzy",
            )

        # Unlinked
        return LinkedEntity(
            original=entity, canonical_name=entity,
            concept_class="Unknown", confidence=0.0, match_type="unlinked",
        )

    def link_all(self, entities: List[str]) -> List[LinkedEntity]:
        """Link a list of entity mentions."""
        return [self.link(e) for e in entities]

    def link_claim(self, claim):
        """Link subject and object of a Claim to BCMO concepts."""
        subj = self.link(claim.subject)
        obj = self.link(claim.object)
        return subj, obj


def demo():
    """Demonstrate entity linking on sample mentions."""
    linker = EntityLinker()

    test_entities = [
        "trastuzumab", "Herceptin", "herceptin",
        "chemo", "chemotherapy", "Chemotherapy",
        "HER2-positive", "HER2+", "HER2/neu",
        "triple negative", "TNBC", "triple-negative breast cancer",
        "tamoxifen", "Tamoxifen",
        "mammogram", "mammography",
        "BRCA1", "BRCA mutation",
        "lump", "breast lump",
        "hair loss", "alopecia",
        "heart damage", "cardiotoxicity",
        "exercise", "regular exercise",
        "Stage IV", "metastatic",
        "vitamin C",  # should be unlinked
        "underwire bra",  # should be unlinked
        "alcohol", "drinking alcohol",
        "radiation therapy", "radiotherapy",
        "palbociclib", "Ibrance",
    ]

    print("=" * 70)
    print("ENTITY LINKING DEMO")
    print("=" * 70)

    linked_count = 0
    for entity in test_entities:
        result = linker.link(entity)
        status = "LINKED" if result.match_type != "unlinked" else "UNLINKED"
        if result.match_type != "unlinked":
            linked_count += 1
        print(f"  {status:>8} | {result}")

    print(f"\nLinked: {linked_count}/{len(test_entities)} "
          f"({linked_count/len(test_entities)*100:.0f}%)")


if __name__ == "__main__":
    demo()
