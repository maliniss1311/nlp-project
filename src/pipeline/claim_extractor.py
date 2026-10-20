"""
Claim Extractor: Decompose patient text into structured medical claim triples.

Extracts (subject, relation, object, strength, negation) from free-text patient
statements about breast cancer. Two approaches:
  1. Rule-based: regex + dependency patterns (no external dependencies)
  2. LLM-based: prompted extraction via OpenAI API (requires `pip install openai`)

The claim format follows the BCMO relation schema.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


BCMO_RELATIONS = [
    "treats", "prevents", "increases_risk_of", "decreases_risk_of",
    "detects", "has_biomarker", "has_side_effect", "contraindicated_for",
    "associated_with", "causes", "reduces_recurrence_risk", "improves_survival",
]


@dataclass
class Claim:
    subject: str
    relation: str
    object: str
    negated: bool = False
    strength: str = "moderate"  # strong, moderate, weak
    certainty: str = "definite"  # definite, probable, possible, uncertain
    source_text: str = ""
    extraction_method: str = "rule_based"

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        neg = "NOT " if self.negated else ""
        return f"({self.subject}, {neg}{self.relation}, {self.object})"


# --- Linguistic patterns for relation detection ---

TREATMENT_VERBS = [
    r"treat(?:s|ed|ing)?", r"cur(?:e[sd]?|ing)", r"help(?:s|ed)?",
    r"work(?:s|ed|ing)?(?:\s+for)?", r"(?:is|are)\s+(?:used|effective|prescribed)\s+for",
    r"fight(?:s|ing)?", r"combat(?:s|ing)?", r"manage(?:s|d)?",
]

PREVENTION_VERBS = [
    r"prevent(?:s|ed|ing)?", r"protect(?:s|ed|ing)?\s+against",
    r"reduce[sd]?\s+(?:the\s+)?(?:risk|chance|likelihood)",
    r"lower[sd]?\s+(?:the\s+)?(?:risk|chance)",
    r"decrease[sd]?\s+(?:the\s+)?(?:risk|chance)",
]

RISK_VERBS = [
    r"(?:increase[sd]?|raise[sd]?|elevate[sd]?)\s+(?:the\s+)?(?:risk|chance|likelihood)",
    r"(?:cause[sd]?|lead[sd]?\s+to|result[sd]?\s+in)",
    r"(?:is|are)\s+(?:a\s+)?risk\s+factor",
    r"(?:is|are)\s+(?:linked|associated|connected|related)\s+to",
    r"(?:contribute[sd]?\s+to)",
]

SIDE_EFFECT_VERBS = [
    r"(?:cause[sd]?|lead[sd]?\s+to|result[sd]?\s+in)",
    r"(?:side\s+effect[sd]?\s+(?:of|include|are))",
    r"(?:make[sd]?\s+(?:you|me|patients?)\s+(?:feel|sick|tired|nauseous))",
    r"(?:can|may|might)\s+(?:cause|lead\s+to|result\s+in)",
]

NEGATION_PATTERNS = [
    r"\b(?:not|n't|never|no|won't|wouldn't|can't|cannot|doesn't|don't|isn't|aren't)\b",
    r"\b(?:without|lack(?:s|ing)?|absence\s+of|fail(?:s|ed)?\s+to)\b",
    r"\b(?:unlikely|impossible|unable)\b",
]

CERTAINTY_MARKERS = {
    "definite": [r"\balways\b", r"\bdefinitely\b", r"\bcertainly\b", r"\bproven\b"],
    "probable": [r"\busually\b", r"\btypically\b", r"\bgenerally\b", r"\boften\b", r"\blikely\b"],
    "possible": [r"\bsometimes\b", r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibly\b"],
    "uncertain": [r"\brarely\b", r"\bunlikely\b", r"\bdoubtful\b", r"\bunclear\b"],
}

# Medical entity patterns
TREATMENT_ENTITIES = [
    "chemotherapy", "chemo", "radiation", "radiation therapy", "radiotherapy",
    "surgery", "mastectomy", "lumpectomy", "targeted therapy",
    "hormone therapy", "endocrine therapy", "immunotherapy",
    "trastuzumab", "herceptin", "pertuzumab", "perjeta",
    "tamoxifen", "anastrozole", "letrozole", "exemestane",
    "palbociclib", "ibrance", "pembrolizumab", "keytruda",
    "olaparib", "lynparza", "neoadjuvant", "adjuvant",
    "stem cell transplant", "bone marrow transplant",
    "aromatase inhibitor", "CDK4/6 inhibitor", "PARP inhibitor",
]

CONDITION_ENTITIES = [
    "breast cancer", "triple-negative breast cancer", "TNBC",
    "HER2-positive breast cancer", "HER2-positive", "HER2+",
    "hormone receptor-positive", "HR-positive", "HR+",
    "ER-positive", "PR-positive", "inflammatory breast cancer",
    "metastatic breast cancer", "DCIS", "invasive ductal carcinoma",
    "invasive lobular carcinoma", "stage IV", "stage III",
    "recurrence", "metastasis",
]

SIDE_EFFECT_ENTITIES = [
    "nausea", "vomiting", "hair loss", "alopecia", "fatigue",
    "neuropathy", "numbness", "tingling", "lymphedema",
    "hot flashes", "joint pain", "bone loss", "osteoporosis",
    "cardiotoxicity", "heart damage", "skin irritation",
    "chemo brain", "brain fog", "infertility", "neutropenia",
    "weight gain", "depression", "anxiety",
]

RISK_FACTOR_ENTITIES = [
    "BRCA1", "BRCA2", "BRCA mutation", "family history",
    "obesity", "alcohol", "smoking", "age", "dense breast",
    "hormone replacement therapy", "HRT", "radiation exposure",
    "early menstruation", "late menopause",
]

PROTECTIVE_ENTITIES = [
    "exercise", "breastfeeding", "mammogram", "screening",
    "prophylactic mastectomy", "genetic testing", "healthy diet",
]


def _find_entities(text, entity_list):
    """Find all matching entities in text using word-boundary matching."""
    found = []
    text_lower = text.lower()
    for entity in sorted(entity_list, key=len, reverse=True):
        pattern = r'\b' + re.escape(entity.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(entity)
    return found


def _detect_negation(text):
    """Check if the text contains negation."""
    for pat in NEGATION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _detect_certainty(text):
    """Detect the certainty level of a statement."""
    for level in ["definite", "probable", "possible", "uncertain"]:
        for pat in CERTAINTY_MARKERS[level]:
            if re.search(pat, text, re.IGNORECASE):
                return level
    return "definite"


def _split_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)', text)
    return [s.strip() for s in sentences if s.strip()]


class RuleBasedExtractor:
    """Extract claims using regex patterns and entity matching."""

    def extract(self, text: str) -> List[Claim]:
        claims = []
        sentences = _split_sentences(text)

        for sent in sentences:
            claims.extend(self._extract_from_sentence(sent))

        # Deduplicate
        seen = set()
        unique = []
        for c in claims:
            key = (c.subject.lower(), c.relation, c.object.lower(), c.negated)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def _extract_from_sentence(self, sent: str) -> List[Claim]:
        claims = []
        negated = _detect_negation(sent)
        certainty = _detect_certainty(sent)

        treatments = _find_entities(sent, TREATMENT_ENTITIES)
        conditions = _find_entities(sent, CONDITION_ENTITIES)
        side_effects = _find_entities(sent, SIDE_EFFECT_ENTITIES)
        risk_factors = _find_entities(sent, RISK_FACTOR_ENTITIES)
        protective = _find_entities(sent, PROTECTIVE_ENTITIES)

        # Treatment-condition claims
        for verb_pats in [TREATMENT_VERBS]:
            for pat in verb_pats:
                if re.search(pat, sent, re.IGNORECASE):
                    for t in treatments:
                        for c in conditions:
                            claims.append(Claim(
                                subject=t, relation="treats", object=c,
                                negated=negated, certainty=certainty,
                                source_text=sent,
                            ))
                    break

        # Prevention claims
        for pat in PREVENTION_VERBS:
            if re.search(pat, sent, re.IGNORECASE):
                subjects = treatments + protective + risk_factors
                objects = conditions or ["breast cancer"]
                for s in subjects:
                    for o in objects:
                        claims.append(Claim(
                            subject=s, relation="decreases_risk_of", object=o,
                            negated=negated, certainty=certainty,
                            source_text=sent,
                        ))
                break

        # Risk factor claims -- only when risk verbs are near risk factor entities
        for pat in RISK_VERBS:
            if re.search(pat, sent, re.IGNORECASE):
                subjects = risk_factors
                if not subjects:
                    break  # don't guess subjects for risk claims
                objects = conditions or ["breast cancer"]
                for s in subjects:
                    for o in objects:
                        rel = "increases_risk_of"
                        if negated:
                            rel = "decreases_risk_of"
                            negated_claim = False
                        else:
                            negated_claim = False
                        claims.append(Claim(
                            subject=s, relation=rel, object=o,
                            negated=negated_claim, certainty=certainty,
                            source_text=sent,
                        ))
                break

        # Side effect claims -- only if both treatment and side effect entities found
        if treatments and side_effects:
            for pat in SIDE_EFFECT_VERBS:
                if re.search(pat, sent, re.IGNORECASE):
                    for t in treatments:
                        for se in side_effects:
                            claims.append(Claim(
                                subject=t, relation="has_side_effect", object=se,
                                negated=negated, certainty=certainty,
                                source_text=sent,
                            ))
                    break

        # Catch-all: if entities found but no relation detected, use associated_with
        if not claims:
            all_entities = treatments + conditions + side_effects + risk_factors + protective
            if len(all_entities) >= 2:
                claims.append(Claim(
                    subject=all_entities[0],
                    relation="associated_with",
                    object=all_entities[1],
                    negated=negated,
                    certainty=certainty,
                    source_text=sent,
                ))

        return claims


class LLMExtractor:
    """Extract claims using an LLM (OpenAI API).
    
    Requires: pip install openai
    Set environment variable OPENAI_API_KEY or pass api_key to constructor.
    """

    SYSTEM_PROMPT = """You are a medical claim extraction system specializing in breast cancer.
Given patient text, extract structured claims as JSON triples.

Each claim should have:
- subject: the medical entity making/receiving an action
- relation: one of [treats, prevents, increases_risk_of, decreases_risk_of, detects, has_biomarker, has_side_effect, contraindicated_for, associated_with, causes, reduces_recurrence_risk, improves_survival]
- object: the target entity
- negated: true if the claim is negated (e.g., "does NOT work")
- strength: strong/moderate/weak
- certainty: definite/probable/possible/uncertain

Focus on extracting factual claims about:
- Treatment effectiveness for specific cancer subtypes
- Side effects of treatments
- Risk factors for breast cancer
- Diagnostic relationships
- Contraindications

Return ONLY a JSON array of claim objects. If no medical claims found, return [].
"""

    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: openai not installed. Use `pip install openai` for LLM extraction.")

    def extract(self, text: str) -> List[Claim]:
        if not self.available:
            print("LLM extractor unavailable. Falling back to rule-based.")
            return RuleBasedExtractor().extract(text)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract medical claims from this patient text:\n\n{text}"},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        try:
            result = json.loads(response.choices[0].message.content)
            claims_data = result if isinstance(result, list) else result.get("claims", [])
            claims = []
            for c in claims_data:
                claims.append(Claim(
                    subject=c.get("subject", ""),
                    relation=c.get("relation", "associated_with"),
                    object=c.get("object", ""),
                    negated=c.get("negated", False),
                    strength=c.get("strength", "moderate"),
                    certainty=c.get("certainty", "definite"),
                    source_text=text[:200],
                    extraction_method="llm",
                ))
            return claims
        except (json.JSONDecodeError, KeyError) as e:
            print(f"LLM extraction parse error: {e}")
            return []


class ClaimExtractor:
    """Unified claim extractor that can use rule-based, LLM, or both methods."""

    def __init__(self, method="rule_based", api_key=None):
        """
        Args:
            method: "rule_based", "llm", or "ensemble" (both, deduplicated)
            api_key: OpenAI API key (only needed for llm/ensemble)
        """
        self.method = method
        self.rule_extractor = RuleBasedExtractor()
        self.llm_extractor = LLMExtractor(api_key=api_key) if method in ("llm", "ensemble") else None

    def extract(self, text: str) -> List[Claim]:
        if self.method == "rule_based":
            return self.rule_extractor.extract(text)
        elif self.method == "llm":
            return self.llm_extractor.extract(text)
        elif self.method == "ensemble":
            rule_claims = self.rule_extractor.extract(text)
            llm_claims = self.llm_extractor.extract(text)
            return self._merge(rule_claims, llm_claims)
        return []

    def _merge(self, claims_a, claims_b):
        seen = set()
        merged = []
        for c in claims_a + claims_b:
            key = (c.subject.lower(), c.relation, c.object.lower())
            if key not in seen:
                seen.add(key)
                merged.append(c)
        return merged


def demo():
    """Demonstrate claim extraction on sample patient statements."""
    extractor = ClaimExtractor(method="rule_based")

    test_cases = [
        "My tumor is HER2-positive, so chemotherapy won't work for me and I need targeted therapy instead.",
        "I heard that BRCA1 mutations cause breast cancer and that tamoxifen can prevent it.",
        "Chemotherapy causes terrible nausea and hair loss. I don't want to go through that.",
        "My doctor said radiation therapy is effective for treating my breast cancer after lumpectomy.",
        "I read that exercise can reduce the risk of breast cancer recurrence.",
        "Wearing underwire bras increases the risk of breast cancer.",
        "Triple-negative breast cancer can't be treated with hormone therapy because it's ER-negative.",
        "Trastuzumab is used to treat HER2-positive breast cancer but it can cause heart damage.",
        "My oncologist prescribed palbociclib for my hormone receptor-positive metastatic breast cancer.",
        "I'm worried that drinking alcohol increases my risk of getting breast cancer.",
    ]

    print("=" * 70)
    print("CLAIM EXTRACTION DEMO (Rule-Based)")
    print("=" * 70)

    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Example {i} ---")
        print(f"Input: {text}")
        claims = extractor.extract(text)
        if claims:
            for c in claims:
                neg = "NOT " if c.negated else ""
                print(f"  -> ({c.subject}, {neg}{c.relation}, {c.object}) "
                      f"[certainty={c.certainty}]")
        else:
            print("  -> No claims extracted")

    print(f"\n{'=' * 70}")
    print(f"Total: {sum(len(extractor.extract(t)) for t in test_cases)} claims from {len(test_cases)} inputs")


if __name__ == "__main__":
    demo()
