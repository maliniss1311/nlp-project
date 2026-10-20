# Topic 4: Ontology-Grounded LLMs for Detecting Breast Cancer Misinformation in Patient Conversations

## Problem Statement

Patients increasingly turn to online communities and LLMs for cancer-related information. These discussions often contain misconceptions expressed indirectly—embedded in personal experiences or as questions with false presuppositions. Current LLMs frequently fail to identify and correct these underlying misconceptions, and making models more cautious increases false positives on benign questions.

## Central Hypothesis

Explicit ontology grounding can make LLM-based misinformation detection more accurate, robust, and explainable than relying on the LLM's parametric knowledge or retrieval alone.

## Approach: Structured Claim Verification

Rather than asking an LLM "is this true or false?", the system will:
1. Convert patient statements into structured medical claims (subject-relation-object triples)
2. Ground those claims against a purpose-built Breast Cancer Misinformation Ontology (BCMO)
3. Verify each claim against ontology constraints and retrieved evidence

**Example:** "My tumor is HER2-positive, so chemotherapy won't work for me" decomposes into claims including `HER2-positive → incompatible_with → chemotherapy`, which the ontology can flag as misleading (both therapies are often used together).

## Research Questions

| RQ | Question |
|----|----------|
| 1 | Does domain fine-tuning improve concept/relation extraction from patient text? |
| 2 | Does ontology grounding improve misinformation detection over LLM-only? |
| 3 | Does the ontology provide value beyond standard RAG? |
| 4 | Does grounding improve robustness to paraphrases and implicit misinformation? |
| 5 | Does grounding produce more faithful, evidence-consistent explanations? |

## Proposed Ontology (BCMO)

- **Concept classes:** Cancer subtypes, biomarkers (ER/PR/HER2), treatments, medications, side effects, risk factors, genetic factors, staging, metastasis, prognosis, prevention/screening
- **Relations:** causes, treats, prevents, contraindicated_for, increases_risk_of, has_side_effect, improves_survival, etc.
- **Key feature:** Represents negation, uncertainty, and relation strength to distinguish probabilistic claims ("can reduce risk") from absolute claims ("always prevents recurrence")

## Data Resources

| Dataset | Size | Role |
|---------|------|------|
| Breastcancer.org posts | ~362K posts | Domain adaptation / LLM fine-tuning |
| MedQuAD (Cancer.gov subset) | 47,457 QA pairs | Concept normalization, ontology construction |
| CancerMyth | 585 + 150 questions | Adversarial evaluation benchmark |
| Expert-annotated Reddit cancer posts | Breast cancer subset | Naturalistic evaluation with multi-dimensional labels |

## Why This Matters

- Moves beyond binary "true/false" classification to structured, explainable claim verification
- Addresses a real clinical need: patients acting on misinformation about treatment decisions
- Provides a reusable ontology artifact for breast cancer NLP research
- Directly compares ontology grounding vs. RAG vs. LLM-only, filling a gap in current literature
