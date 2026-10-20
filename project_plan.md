# Project Plan
## Ontology-Grounded LLMs for Detecting Breast Cancer Misinformation in Patient Conversations
### NLP Course Assignment

---

## 1. Project Overview

**Objective:** Build a system that detects misinformation in breast cancer patient conversations by decomposing patient statements into structured claims and verifying them against a purpose-built Breast Cancer Misinformation Ontology (BCMO).

**Central Hypothesis:** Explicit ontology grounding makes LLM-based misinformation detection more accurate, robust, and explainable than relying on the LLM's parametric knowledge or retrieval alone.

**Deliverables:**
1. Breast Cancer Misinformation Ontology (BCMO) in JSON/OWL format
2. Claim decomposition pipeline (patient text → structured triples)
3. Ontology-grounded verification system
4. Comparative evaluation: Ontology-grounded vs. LLM-only vs. RAG
5. Final paper with results and analysis

---

## 2. Research Questions

| RQ | Question | How We Answer It |
|----|----------|------------------|
| RQ1 | Does domain fine-tuning improve concept/relation extraction? | Compare base vs. fine-tuned LLM on claim extraction task |
| RQ2 | Does ontology grounding beat LLM-only? | Compare system with/without BCMO on misinfo detection |
| RQ3 | Does ontology provide value beyond RAG? | Compare ontology-grounded vs. standard RAG retrieval |
| RQ4 | Does grounding improve robustness? | Test on paraphrased/implicit misinformation examples |
| RQ5 | Does grounding produce faithful explanations? | Human evaluation of explanation quality |

---

## 3. Data Sources & Access Plan

### 3.1 MedQuAD — Cancer.gov Subset (PRIMARY DATA) ✅ Available Now
- **What:** 116 XML files with cancer QA pairs, UMLS CUIs, semantic types, question types
- **Location:** `MedQuAD/1_CancerGov_QA/` (already cloned)
- **License:** CC BY 4.0
- **Use:** Ontology concept extraction, domain adaptation fine-tuning, claim templates

### 3.2 Reddit r/breastcancer Posts ✅ Accessible via API
- **What:** Patient-generated discussions about breast cancer (informal language)
- **Access:** Reddit API (free developer account)
- **Use:** Domain language adaptation, test examples of patient misinformation
- **Plan:** Collect 5,000-10,000 posts from r/breastcancer, r/cancer, r/HER2positive

### 3.3 Cancer-Myth Benchmark ⚠️ Request from Authors
- **What:** 585 cancer questions with false presuppositions + 150 controls
- **Paper:** Zhu et al., arXiv:2504.11373 (USC)
- **Use:** Adversarial evaluation benchmark
- **Fallback:** If unavailable, construct 50-100 breast-cancer-specific questions with false presuppositions ourselves (using MedQuAD knowledge + clinical guidelines)

### 3.4 NCI Thesaurus / UMLS ✅ Free with License
- **What:** Structured medical terminology, breast cancer concepts, treatment hierarchies
- **Access:** NLM UMLS license (free, 1-2 day approval)
- **Use:** Foundation for BCMO ontology construction

### 3.5 Self-Constructed Evaluation Set (OUR CONTRIBUTION)
- **What:** 100-200 breast cancer patient statements annotated for misinformation
- **Source:** Derived from Reddit posts + paraphrased MedQuAD content + synthetic examples
- **Annotation:** Binary (misinfo/not) + claim-level labels + explanation

---

## 4. System Architecture

```
Patient Statement
       │
       ▼
┌──────────────────┐
│ Claim Decomposer │  (LLM extracts subject-relation-object triples)
│  (Fine-tuned)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Entity Linker    │  (Maps entities to BCMO concepts)
│  (ScispaCy/UMLS) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Ontology Verifier│  (Checks claims against BCMO relations)
│  (BCMO + Rules)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Evidence Scorer  │  (Scores: supported / contradicted / unsupported)
│  + Explainer     │
└──────────────────┘
         │
         ▼
   VERDICT + EXPLANATION
   (e.g., "MISLEADING: HER2-targeted therapy and chemotherapy
    are often used together, not as alternatives")
```

---

## 5. Implementation Plan (Phase by Phase)

### Phase 1: Data Preparation & Ontology Construction
**Goal:** Build BCMO and prepare all datasets

**Tasks:**
- [ ] Parse MedQuAD Cancer.gov XML files → extract breast-cancer concepts, relations, treatments
- [ ] Collect Reddit r/breastcancer posts via API (5K-10K posts)
- [ ] Apply for UMLS license, download NCI Thesaurus breast cancer subset
- [ ] Construct BCMO v1.0:
  - Define concept classes (16 classes from proposal)
  - Define relation types (12 relations from proposal)
  - Populate with breast cancer concepts from MedQuAD + NCI
  - Add negation/uncertainty/strength modifiers
- [ ] Create evaluation dataset: 100-200 annotated patient statements

**Output:** `bcmo.json` (ontology), `eval_dataset.csv` (annotated test set), `reddit_posts.jsonl` (patient language)

### Phase 2: Claim Decomposition Pipeline
**Goal:** Convert patient text into structured claims

**Tasks:**
- [ ] Design claim format: `(subject, relation, object, strength, negation)`
  - Example: `("HER2-positive", "incompatible_with", "chemotherapy", "absolute", false)` → MISLEADING
- [ ] Implement prompted claim extraction using base LLM (Llama-3-8B or GPT-4 API)
- [ ] Fine-tune a smaller model on MedQuAD-derived claim extraction examples (LoRA)
- [ ] Evaluate: precision/recall of extracted claims vs. manual annotation
- [ ] Compare base vs. fine-tuned model (answers RQ1)

**Output:** `claim_extractor.py`, fine-tuned model weights, extraction evaluation results

### Phase 3: Ontology-Grounded Verification
**Goal:** Build the core verification system

**Tasks:**
- [ ] Implement entity linking (ScispaCy or QuickUMLS → BCMO nodes)
- [ ] Implement claim verification logic:
  - Claim matches BCMO relation → SUPPORTED
  - Claim contradicts BCMO relation → CONTRADICTED (misinformation)
  - Claim has no BCMO match → UNSUPPORTED (flag for review)
  - Claim overstates strength ("always" vs "can") → MISLEADING
- [ ] Build RAG baseline (retrieve relevant PubMed/Cancer.gov passages, ask LLM to verify)
- [ ] Build LLM-only baseline (ask LLM directly without ontology or retrieval)
- [ ] Compare all three systems on evaluation set (answers RQ2, RQ3)

**Output:** `ontology_verifier.py`, `rag_baseline.py`, `llm_baseline.py`, comparison results

### Phase 4: Robustness & Explanation Evaluation
**Goal:** Test robustness and explanation quality

**Tasks:**
- [ ] Create paraphrased test set (same misinformation, different wording)
- [ ] Create implicit misinformation examples (false presuppositions in questions)
- [ ] Create counterfactual claims (swap entities to test if model relies on shortcuts)
- [ ] Evaluate all three systems on robustness set (answers RQ4)
- [ ] Human evaluation of explanation faithfulness:
  - Does the explanation cite the correct ontology relation?
  - Is the explanation consistent with the verdict?
  - Rate 1-5 on clarity and helpfulness (answers RQ5)

**Output:** Robustness results, explanation evaluation scores, qualitative examples

### Phase 5: Paper Writing & Presentation
**Goal:** Document findings

**Tasks:**
- [ ] Write Introduction + Related Work (using our research docs)
- [ ] Write Methodology (system architecture, BCMO design, baselines)
- [ ] Write Results (tables comparing systems across RQ1-RQ5)
- [ ] Write Discussion (what worked, limitations, future work)
- [ ] Prepare presentation slides

---

## 6. Technical Stack

| Component | Tool | Why |
|-----------|------|-----|
| Language | Python 3.10+ | Standard for NLP |
| LLM (base) | Llama-3-8B-Instruct | Open-weight, fine-tunable, strong baseline |
| LLM (API) | GPT-4o or Claude | For claim extraction comparison / annotation assist |
| Fine-tuning | HuggingFace PEFT + LoRA | Efficient, runs on single GPU |
| Entity linking | ScispaCy + UMLS | Medical NER + concept normalization |
| Ontology storage | JSON (lightweight) or NetworkX (graph queries) | Simple, no DB setup needed |
| RAG | LangChain + FAISS + BioSentVec | Standard RAG pipeline |
| Evaluation | scikit-learn (F1, precision, recall) | Standard metrics |
| Data processing | pandas, xml.etree, praw (Reddit) | Data parsing + collection |

---

## 7. Expected Results & Contributions

### Quantitative (Expected)
| System | Misinfo Detection F1 | Robustness (paraphrase) | False Positive Rate |
|--------|----------------------|-------------------------|---------------------|
| LLM-only | ~0.55-0.65 | Low (drops on paraphrase) | High |
| RAG | ~0.65-0.75 | Medium | Medium |
| **Ontology-grounded** | **~0.75-0.85** | **High (stable)** | **Low** |

### Qualitative
- Ontology-grounded system provides traceable explanations citing specific BCMO relations
- LLM-only system gives vague "this may not be accurate" responses
- RAG system retrieves relevant passages but may not catch implicit contradictions

### Novel Contributions
1. **BCMO** — first breast-cancer-specific misinformation ontology with negation/uncertainty
2. **Structured claim verification** for patient-generated text (vs. existing work on scientific literature)
3. **Comparative analysis** of ontology vs. RAG vs. LLM-only for medical misinformation
4. **Evaluation dataset** of annotated breast cancer patient misinformation

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Cancer-Myth dataset not available | Build own 50-100 adversarial questions from MedQuAD knowledge |
| Reddit API rate limits | Collect data early; cache locally; 5K posts is sufficient |
| LLM fine-tuning compute limits | Use LoRA (0.5% params); Colab Pro or university GPU |
| Ontology too small to cover all cases | Focus on top-20 breast cancer misinformation topics (HER2, chemo, hormone therapy, BRCA, staging) |
| Human evaluation resources limited | Annotate 50-100 examples ourselves; use LLM-as-judge as supplement |

---

## 9. File Structure

```
nlp/
├── data/
│   ├── MedQuAD/                    # Cloned (done)
│   ├── reddit_posts/               # Collected via API
│   ├── eval_dataset.csv            # Our annotated test set
│   └── bcmo.json                   # Ontology
├── src/
│   ├── data_prep/
│   │   ├── parse_medquad.py        # Extract breast cancer content
│   │   └── collect_reddit.py       # Reddit API collector
│   ├── ontology/
│   │   ├── build_bcmo.py           # Ontology construction
│   │   └── bcmo_queries.py         # Query/verify functions
│   ├── pipeline/
│   │   ├── claim_extractor.py      # Decompose text → claims
│   │   ├── entity_linker.py        # Link to BCMO
│   │   └── verifier.py             # Ontology-grounded verification
│   ├── baselines/
│   │   ├── llm_only.py             # LLM-only baseline
│   │   └── rag_baseline.py         # RAG baseline
│   └── evaluation/
│       ├── evaluate.py             # Metrics computation
│       └── robustness_tests.py     # Paraphrase/implicit tests
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_ontology_construction.ipynb
│   ├── 03_claim_extraction.ipynb
│   ├── 04_verification_experiments.ipynb
│   └── 05_results_analysis.ipynb
├── paper/
│   └── main.tex (or .docx)
└── docs/
    ├── topic4_summary.md
    ├── topic4_related_work.md
    ├── topic4_data_analysis.md
    └── project_plan.md             # This file
```

---

## 10. What We Submit

1. **Paper** (8-12 pages): Introduction, Related Work, Methodology, Experiments, Results, Discussion
2. **Code repository**: All source code, notebooks, and data processing scripts
3. **BCMO ontology**: The constructed ontology (shareable artifact)
4. **Evaluation dataset**: Annotated breast cancer misinformation examples
5. **Presentation**: 10-15 minute summary of approach and findings
