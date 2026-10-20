# Data Source Analysis: Feasibility & Access Assessment
## Topic 4: Ontology-Grounded LLMs for Breast Cancer Misinformation Detection

---

## 1. MedQuAD — Medical Question Answering Dataset

| Attribute | Details |
|-----------|---------|
| **Size** | 47,457 QA pairs from 12 NIH websites |
| **Cancer subset** | `1_CancerGov_QA` folder — QA pairs from Cancer.gov |
| **Format** | XML files with metadata (question type, focus, UMLS CUI, semantic type) |
| **License** | CC BY 4.0 (fully open) |
| **Access** | Publicly available on GitHub: `github.com/abachaa/MedQuAD` |
| **Status** | ✅ **IMMEDIATELY ACCESSIBLE** — can clone and use right now |
| **Limitations** | Answers removed from 3 MedlinePlus subsets (copyright); Cancer.gov subset is intact |
| **Citation** | Ben Abacha & Demner-Fushman, BMC Bioinformatics, 2019 |

**Use in our project:**
- Domain adaptation / fine-tuning (cancer-specific QA pairs)
- Ontology construction (UMLS CUIs already annotated)
- Concept normalization training
- Question-type understanding (37 types: Treatment, Diagnosis, Side Effects, etc.)

**Action needed:** `git clone https://github.com/abachaa/MedQuAD` → use `1_CancerGov_QA/` folder

---

## 2. Cancer-Myth — Adversarial Evaluation Benchmark

| Attribute | Details |
|-----------|---------|
| **Size** | 585 cancer questions with false presuppositions + 150 non-false-presupposition controls |
| **Authors** | Zhu, Chen, Yu, Lin, Law, Jizzini, Nieva, Liu, Jia (USC) |
| **Paper** | arXiv:2504.11373 (April 2025, revised Oct 2025) |
| **License** | Not explicitly stated in arXiv; paper says "expert-verified adversarial dataset" |
| **Access** | ⚠️ **UNCERTAIN** — no public GitHub repo found; no HuggingFace dataset page found |
| **Status** | ⚠️ Need to contact authors (Wang Bill Zhu, Robin Jia at USC) for access |

**Key findings from the paper:**
- No frontier LLM (GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet) corrects false presuppositions more than 43% of the time
- Precautionary prompting raises accuracy to 80% BUT misidentifies 41% of benign questions as containing misinformation
- Shows prompting alone is not a reliable remedy

**Use in our project:**
- Primary adversarial evaluation benchmark (not for training)
- Measures false-positive rate on benign questions (Cancer-Myth-NFP)
- Tests whether ontology grounding avoids the accuracy/false-positive tradeoff

**Action needed:** Contact authors at USC to request dataset access. Check if supplementary materials contain data link.

---

## 3. Reddit Cancer Misinformation (Expert-Annotated, Multi-Dimensional)

| Attribute | Details |
|-----------|---------|
| **Size** | Multi-subreddit; covers breast, lung, colon, prostate cancer |
| **Authors** | Pessianzadeh, Jamie, Sultana, Himmelstein, Zektser, Ganz, Hosseinmardi, Ghasemian, Rezapour |
| **Paper** | arXiv:2607.12383 (July 2026, revised Aug 2026) |
| **Taxonomy** | 7 dimensions: misinfo presence, information type, risk level, stance, topical focus + more |
| **License** | Not yet stated (paper says "taxonomy, dataset, and findings" shared) |
| **Access** | ⚠️ **VERY RECENT** (Aug 2026) — likely in pre-publication; dataset may not be public yet |
| **Status** | ⚠️ Contact authors (Rezvaneh Rezapour) to request early access |

**Key findings:**
- ~6% of Reddit cancer discussions contain misinformation
- Recurring narratives: unsupported treatments, distrust of conventional medicine, misleading diagnosis claims
- Few-shot LLM prompting improves classification on nuanced taxonomy dimensions
- Multi-dimensional annotation (not binary true/false)

**Use in our project:**
- Naturalistic evaluation data (real patient/community language, not synthetic)
- Breast cancer subset for testing
- Multi-dimensional labels allow fine-grained analysis
- Tests whether ontology grounding helps with nuanced, multi-dimensional misinfo types

**Action needed:** Contact Rezapour's lab; monitor for dataset release; may need to wait or use Reddit API to replicate with own annotation.

---

## 4. Breastcancer.org Patient Forum Data

| Attribute | Details |
|-----------|---------|
| **Size** | ~500K posts (362K retained about early breast cancer in prior studies); also 28K original posts + 663K replies in another study |
| **Format** | Forum posts (unstructured text, conversational patient language) |
| **License** | ❌ **PROPRIETARY** — Breastcancer.org has its own terms of service |
| **Access** | ❌ **NOT FREELY AVAILABLE** — requires data-sharing agreement with Breastcancer.org |
| **Status** | ❌ **DIFFICULT TO OBTAIN** — previous researchers had formal collaborations |

**Prior work using this data:**
- Bessa et al. analyzed 362,074 posts for treatment experiences/symptoms
- Wen et al. studied 28,139 posts + 663,748 replies for community communication patterns
- Both required formal agreements with the platform

**Use in our project:**
- Domain language for LLM adaptation (informal patient vocabulary, breast cancer terminology)
- Training data for understanding patient expression patterns

**Alternatives if not accessible:**
1. **Reddit r/breastcancer** — publicly available via Reddit API/Pushshift (though Pushshift access now restricted)
2. **Reddit cancer subreddits** — r/cancer, r/breastcancer, r/HER2positive (accessible via Reddit API)
3. **Patient forum scraping** — community.breastcancer.org (check ToS carefully)
4. **Synthetic patient language** — generate with LLMs seeded from available examples

**Action needed:** Contact Breastcancer.org research team OR pivot to Reddit as primary patient language source.

---

## 5. Ontology Construction Resources (Freely Available)

| Resource | Access | Use |
|----------|--------|-----|
| **UMLS Metathesaurus** | Free with license (NLM account) | Medical concept identifiers, relations, semantic types |
| **NCI Thesaurus** | Open (cancer-specific ontology) | Breast cancer concepts, staging, treatments |
| **SNOMED CT** | Free for research (NLM license) | Clinical terminology, hierarchical relations |
| **Breast Cancer Ontology (BCO)** | Check BioPortal | Existing partial ontology if available |
| **DrugBank** | Free academic license | Drug interactions, contraindications |
| **NCCN Guidelines** | Require subscription | Gold-standard treatment protocols (for validation) |

---

## Feasibility Summary

| Data Source | Accessible? | Role in Project | Priority |
|-------------|-------------|-----------------|----------|
| MedQuAD (Cancer.gov) | ✅ Yes — CC BY 4.0 | Domain adaptation, ontology construction | HIGH |
| Cancer-Myth | ⚠️ Contact authors | Adversarial evaluation benchmark | HIGH |
| Reddit Cancer Misinfo | ⚠️ Contact authors / very new | Naturalistic evaluation | MEDIUM |
| Breastcancer.org | ❌ Requires agreement | Domain language adaptation | LOW (use alternatives) |
| UMLS / NCI Thesaurus | ✅ Free with license | Ontology construction foundation | HIGH |
| Reddit (r/breastcancer) | ✅ Via Reddit API | Alternative patient language source | HIGH |

---

## Recommended Implementation Plan

### Phase 1: Start immediately (all accessible)
1. Clone MedQuAD → extract Cancer.gov QA subset → analyze for ontology concepts
2. Apply for UMLS license (free, takes ~1-2 business days)
3. Collect Reddit r/breastcancer posts via Reddit API (public, free)
4. Begin BCMO ontology construction using NCI Thesaurus + MedQuAD concepts

### Phase 2: Request access (1-4 weeks)
5. Contact Cancer-Myth authors (Zhu/Jia at USC) for benchmark access
6. Contact Pessianzadeh/Rezapour for Reddit misinformation dataset
7. If no response: construct own evaluation set using Reddit data + expert annotation

### Phase 3: Model development
8. Fine-tune LLM on MedQuAD + Reddit patient language (domain adaptation — RQ1)
9. Implement claim decomposition pipeline
10. Build ontology-grounded verification system
11. Evaluate: ontology vs. LLM-only vs. RAG (RQ2, RQ3)
12. Test robustness with Cancer-Myth benchmark (RQ4)
13. Evaluate explanation faithfulness (RQ5)

---

## Methods & Tools

| Component | Recommended Tools |
|-----------|-------------------|
| LLM base model | Llama-3 8B or Mistral 7B (open-weight, fine-tunable) |
| Fine-tuning | LoRA/QLoRA via HuggingFace PEFT |
| Ontology format | OWL/RDF (Protege for construction) OR simple JSON-LD |
| Entity linking | SapBERT, ScispaCy, or UMLS QuickUMLS |
| Claim decomposition | Prompted LLM (GPT-4 or fine-tuned model) |
| Evidence retrieval | PubMed API + sentence embeddings (BioSentVec) |
| Evaluation metrics | F1, AUROC, explanation faithfulness (human eval) |
| KG querying | NetworkX (lightweight) or Neo4j (full graph DB) |

---

## Documentation / Key References

1. **MedQuAD:** Ben Abacha & Demner-Fushman. "A Question-Entailment Approach to QA." BMC Bioinformatics, 2019.
2. **Cancer-Myth:** Zhu et al. "Cancer-Myth: Evaluating LLMs on Patient Questions with False Presuppositions." arXiv:2504.11373, 2025.
3. **Reddit Cancer Misinfo:** Pessianzadeh et al. "Beyond Binary Detection: A Multi-Dimensional Taxonomy of Cancer Misinformation on Reddit." arXiv:2607.12383, 2026.
4. **FAITH (methodology reference):** Zhou et al. "Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs." AAAI 2026.
5. **ANCHOR-RE (methodology reference):** Ming et al. "An Agentic Neuro-Symbolic Framework for Grounded Biomedical Relation Extraction." arXiv:2608.03154, 2026.
