# Related Work: Ontology-Grounded LLMs for Detecting Breast Cancer Misinformation

## 1. Medical Fact-Checking with LLMs

### FAITH: Fact Verification via Knowledge Graphs (AAAI 2026)
- **Authors:** Zhou et al.
- **Paper:** arXiv:2511.12817
- **Key idea:** Decomposes LLM responses into atomic claims, links them to a medical knowledge graph, and scores them based on evidence paths. No reference answers needed.
- **Relevance:** Directly validates our approach of structured claim verification against a knowledge base. Shows KG-grounded evaluation achieves higher correlation with clinician judgments than text-only methods.

### CER: Combining Evidence and Reasoning (SIGIR 2025)
- **Authors:** Barone et al.
- **Paper:** arXiv:2509.13888
- **Key idea:** Integrates scientific evidence retrieval + LLM reasoning + supervised veracity prediction for biomedical fact-checking.
- **Relevance:** Demonstrates that grounding in scientific evidence outperforms LLM-only approaches on HealthFC, BioASQ-7b, and SciFact benchmarks. Shows cross-dataset generalization is achievable.

### GraphCheck: KG-Powered Fact-Checking (2025)
- **Authors:** Chen et al.
- **Paper:** arXiv:2502.16514
- **Key idea:** Extracts knowledge graphs from text, uses Graph Neural Networks as soft prompts for LLMs. Captures multi-hop reasoning chains.
- **Relevance:** Shows up to 7.1% improvement over baselines across general and medical domains. Demonstrates KG structure helps with complex relational reasoning needed for our ontology-grounded approach.

### TrumorGPT: Graph-Based RAG for Health Fact-Checking (2025)
- **Authors:** Hang et al.
- **Paper:** arXiv:2505.07891
- **Key idea:** Constructs semantic health knowledge graphs from medical news, uses GraphRAG for fact-checking health claims. Addresses hallucination via graph structure.
- **Relevance:** Most similar to our proposed system—combines KG with LLM for health misinformation. However, uses general health KGs rather than a domain-specific ontology, and does not focus on patient-generated language.

### MedGraphRAG: Graph-Based RAG for Medical LLMs (2024)
- **Authors:** Wu et al.
- **Paper:** arXiv:2408.04187
- **Key idea:** Triple-linked graph connecting user documents to credible medical sources and controlled vocabularies. U-Retrieval combines top-down precise retrieval with bottom-up refinement.
- **Relevance:** Validates graph-based retrieval over standard RAG in medical domains. Tested on health fact-checking benchmarks. Their triple-linked structure is analogous to our ontology-to-evidence grounding.

### Atomic Fact-Checking for Medical QA (2025)
- **Authors:** Vladika et al.
- **Paper:** arXiv:2505.24830
- **Key idea:** Decomposes LLM responses into atomic facts, independently verifies each against medical guidelines knowledge base. Achieves 40% answer improvement and 50% hallucination detection.
- **Relevance:** Validates the claim decomposition approach proposed in our system. Their "atomic facts" parallel our decomposition of patient statements into subject-relation-object triples.

---

## 2. Biomedical Ontology + LLM Integration

### ANCHOR-RE: Neuro-Symbolic Biomedical Relation Extraction (2026)
- **Authors:** Ming et al.
- **Paper:** arXiv:2608.03154
- **Key idea:** Integrates ontology-guided reasoning, external knowledge grounding, and verification rules into LLM inference for biomedical relation extraction. No fine-tuning needed.
- **Relevance:** Directly relevant to our ontology-grounded approach. Shows ontology constraints improve LLM precision from 0.654 to 0.676 F1 on biomedical RE without parameter updates.

### RELATE: Ontology-Constrained Relation Extraction (2025)
- **Authors:** Olasunkanmi et al.
- **Paper:** arXiv:2509.19057
- **Key idea:** Three-stage pipeline mapping LLM-extracted relations to standardized ontology predicates. Combines vector search with contextual LLM reasoning.
- **Relevance:** Shows how to constrain LLM outputs to ontology-valid relations. Their negation handling (0.4% rejection rate for irrelevant associations) is relevant to our need for representing negation/uncertainty.

### CLOZE: LLMs for Medical Ontology Extension (2025)
- **Authors:** Wu et al.
- **Paper:** arXiv:2511.16548
- **Key idea:** Uses LLMs to extract medical entities from clinical notes and integrate them into hierarchical ontologies. Zero-shot, no training needed.
- **Relevance:** Demonstrates feasibility of LLM-based ontology construction/extension from unstructured text—relevant to our BCMO construction phase.

### ChronoMedKG: Temporal Biomedical Knowledge Graph (2026)
- **Authors:** Ahmed et al.
- **Paper:** arXiv:2605.22734
- **Key idea:** Temporal KG with 460K evidence-linked triples. Multi-agent LLM pipeline extracts knowledge with multi-model consensus. RAG rescues 47-65% of LLM failures.
- **Relevance:** Shows that structured KG retrieval substantially outperforms standard RAG for medical reasoning. Their evidence-linking and credibility scoring parallel our ontology verification approach.

### From LLMs to KGs for Cancer Biomarker Discovery (2023)
- **Authors:** Karim et al.
- **Paper:** arXiv:2310.08365
- **Key idea:** Constructs OncoNet Ontology (ONO) for cancer, enriches KG using BioBERT/SciBERT extractors, fine-tunes with LLMs on recent literature.
- **Relevance:** Most directly relevant prior work on cancer-specific ontology + LLM. However, focuses on biomarker discovery rather than misinformation detection. Their ontology construction methodology informs our BCMO design.

---

## 3. Medical Misinformation & Patient Language

### Medical Sycophancy in LLMs (2026)
- **Authors:** Ping et al.
- **Paper:** arXiv:2608.01017
- **Key idea:** Studies how LLMs abandon correct medical answers under user pushback. Tested on 500 MedQuAD questions across 5 models.
- **Relevance:** Directly motivates our work—shows LLMs are vulnerable to user misinformation pressure. Models that "reason about medical facts" resist better than those that re-examine their own answers.

### HealthFC: Health Claim Verification Dataset (LREC-COLING 2024)
- **Authors:** Vladika et al.
- **Paper:** arXiv:2309.08503
- **Key idea:** 750 health claims fact-checked by medical experts with evidence from systematic reviews. Bilingual (German/English).
- **Relevance:** Establishes methodology for expert-annotated health claim verification. Could serve as additional evaluation resource or methodological reference.

### CoVERT: Fact-Checked Biomedical Tweets (LREC 2022)
- **Authors:** Mohr et al.
- **Paper:** arXiv:2204.12164
- **Key idea:** 300 COVID-19 tweets annotated with medical named entities, relations, fact-checking labels, and supporting evidence. Crowdsourced methodology.
- **Relevance:** Shows that real-world evidence retrieval outperforms pretrained LM knowledge alone for medical fact-checking. Their entity + relation annotation scheme is relevant to our claim extraction design.

### MedFact: Chinese Medical Fact-Checking (EMNLP 2025)
- **Authors:** Chen et al.
- **Paper:** arXiv:2509.17436
- **Key idea:** First evidence-based fact-checking dataset for LLM-generated medical content. 1,321 questions, 7,409 claims.
- **Relevance:** Methodology for decomposing medical text into verifiable claims and evaluating LLM fact-checking capabilities.

---

## 4. Key Datasets Referenced in the Proposal

| Dataset | Citation | Notes |
|---------|----------|-------|
| MedQuAD | Ben Abacha & Demner-Fushman (2019), arXiv:1901.08079 | 47,457 QA pairs from 12 NIH websites. Original paper. |
| CancerMyth | Referenced in proposal (2024-2025) | 585 cancer questions with false presuppositions + 150 controls |
| Breastcancer.org corpus | Multiple studies (Bessa et al., Wen et al.) | ~362K posts about early breast cancer treatment experiences |
| Expert-annotated Reddit cancer misinfo | Recent dataset (2024-2025) | Multi-dimensional annotation beyond binary true/false |

---

## 5. Gap Analysis: What Our Work Adds

| Existing Work | Gap Our Paper Fills |
|---------------|---------------------|
| FAITH, GraphCheck, TrumorGPT | Use general medical KGs; none builds a domain-specific misinformation ontology for breast cancer |
| ANCHOR-RE, RELATE | Focus on literature mining; don't handle patient-generated informal language |
| CER, HealthFC | Verify factual claims from news/web; don't address implicit misinformation in patient questions |
| OncoNet (Karim et al.) | Cancer KG for biomarker discovery; not designed for misinformation relations (contraindicated_for, etc.) |
| MedGraphRAG | General medical graph; doesn't model negation, uncertainty, or relation strength |
| Medical sycophancy work | Identifies the problem but doesn't propose structured verification solutions |

**Our unique contribution:** A purpose-built breast cancer misinformation ontology (BCMO) that models negation/uncertainty/strength, combined with structured claim decomposition from patient-generated text, specifically targeting the gap between what patients say and what is medically accurate.
