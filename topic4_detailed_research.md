# Detailed Related Work Research: Topic 4
## Ontology-Grounded LLMs for Detecting Breast Cancer Misinformation in Patient Conversations

---

## CATEGORY A: Medical Fact-Checking & Claim Verification with LLMs

### Paper 1: FAITH — KG-Based Fact-Checking for Medical LLMs
- **Full title:** Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs
- **Authors:** Zhou, Huang, Cole, Britton, Yin, Wolber, Li
- **Venue:** AAAI 2026
- **arXiv:** 2511.12817
- **Approach:**
  - Decomposes LLM-generated medical responses into **atomic claims**
  - Links each claim to a medical knowledge graph (KG)
  - Scores claims based on **evidence paths** in the KG
  - Operates WITHOUT reference answers — fully automated
- **Key results:** KG-grounded evaluation achieves higher correlation with clinician judgments than text-only methods; robust to textual variances; inherently explainable
- **Relevance to us:** Validates our core approach of decomposing text into claims and verifying against structured knowledge. We extend this by building a domain-specific ontology rather than using a general medical KG.

---

### Paper 2: CER — Combining Evidence and Reasoning for Biomedical Fact-Checking
- **Full title:** Combating Biomedical Misinformation through Multi-modal Claim Detection and Evidence-Based Verification
- **Authors:** Barone, Romano, Riccio, Postiglione, Moscato
- **Venue:** SIGIR 2025
- **arXiv:** 2509.13888
- **Approach:**
  - Integrates scientific **evidence retrieval** + **LLM reasoning** + supervised veracity prediction
  - Retrieves high-quality biomedical evidence to ground LLM outputs
  - Prevents hallucination by ensuring generated outputs are grounded in verifiable sources
- **Key results:** State-of-the-art on HealthFC, BioASQ-7b, SciFact; promising cross-dataset generalization
- **Relevance to us:** Shows that evidence-grounded verification outperforms LLM-only. Our system adds structured ontology reasoning on top of evidence retrieval.

---

### Paper 3: Atomic Fact-Checking for Medical QA
- **Full title:** Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs
- **Authors:** Vladika, Domres, Nguyen, Moser, Nano, Busch, Adams, Bressem, Bernhardt, Combs, Borm, Matthes, Peeken
- **arXiv:** 2505.24830 (2025)
- **Approach:**
  - Decomposes LLM responses into discrete **atomic facts**
  - Each atomic fact independently verified against authoritative medical guidelines knowledge base
  - Enables targeted error correction and direct tracing to source literature
- **Key results:** Up to **40% overall answer improvement** and **50% hallucination detection rate**
- **Relevance to us:** Directly validates claim decomposition methodology. Their "atomic facts" parallel our decomposition of patient statements into subject-relation-object triples.

---

### Paper 4: MedGraphRAG — Graph-Based RAG for Safe Medical LLMs
- **Full title:** Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation
- **Authors:** Wu, Zhu, Qi, Chen, Xu, Menolascina, Grau
- **arXiv:** 2408.04187 (2024)
- **Approach:**
  - **Triple Graph Construction** — connects user documents to credible medical sources and controlled vocabularies
  - **U-Retrieval** — combines Top-down Precise Retrieval with Bottom-up Response Refinement
  - Creates a three-layer linked structure for evidence tracing
- **Key results:** Outperforms SOTA across 9 medical QA benchmarks + 2 health fact-checking benchmarks; responses include credible source documentation
- **Relevance to us:** Their triple-linked structure is analogous to our ontology-to-evidence grounding. We go further by designing domain-specific relations for misinformation.

---

### Paper 5: TrumorGPT — Graph-RAG for Health Fact-Checking
- **Full title:** TrumorGPT: Graph-Based Retrieval-Augmented Large Language Model for Fact-Checking
- **Authors:** Hang, Yu, Tan
- **arXiv:** 2505.07891 (2025)
- **Approach:**
  - Constructs **semantic health knowledge graphs** from latest medical news
  - Uses few-shot LLM learning for KG construction and semantic reasoning
  - **GraphRAG** addresses hallucination by grounding in regularly updated KGs
- **Key results:** Superior performance on healthcare fact-checking datasets; effectively distinguishes "trumors" (true rumors) from false claims
- **Relevance to us:** Most similar architecture to ours. Key differences: they use general health KGs (not domain-specific), don't model negation/uncertainty, and don't target patient-generated informal language.

---

### Paper 6: GraphCheck — KG + GNN for Multi-Hop Fact-Checking
- **Full title:** GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking
- **Authors:** Chen et al.
- **arXiv:** 2502.16514 (2025)
- **Approach:**
  - Extracts KGs from text, uses **Graph Neural Networks** as soft prompts for LLMs
  - Captures **multi-hop reasoning chains** often missed by existing methods
  - Single inference call (efficient)
- **Key results:** Up to 7.1% improvement over baselines; comparable to DeepSeek-V3 and OpenAI-o1 with far fewer parameters
- **Relevance to us:** Demonstrates that graph structure helps with complex relational reasoning. Our BCMO enables similar multi-hop reasoning (e.g., HER2+ → treatment → contraindications).

---

### Paper 7: HealthFC — Expert-Annotated Health Claim Verification Dataset
- **Full title:** HealthFC: Verifying Health Claims with Evidence-Based Medical Fact-Checking
- **Authors:** Vladika, Schneider, Matthes
- **Venue:** LREC-COLING 2024
- **arXiv:** 2309.08503
- **Approach:**
  - 750 health claims (German + English) labeled by **medical experts**
  - Evidence from systematic reviews and clinical trials
  - Baseline systems using NLI, text similarity, and LLMs
- **Key results:** Challenging benchmark; establishes methodology for expert-annotated health verification
- **Relevance to us:** Methodological reference for evaluation design. We could use similar expert annotation protocols for our breast cancer misinfo dataset.

---

## CATEGORY B: Biomedical Ontology + LLM Integration

### Paper 8: ANCHOR-RE — Neuro-Symbolic Biomedical Relation Extraction
- **Full title:** ANCHOR-RE: An Agentic Neuro-Symbolic Framework for Grounded Biomedical Relation Extraction
- **Authors:** Ming, Han, Hong, Zhang, Kilicoglu
- **arXiv:** 2608.03154 (Aug 2026, under review at J Biomedical Informatics)
- **Approach:**
  - Integrates **ontology-guided reasoning** + external knowledge grounding + data-driven verification rules into LLM inference
  - No fine-tuning required — training-free
  - Evaluated on SemRepGS, DDI, ChemProt benchmarks
  - Temporal evaluation on 100 articles from 2026 (post-cutoff)
- **Key results:**
  - F1 improvement: 0.654 → 0.676 (SemRepGS), 0.769 → 0.872 (DDI)
  - 69% precision on previously unseen 2026 literature
  - Works with both proprietary and open-weight LLMs
- **Relevance to us:** MOST directly relevant methodological paper. Shows ontology constraints improve LLM reliability without fine-tuning. We can adopt their neuro-symbolic framework for our BCMO-grounded verification.

---

### Paper 9: RELATE — Ontology-Constrained Relation Mapping
- **Full title:** RELATE: Relation Extraction in Biomedical Abstracts with LLMs and Ontology Constraints
- **Authors:** Olasunkanmi, Satusky, Yi, Bizon, Lee, Ahalt
- **arXiv:** 2509.19057 (2025)
- **Approach:**
  - Three-stage pipeline: (1) ontology preprocessing with predicate embeddings, (2) similarity-based retrieval with SapBERT, (3) LLM-based reranking with negation handling
  - Maps free-text LLM outputs to **standardized ontology predicates**
  - Uses ChemProt + Biolink Model
- **Key results:** 52% exact match, 94% accuracy@10; effectively rejects irrelevant associations (0.4%); handles negated assertions
- **Relevance to us:** Shows how to constrain LLM outputs to ontology-valid relations. Their explicit negation handling directly informs our need to model "Treatment X does NOT cause Y."

---

### Paper 10: CLOZE — LLM-Based Medical Ontology Extension
- **Full title:** Utilizing Large Language Models for Zero-Shot Medical Ontology Extension from Clinical Notes
- **Authors:** Wu, Xie, Wu, He, Shao, Hu, Yang
- **Venue:** BIBM 2025 Workshop
- **arXiv:** 2511.16548
- **Approach:**
  - Zero-shot framework using LLMs to extract entities from clinical notes
  - Integrates them into hierarchical medical ontologies
  - No training data or labeled annotations needed
  - Automated PHI removal for privacy
- **Relevance to us:** Demonstrates feasibility of LLM-based ontology construction. Relevant to our BCMO construction phase — we could use LLMs to help populate the ontology from breast cancer literature.

---

### Paper 11: ChronoMedKG — Temporal Biomedical Knowledge Graph
- **Full title:** ChronoMedKG: A Temporally-Grounded Biomedical Knowledge Graph and Benchmark for Clinical Reasoning
- **Authors:** Ahmed, Firoozbakht, Galke Poech, Baumbach, Röttger
- **arXiv:** 2605.22734 (2026)
- **Approach:**
  - 460,497 evidence-linked triples covering 13,431 diseases
  - Multi-agent LLM pipeline: multiple frontier LLMs independently extract, only keep relations with **multi-model consensus**
  - Temporal grounding (when findings become relevant)
  - Credibility scoring per triple
- **Key results:** 92.7% agreement against reference; KG retrieval rescues **47-65%** of LLM failures vs. 17-29% for standard RAG
- **Relevance to us:** Strong evidence that structured KG retrieval substantially outperforms standard RAG for medical reasoning. Their multi-model consensus approach could improve our ontology population reliability.

---

### Paper 12: OncoNet — Cancer KG for Biomarker Discovery
- **Full title:** From Large Language Models to Knowledge Graphs for Biomarker Discovery in Cancer
- **Authors:** Karim, Molinas Comet, Shajalal, Beyan, Rebholz-Schuhmann, Decker
- **arXiv:** 2310.08365 (2023)
- **Approach:**
  - Constructs **OncoNet Ontology (ONO)** for cancer-specific knowledge
  - Uses BioBERT and SciBERT for information extraction from scientific articles
  - Fine-tunes KG with LLMs based on recent articles
  - Enables semantic reasoning for gene-disease relation validation
- **Relevance to us:** Most relevant prior work on cancer-specific ontology + LLM. However, focuses on biomarker discovery, not misinformation. Their methodology for ontology construction from scientific literature informs our BCMO design.

---

## CATEGORY C: LLM Behavior with Patient Misinformation

### Paper 13: ThReadMed-QA — Multi-Turn Medical Misconception Evaluation
- **Full title:** Evaluating Large Language Models on Misconceptions in Multi-Turn Medical Conversations
- **Authors:** Munnangi, Savage
- **Venue:** MLHC 2026
- **arXiv:** 2607.12884 (July 2026)
- **Approach:**
  - 2,437 real patient-physician conversation threads (8,204 QA pairs) from AskDocs
  - Evaluates whether LLMs can detect and correct **misconceptions embedded in patient questions** over multiple turns
  - Rubric-based LLM-as-a-Judge scoring
- **Key results:**
  - GPT-5 and Claude-Haiku correct false presuppositions ~85% initially but **drop to ~50% within 2 follow-ups**
  - Error propagation is the dominant failure mode
  - Even with perfect prior context, performance remains imperfect
- **Relevance to us:** DIRECTLY motivates our work. Shows that (1) patients embed misconceptions in questions, (2) LLMs degrade over conversation turns, (3) structured verification is needed. Our ontology-grounded approach could address the error propagation problem by checking claims against stable structured knowledge.

---

### Paper 14: Medical Sycophancy in LLMs
- **Full title:** Why LLMs Give In: Conversational Factors and Reasoning Behind Medical Sycophancy
- **Authors:** Ping, Çarık, Wohn, Ding, Wang, Rho
- **arXiv:** 2608.01017 (Aug 2026)
- **Approach:**
  - Factorial experiment: 4 conversational factors × 5 models × 500 MedQuAD questions (1.2M trials)
  - Studies when LLMs abandon correct answers under user pushback
  - Chain-of-thought trace analysis
- **Key results:**
  - Fabricated sources raise sycophancy 2x when accompanying the question
  - Sycophancy varies 67x across questions vs. 3x across models
  - Models that "reason about medical facts" resist; those that re-examine their own answers concede
- **Relevance to us:** Shows LLMs are vulnerable to patient misinformation pressure. Grounding in an external ontology would provide a stable reference that resists conversational pressure.

---

## CATEGORY D: Domain Adaptation & Fine-Tuning for Medical NLP

### Paper 15: The Limited Impact of Medical Adaptation (EMNLP 2024, extended 2025)
- **Authors:** Jeong, Mani, Garg, Lipton, Oberst
- **arXiv:** 2411.08870
- **Key finding:** Medical LLMs fail to consistently improve over base models in zero/few-shot and supervised fine-tuning for medical QA. In 56.7% of cases, medical LLMs perform WORSE than general models.
- **Relevance:** Suggests that naive domain adaptation may not help — our approach of adding structured external knowledge (ontology) is a better path than just fine-tuning on medical text.

### Paper 16: RAG + Fine-Tuning for Medical QA (2025)
- **Authors:** Hassan et al.
- **arXiv:** 2512.05863
- **Key finding:** Fine-tuned LLaMA-2 with RAG achieves 71.8% on PubMedQA (vs 55.4% zero-shot); grounding in retrieved evidence **reduces unsupported content by ~60%**.
- **Relevance:** Demonstrates RAG value, but we hypothesize ontology adds structured reasoning RAG cannot provide.

### Paper 17: PVminer — Domain-Adapted NLP for Patient Voice
- **Authors:** Fodeh et al.
- **arXiv:** 2602.21165 (2026)
- **Key approach:** Domain-adapted BERT models (PV-BERT) for structuring patient voice in secure messages. Multi-label classification with topic augmentation.
- **Key results:** F1 of 82.25% on patient communication coding; outperforms biomedical/clinical baselines.
- **Relevance:** Shows domain-adapted models for patient-generated language outperform generic biomedical models. Supports our RQ1 on domain fine-tuning for patient text.

---

## CATEGORY E: Datasets & Benchmarks

### Paper 18: MedQuAD — The Original Dataset Paper
- **Full title:** A Question-Entailment Approach to Question Answering
- **Authors:** Ben Abacha, Demner-Fushman
- **Venue:** BMC Bioinformatics, 2019
- **arXiv:** 1901.08079
- **Dataset:** 47,457 QA pairs from 12 NIH websites (including Cancer.gov)
- **Relevance:** One of our primary data sources for domain adaptation and concept normalization.

### Paper 19: CoVERT — Fact-Checked Biomedical Social Media
- **Authors:** Mohr, Wührl, Klinger
- **Venue:** LREC 2022
- **arXiv:** 2204.12164
- **Dataset:** 300 COVID-19 tweets with medical entity annotations, relations, fact-checking labels, and evidence
- **Key finding:** Real-world evidence retrieval outperforms pretrained LM parametric knowledge for medical fact-checking
- **Relevance:** Methodological reference for annotating patient-generated social media for fact-checking. Similar annotation scheme needed for our breast cancer Reddit data.

### Paper 20: FactKG — Fact Verification via Reasoning on Knowledge Graphs
- **Authors:** Kim, Park, Kwon, Jo, Thorne, Choi
- **Venue:** ACL 2023
- **arXiv:** 2305.06590
- **Dataset:** 108K natural language claims with 5 reasoning types: One-hop, Conjunction, Existence, Multi-hop, Negation
- **Relevance:** Establishes taxonomy of reasoning types needed for KG-based verification. Our BCMO verification will need similar multi-hop and negation reasoning.

---

## Summary: How These Papers Map to Our Research Questions

| Our RQ | Supporting Papers | What They Show |
|--------|-------------------|----------------|
| **RQ1** (Domain fine-tuning) | PVminer, Limited Impact paper, BioMamba | Patient-language adaptation helps but naive medical fine-tuning doesn't always; domain-specific training on patient text is key |
| **RQ2** (Ontology vs LLM-only) | FAITH, ANCHOR-RE, ChronoMedKG | KG/ontology grounding consistently outperforms LLM-only: +40% accuracy, rescues 47-65% of failures |
| **RQ3** (Ontology vs RAG) | MedGraphRAG, ChronoMedKG, RAG+FT paper | Graph-structured retrieval outperforms flat RAG; structured relations enable multi-hop reasoning RAG misses |
| **RQ4** (Robustness) | ThReadMed-QA, Medical Sycophancy | LLMs degrade over turns and under pressure; stable external knowledge provides robustness |
| **RQ5** (Explanation faithfulness) | FAITH, Atomic Fact-Checking, GraphCheck | Structured verification enables traceable explanations with evidence paths |

---

## Key Methodological Insights for Our Implementation

1. **Claim decomposition works** — Multiple papers (FAITH, Atomic Fact-Checking) confirm that breaking text into atomic verifiable units dramatically improves accuracy
2. **Ontology constraints beat general KGs** — ANCHOR-RE shows domain-specific ontology guidance improves F1 by 10+ points over direct prompting
3. **Negation handling is critical** — RELATE explicitly handles negation; medical misinfo often involves negated or overstated claims
4. **Multi-model consensus improves reliability** — ChronoMedKG uses multiple LLMs to validate extracted triples; relevant for our ontology population
5. **Patient language ≠ clinical language** — PVminer and ThReadMed-QA show patient-generated text needs specialized adaptation beyond standard biomedical models
6. **RAG alone is insufficient** — ChronoMedKG shows KG retrieval rescues 47-65% of failures vs only 17-29% for standard RAG; validates our RQ3
7. **LLMs fail on implicit misinfo over time** — ThReadMed-QA shows 85%→50% accuracy degradation; structured verification addresses this
