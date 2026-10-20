---
geometry: margin=1in
mainfont: Arial
fontsize: 11pt
header-includes:
  - \usepackage{titlesec}
  - \usepackage{enumitem}
  - \titleformat{\section}{\fontsize{12}{14}\bfseries}{}{0pt}{}
  - \titleformat{\subsection}{\fontsize{11}{13}\bfseries}{}{0pt}{}
  - \titlespacing*{\section}{0pt}{12pt}{6pt}
  - \titlespacing*{\subsection}{0pt}{8pt}{4pt}
  - \setlength{\parskip}{6pt}
  - \setlength{\parindent}{0pt}
  - \setlist{itemsep=2pt,leftmargin=18pt}
  - \pagenumbering{arabic}
---

# Project Plan

Ontology-Grounded LLMs for Detecting Breast Cancer Misinformation in Patient Conversations

NLP Course Assignment

# 1. Project Overview

This project builds a system that detects misinformation in breast cancer patient conversations by decomposing patient statements into structured claims and verifying them against a purpose-built Breast Cancer Misinformation Ontology (BCMO).

Central Hypothesis: Explicit ontology grounding makes LLM-based misinformation detection more accurate, robust, and explainable than relying on the LLM's parametric knowledge or retrieval alone.

Deliverables:

- Breast Cancer Misinformation Ontology (BCMO) in JSON format
- Claim decomposition pipeline (patient text to structured triples)
- Ontology-grounded verification system
- Comparative evaluation: Ontology-grounded vs. LLM-only vs. RAG
- Final paper with results and analysis

# 2. Research Questions

RQ1 (Domain adaptation): Does fine-tuning an LLM on breast-cancer-related language improve its ability to extract medical concepts, relations, and claims from patient-generated text?

RQ2 (Ontology grounding): Does explicitly representing breast-cancer concepts and relationships improve misinformation detection compared with an LLM alone?

RQ3 (Ontology versus RAG): Does an ontology provide information that ordinary retrieval-augmented generation does not?

RQ4 (Robustness): Does ontology grounding make models more robust to paraphrases, implicit misinformation, and counterfactual claims?

RQ5 (Explanation faithfulness): Does grounding predictions in structured medical knowledge produce explanations that are more consistent with the evidence supporting the final decision?

# 3. Data Sources

## 3.1 MedQuAD Cancer.gov Subset (Available now)

116 XML files with cancer QA pairs from the National Cancer Institute. Each file includes UMLS concept identifiers, semantic types, and question types (treatment, diagnosis, symptoms, risk factors, staging, prognosis). Licensed under CC BY 4.0. Already cloned to our project directory.

Use: Ontology concept extraction, domain adaptation fine-tuning, claim verification templates.

## 3.2 Reddit Breast Cancer Posts (Accessible via API)

Patient-generated discussions from r/breastcancer, r/cancer, and r/HER2positive. Informal language containing real patient questions, misconceptions, and treatment discussions. Accessible through the Reddit API with a free developer account.

Use: Domain language adaptation, real examples of patient misinformation for testing.

Plan: Collect 5,000 to 10,000 posts.

## 3.3 Cancer-Myth Benchmark (Request from authors)

585 cancer questions with false presuppositions plus 150 control questions without false presuppositions. Created by Zhu et al. at USC (arXiv:2504.11373, 2025). No public repository found yet.

Use: Adversarial evaluation benchmark.

Fallback: If unavailable, construct 50 to 100 breast-cancer-specific questions with false presuppositions ourselves using MedQuAD knowledge and clinical guidelines.

## 3.4 NCI Thesaurus and UMLS (Free with license)

Structured medical terminology with breast cancer concepts, treatment hierarchies, and biomarker relationships. Available through the NLM UMLS license (free, approval takes 1 to 2 days).

Use: Foundation for BCMO ontology construction.

## 3.5 Self-Constructed Evaluation Set (Our contribution)

100 to 200 breast cancer patient statements annotated for misinformation. Derived from Reddit posts, paraphrased MedQuAD content, and synthetic examples. Annotated with binary labels (misinformation or not), claim-level labels, and explanations.

# 4. System Architecture

The system has four stages:

Stage 1 - Claim Decomposer: Takes patient text as input and uses a fine-tuned LLM to extract subject-relation-object triples. For example, the statement "chemotherapy won't work for HER2-positive" becomes the triple (HER2-positive, incompatible_with, chemotherapy).

Stage 2 - Entity Linker: Maps extracted entities to BCMO concepts using ScispaCy or QuickUMLS for medical concept normalization.

Stage 3 - Ontology Verifier: Checks each claim against BCMO relations and produces a verdict: supported, contradicted, unsupported, or misleading (overstated strength).

Stage 4 - Evidence Scorer and Explainer: Produces a final verdict with a traceable explanation citing the specific ontology relations that support or contradict the claim.

# 5. Implementation Plan

## Phase 1: Data Preparation and Ontology Construction

Tasks:

- Parse MedQuAD Cancer.gov XML files and extract breast-cancer concepts, relations, and treatments
- Collect Reddit r/breastcancer posts via API (5,000 to 10,000 posts)
- Apply for UMLS license and download NCI Thesaurus breast cancer subset
- Construct BCMO version 1.0 with 16 concept classes, 12 relation types, and negation/uncertainty modifiers
- Create evaluation dataset of 100 to 200 annotated patient statements

Outputs: bcmo.json (ontology), eval_dataset.csv (annotated test set), reddit_posts.jsonl (patient language corpus)

## Phase 2: Claim Decomposition Pipeline

Tasks:

- Design claim format as (subject, relation, object, strength, negation)
- Implement prompted claim extraction using a base LLM (Llama-3-8B or GPT-4 API)
- Fine-tune a smaller model on MedQuAD-derived claim extraction examples using LoRA
- Evaluate precision and recall of extracted claims against manual annotation
- Compare base versus fine-tuned model (answers RQ1)

Outputs: claim_extractor.py, fine-tuned model weights, extraction evaluation results

## Phase 3: Ontology-Grounded Verification

Tasks:

- Implement entity linking from extracted claims to BCMO nodes
- Implement claim verification logic with four verdict types: supported, contradicted, unsupported, misleading
- Build RAG baseline (retrieve relevant Cancer.gov passages, ask LLM to verify)
- Build LLM-only baseline (ask LLM directly without ontology or retrieval)
- Compare all three systems on the evaluation set (answers RQ2 and RQ3)

Outputs: ontology_verifier.py, rag_baseline.py, llm_baseline.py, comparison results

## Phase 4: Robustness and Explanation Evaluation

Tasks:

- Create paraphrased test set (same misinformation, different wording)
- Create implicit misinformation examples (false presuppositions in questions)
- Create counterfactual claims (swap entities to test shortcut reliance)
- Evaluate all three systems on robustness set (answers RQ4)
- Conduct human evaluation of explanation faithfulness on 50 to 100 examples (answers RQ5)

Outputs: Robustness results, explanation evaluation scores, qualitative examples

## Phase 5: Paper Writing and Presentation

Tasks:

- Write Introduction and Related Work
- Write Methodology (system architecture, BCMO design, baselines)
- Write Results (tables comparing systems across RQ1 through RQ5)
- Write Discussion (what worked, limitations, future work)
- Prepare presentation slides

# 6. Technical Stack

Programming language: Python 3.10+

LLM base model: Llama-3-8B-Instruct (open-weight, fine-tunable, strong baseline)

LLM API: GPT-4o or Claude (for claim extraction comparison and annotation assistance)

Fine-tuning: HuggingFace PEFT with LoRA (efficient, runs on single GPU)

Entity linking: ScispaCy with UMLS (medical NER and concept normalization)

Ontology storage: JSON (lightweight) or NetworkX (for graph queries)

RAG pipeline: LangChain with FAISS and BioSentVec embeddings

Evaluation: scikit-learn for F1, precision, recall

Data processing: pandas, xml.etree, praw (Reddit API)

# 7. Expected Results

The ontology-grounded system is expected to outperform both baselines:

- LLM-only baseline: Estimated F1 of 0.55 to 0.65, low robustness to paraphrasing, high false positive rate
- RAG baseline: Estimated F1 of 0.65 to 0.75, medium robustness, medium false positive rate
- Ontology-grounded system: Estimated F1 of 0.75 to 0.85, high robustness (stable across paraphrases), low false positive rate

The ontology-grounded system will provide traceable explanations citing specific BCMO relations, while the LLM-only system gives vague responses and the RAG system retrieves relevant passages but may miss implicit contradictions.

# 8. Novel Contributions

1. BCMO: First breast-cancer-specific misinformation ontology with negation, uncertainty, and relation strength modeling
2. Structured claim verification for patient-generated text (versus existing work on scientific literature)
3. Comparative analysis of ontology versus RAG versus LLM-only for medical misinformation detection
4. Evaluation dataset of annotated breast cancer patient misinformation

# 9. Risk Mitigation

Cancer-Myth dataset not available: Build our own 50 to 100 adversarial questions from MedQuAD knowledge.

Reddit API rate limits: Collect data early and cache locally. 5,000 posts is sufficient for our purposes.

LLM fine-tuning compute limits: Use LoRA which updates only 0.5 percent of parameters. Use Colab Pro or university GPU cluster.

Ontology too small to cover all cases: Focus on top 20 breast cancer misinformation topics including HER2, chemotherapy, hormone therapy, BRCA, and staging.

Human evaluation resources limited: Annotate 50 to 100 examples ourselves and supplement with LLM-as-judge evaluation.

# 10. Team Responsibilities

This project will be completed collaboratively by the team, with responsibilities divided into the following components.

Team Member 1: Literature review and related work, and analysis of existing breast cancer misinformation research and datasets. This includes surveying prior work on medical fact-checking with LLMs (FAITH, ANCHOR-RE, CER, MedGraphRAG), ontology-grounded biomedical NLP, and LLM behavior on patient misinformation. Team Member 1 also identifies relevant benchmarks and datasets, assesses their accessibility, and writes the Related Work section of the paper.

Team Member 2: Data preprocessing, corpus preparation, and development of the breast cancer ontology. This includes parsing MedQuAD XML files to extract breast cancer concepts, collecting Reddit posts via the API, applying for and integrating the UMLS/NCI Thesaurus, constructing the BCMO ontology (concept classes, relations, negation/uncertainty modifiers), and preparing the annotated evaluation dataset. Team Member 2 writes the Data and Ontology sections of the paper.

Team Member 3: Model implementation, including baseline LLM and ontology-grounded approaches, as well as experimental evaluation. This includes building the claim decomposition pipeline, entity linker, ontology verifier, RAG baseline, and LLM-only baseline. Team Member 3 also runs all experiments, computes metrics, conducts robustness tests, and writes the Experiments and Results sections of the paper.

All team members: Develop research questions, analyze results, conduct error analysis, write and revise the paper, and prepare the final presentation.

# 11. Project File Structure

The project is organized as follows:

- data/ contains MedQuAD (already cloned), reddit_posts (collected via API), eval_dataset.csv (annotated test set), and bcmo.json (ontology)
- src/data_prep/ contains parse_medquad.py and collect_reddit.py
- src/ontology/ contains build_bcmo.py and bcmo_queries.py
- src/pipeline/ contains claim_extractor.py, entity_linker.py, and verifier.py
- src/baselines/ contains llm_only.py and rag_baseline.py
- src/evaluation/ contains evaluate.py and robustness_tests.py
- notebooks/ contains exploration, construction, extraction, experiments, and results notebooks
- paper/ contains the final paper

# 12. What We Submit

1. Paper (8 to 12 pages): Introduction, Related Work, Methodology, Experiments, Results, Discussion
2. Code repository: All source code, notebooks, and data processing scripts
3. BCMO ontology: The constructed ontology as a shareable artifact
4. Evaluation dataset: Annotated breast cancer misinformation examples
5. Presentation: 10 to 15 minute summary of approach and findings
