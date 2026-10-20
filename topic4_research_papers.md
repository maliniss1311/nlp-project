---
geometry: margin=0.55in
header-includes:
  - \usepackage{titlesec}
  - \usepackage{enumitem}
  - \titleformat{\section}{\normalsize\bfseries}{}{0pt}{}
  - \titleformat{\subsection}{\small\bfseries}{}{0pt}{}
  - \titlespacing*{\section}{0pt}{6pt}{2pt}
  - \titlespacing*{\subsection}{0pt}{4pt}{1pt}
  - \setlength{\parskip}{2pt}
  - \setlength{\parindent}{0pt}
  - \setlist{nosep,leftmargin=12pt,itemsep=1pt}
  - \pagenumbering{gobble}
  - \small
---

\begin{center}
\large\textbf{Related Work: Ontology-Grounded LLMs for Breast Cancer Misinformation Detection}\\
\small\textit{Key Papers, Approaches, and Findings}
\end{center}

# A. Medical Fact-Checking via Structured Claim Verification

\textbf{1. FAITH} (Zhou et al., AAAI 2026, arXiv:2511.12817)\\
\textit{Approach:} Decomposes LLM responses into atomic claims, links each to a medical knowledge graph, scores via evidence paths. No reference answers needed.\\
\textit{Result:} KG-grounded evaluation correlates higher with clinician judgments than text-only; robust to paraphrases.

\textbf{2. CER} (Barone et al., SIGIR 2025, arXiv:2509.13888)\\
\textit{Approach:} Evidence retrieval + LLM reasoning + supervised veracity prediction. Grounds outputs in scientific evidence.\\
\textit{Result:} SOTA on HealthFC, BioASQ-7b, SciFact with cross-dataset generalization.

\textbf{3. Atomic Fact-Checking for Medical QA} (Vladika et al., 2025, arXiv:2505.24830)\\
\textit{Approach:} Decomposes responses into atomic facts, each independently verified against medical guidelines KB.\\
\textit{Result:} 40\% answer improvement, 50\% hallucination detection rate.

\textbf{4. MedGraphRAG} (Wu et al., 2024, arXiv:2408.04187)\\
\textit{Approach:} Triple-linked graph (documents \textrightarrow\ medical sources \textrightarrow\ controlled vocabularies) + U-Retrieval (top-down precise + bottom-up refinement).\\
\textit{Result:} Outperforms SOTA on 9 medical QA + 2 fact-checking benchmarks with source traceability.

\textbf{5. TrumorGPT} (Hang et al., 2025, arXiv:2505.07891)\\
\textit{Approach:} Semantic health KG from medical news + few-shot LLM for KG construction + GraphRAG for grounding.\\
\textit{Result:} Superior health fact-checking; distinguishes true rumors from false claims.

\textbf{6. GraphCheck} (Chen et al., 2025, arXiv:2502.16514)\\
\textit{Approach:} Extracts KGs from text, uses GNN as soft prompt for LLMs. Single-inference multi-hop reasoning.\\
\textit{Result:} 7.1\% improvement over baselines; comparable to DeepSeek-V3/o1 with fewer parameters.

# B. Ontology-Constrained Biomedical NLP

\textbf{7. ANCHOR-RE} (Ming et al., Aug 2026, arXiv:2608.03154)\\
\textit{Approach:} Neuro-symbolic framework: ontology-guided reasoning + external knowledge grounding + verification rules. Training-free.\\
\textit{Result:} F1 0.654\textrightarrow 0.676 (SemRepGS), 0.769\textrightarrow 0.872 (DDI). 69\% precision on unseen 2026 literature.

\textbf{8. RELATE} (Olasunkanmi et al., 2025, arXiv:2509.19057)\\
\textit{Approach:} Three-stage: ontology predicate embeddings \textrightarrow\ SapBERT similarity retrieval \textrightarrow\ LLM reranking with explicit negation handling.\\
\textit{Result:} 94\% accuracy@10; handles negated assertions; rejects 0.4\% irrelevant associations.

\textbf{9. ChronoMedKG} (Ahmed et al., 2026, arXiv:2605.22734)\\
\textit{Approach:} Multi-agent LLM pipeline (multi-model consensus) builds temporal KG with 460K evidence-linked triples + credibility scoring.\\
\textit{Result:} KG retrieval rescues \textbf{47--65\%} of LLM failures vs. 17--29\% for standard RAG.

\textbf{10. OncoNet Ontology} (Karim et al., 2023, arXiv:2310.08365)\\
\textit{Approach:} Cancer-specific ontology (ONO) + BioBERT/SciBERT extractors + LLM fine-tuning for KG enrichment.\\
\textit{Focus:} Biomarker discovery and gene-disease validation (not misinformation detection).

# C. LLM Behavior on Patient Misinformation

\textbf{11. ThReadMed-QA} (Munnangi \& Savage, MLHC 2026, arXiv:2607.12884)\\
\textit{Approach:} 2,437 real patient-physician threads (8,204 QA pairs). Tests LLM misconception correction over multiple turns.\\
\textit{Result:} GPT-5/Claude correct misconceptions \textasciitilde85\% initially but \textbf{drop to \textasciitilde50\% within 2 follow-ups}. Error propagation dominates.

\textbf{12. Medical Sycophancy} (Ping et al., Aug 2026, arXiv:2608.01017)\\
\textit{Approach:} Factorial experiment: 4 factors x 5 models x 500 questions (1.2M trials). CoT trace analysis.\\
\textit{Result:} Fabricated sources raise sycophancy 2x; models reasoning about facts resist, those re-examining own answers concede.

\textbf{13. Limited Impact of Medical Adaptation} (Jeong et al., EMNLP 2024, arXiv:2411.08870)\\
\textit{Finding:} Domain-adapted medical LLMs perform WORSE than base models in 56.7\% of clinical QA cases.\\
\textit{Implication:} Naive fine-tuning insufficient; external structured knowledge (ontology) is a better path.

# D. Key Datasets

\begin{itemize}
\item \textbf{MedQuAD} (Ben Abacha \& Demner-Fushman, BMC Bioinf. 2019): 47,457 QA pairs from 12 NIH sites
\item \textbf{HealthFC} (Vladika et al., LREC-COLING 2024): 750 expert fact-checked health claims with evidence
\item \textbf{FactKG} (Kim et al., ACL 2023): 108K claims with 5 reasoning types (one-hop, conjunction, multi-hop, negation)
\item \textbf{CoVERT} (Mohr et al., LREC 2022): 300 COVID tweets with entities, relations, fact-check labels
\end{itemize}

# E. Summary of Methodological Takeaways

\begin{enumerate}
\item \textbf{Claim decomposition works} --- atomic verification yields 40--50\% improvement (FAITH, Vladika)
\item \textbf{Ontology > general KG > RAG > LLM-only} --- structured domain knowledge rescues 47--65\% of failures (ChronoMedKG)
\item \textbf{Negation modeling is essential} --- medical misinfo often involves overstated/negated claims (RELATE)
\item \textbf{Patient language needs specific adaptation} --- biomedical models alone insufficient (PVminer, ThReadMed-QA)
\item \textbf{LLMs degrade on multi-turn misconceptions} --- 85\%\textrightarrow 50\% accuracy; structured grounding prevents drift (ThReadMed-QA)
\item \textbf{Neuro-symbolic approach is training-free} --- ontology constraints improve LLMs without fine-tuning (ANCHOR-RE)
\item \textbf{No prior work combines:} domain-specific cancer misinfo ontology + patient-language claim decomposition + structured verification
\end{enumerate}
