---
geometry: margin=0.65in
fontsize: 10pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{hyperref}
  - \usepackage{enumitem}
  - \usepackage{multicol}
  - \setlist{nosep,leftmargin=12pt}
  - \setlength{\parskip}{3pt}
  - \setlength{\parindent}{0pt}
  - \usepackage{titlesec}
  - \titleformat{\section}{\normalsize\bfseries}{}{0pt}{}
  - \titleformat{\subsection}{\small\bfseries}{}{0pt}{}
  - \titlespacing*{\section}{0pt}{8pt}{3pt}
  - \titlespacing*{\subsection}{0pt}{6pt}{2pt}
---

\begin{center}
\large\textbf{BCMO: A Data-Grounded Breast Cancer Ontology\\for Misinformation Detection from MedQuAD}\\[6pt]
\normalsize Meharry Medical College --- NLP Course Project \hfill October 2026
\end{center}

\vspace{4pt}

# Abstract

Breast cancer misinformation in online patient communities threatens treatment decisions, yet LLMs struggle to detect misconceptions in informal patient language. We present the Breast Cancer Misinformation Ontology (BCMO v2.0), constructed through systematic extraction from MedQuAD combined with expert-curated clinical knowledge. Our pipeline parses 43 breast-cancer-focused documents across 6 MedQuAD subsets, extracts 118 normalized concepts and 98 data-grounded relations, and merges these with 55 expert-curated relations to produce an ontology of 15 concept classes, 168 instances, 14 relation types, and 136 provenance-tracked factual relations. BCMO achieves 100\% concept coverage and supports structured claim verification with four verdict categories (supported, contradicted, unsupported, misleading). Every relation carries a provenance flag distinguishing MedQuAD-grounded evidence (60\%) from expert curation (40\%), enabling downstream systems to reason about evidence quality.

# 1. Introduction

Patients increasingly seek breast cancer information from online communities and LLMs, but these discussions frequently contain misinformation expressed indirectly---embedded in personal narratives or framed as partial truths. Current LLMs rely on parametric memory that may be outdated or confidently wrong. The CancerMyth benchmark shows frontier models fail on nuanced breast cancer claims, particularly those that are \textit{partially} true but misleading (e.g., ``HER2-positive breast cancer doesn't respond to chemotherapy''---while HER2-targeted therapy is important, chemotherapy remains a standard component of HER2+ regimens).

Prior knowledge-grounded approaches (FAITH, CER, MedGraphRAG) demonstrate the value of structured verification but use general medical knowledge graphs lacking breast-cancer-specific contraindication reasoning, biomarker-subtype specificity, and probabilistic vs.\ absolute claim distinction. We address this gap with BCMO, featuring: (1) \textbf{data-grounded construction} from MedQuAD with traceable provenance, (2) \textbf{misinformation-specific relations} including \texttt{contraindicated\_for} and negated biomarker assertions with strength/certainty modifiers, and (3) \textbf{four-category claim verification} (supported, contradicted, unsupported, misleading) rather than binary classification.

# 2. Method

## 2.1 Data Source and Filtering

MedQuAD (Ben Abacha \& Demner-Fushman, 2019) contains 47,457 medical QA pairs from NIH sources in 12 subsets. Each XML document has a \texttt{<Focus>} tag, UMLS annotations, and typed QA pairs (\texttt{qtype}: treatment, symptoms, susceptibility, stages, etc.).

We scan all 12 subdirectories with strict Focus-tag filtering using 28 curated patterns. This selects documents whose \textit{primary topic} is breast cancer, yielding 43 documents across 6 subsets (Table 1)---avoiding the over-inclusion problem where keyword matching on answer text pulls in unrelated cancer documents.

\begin{table}[H]
\centering
\small
\caption{Breast cancer documents by MedQuAD subset}
\vspace{-4pt}
\begin{tabular}{lrlr}
\toprule
\textbf{Subset} & \textbf{Docs} & \textbf{Subset} & \textbf{Docs} \\
\midrule
MPlus ADAM & 27 & GARD & 4 \\
CancerGov & 5 & GHR & 1 \\
MPlus Health Topics & 5 & SeniorHealth & 1 \\
\bottomrule
\multicolumn{4}{r}{\textbf{Total: 43 documents, 161 QA pairs}} \\
\end{tabular}
\end{table}

## 2.2 Concept Extraction and Normalization

From 161 QA pairs, we extract concepts via category-specific regex patterns across 10 categories: subtypes, biomarkers, treatments, diagnostics, symptoms, risk factors, protective factors, genetic factors, stages, and side effects. Canonical normalization maps each match to a standardized name, eliminating duplicates from case variants (``mammogram''/``Mammogram'') and abbreviations (``DCIS''/``ductal carcinoma in situ''). This yields \textbf{118 unique concepts}: treatment (26), subtype (16), diagnostic (15), stage (15), risk factor (13), biomarker (9), symptom (8), side effect (7), genetic factor (6), protective factor (3).

## 2.3 Question-Type-Guided Relation Extraction

We exploit MedQuAD's \texttt{qtype} metadata as semantic signals: \texttt{treatment} $\rightarrow$ \texttt{treats}/\texttt{has\_side\_effect}; \texttt{susceptibility} $\rightarrow$ \texttt{increases\_risk\_of}; \texttt{symptoms} $\rightarrow$ \texttt{has\_symptom}; \texttt{exams and tests} $\rightarrow$ \texttt{detects}; \texttt{prevention} $\rightarrow$ \texttt{decreases\_risk\_of}; \texttt{stages} $\rightarrow$ \texttt{has\_stage}. Domain-specific extractors parse constructions like ``Treatment of [condition] may include [therapy]'' and normalize condition names. Each relation carries provenance (source document ID, QA pair ID, \texttt{data\_grounded: true}). After deduplication: \textbf{98 unique relations} (Table 2).

\begin{table}[H]
\centering
\small
\caption{Data-grounded relations from MedQuAD}
\vspace{-4pt}
\begin{tabular}{lrlr}
\toprule
\textbf{Relation} & \textbf{N} & \textbf{Relation} & \textbf{N} \\
\midrule
treats & 32 & has\_side\_effect & 10 \\
increases\_risk\_of & 16 & has\_symptom & 7 \\
has\_stage & 14 & decreases\_risk\_of & 4 \\
detects & 12 & has\_biomarker & 3 \\
\bottomrule
\multicolumn{4}{r}{\textbf{Total: 98 relations}} \\
\end{tabular}
\end{table}

## 2.4 Ontology Construction

BCMO v2.0 merges MedQuAD extractions with expert knowledge from NCCN Guidelines, NCI Thesaurus, and UMLS. \textbf{Concept classes} (15) have expert-curated instances with UMLS CUI codes and aliases; MedQuAD concepts not already present are added with \texttt{source: ``MedQuAD''}. \textbf{Factual relations} merge 98 data-grounded + 55 expert-curated relations (expert takes priority on duplicates), yielding 136 total. Expert relations include contraindications (endocrine therapy contraindicated for TNBC), clinical trial results (Trastuzumab improves survival per HERA trial), and negated biomarker assertions (TNBC is HER2-negative by definition). Six \textbf{modifier types} (negation, certainty, strength, temporality, population scope, source quality) support nuanced claim representation. Ten \textbf{misinformation patterns} encode common myths with verdicts and explanations.

## 2.5 Claim Verification

The verification engine produces four verdicts: \textbf{Supported} (triple matches a factual relation), \textbf{Contradicted} (counter-relation exists, e.g., \texttt{treats} vs.\ \texttt{contraindicated\_for}), \textbf{Unsupported} (entities known but relation unrecorded), and \textbf{Misleading} (matches a misinformation pattern). Each verdict includes explanatory notes from the ontology.

# 3. Results

## 3.1 Ontology Statistics

\begin{table}[H]
\centering
\small
\caption{BCMO v2.0 summary}
\vspace{-4pt}
\begin{tabular}{lr|lr}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{Metric} & \textbf{Value} \\
\midrule
Concept classes & 15 & Relation types & 14 \\
Total instances & 168 & Total relations & 136 \\
\quad Expert & 131 & \quad Expert & 55 (40\%) \\
\quad MedQuAD & 37 & \quad MedQuAD & 81 (60\%) \\
Modifiers & 6 types & Misinfo patterns & 10 \\
\bottomrule
\end{tabular}
\end{table}

The largest classes are Treatment (27), DiagnosticTest (16), BreastCancerSubtype (16), and SideEffect (16). Treatment benefited most from MedQuAD extraction (+16 instances). Validation confirms \textbf{100\% concept coverage} (118/118 extracted concepts found in ontology). Only 4 minor consistency issues arise from verbose MedQuAD phrasings. Sixteen cross-class overlaps are intentional (e.g., drugs in both Treatment and Medication; BRCA1/2 in both GeneticFactor and Biomarker). Zero duplicate relations.

## 3.2 Claim Verification

\begin{table}[H]
\centering
\small
\caption{Claim verification examples}
\vspace{-4pt}
\begin{tabular}{p{4.8cm}ll}
\toprule
\textbf{Claim} & \textbf{Verdict} & \textbf{Note} \\
\midrule
Trastuzumab treats HER2+ BC & Supported & NCCN \\
Endocrine Tx treats TNBC & Contradicted & Lacks HR \\
Chemo treats HER2+ BC & Supported & + targeted Tx \\
Vitamin C treats BC & Unsupported & No evidence \\
\bottomrule
\end{tabular}
\end{table}

The system correctly identifies Endocrine Therapy / TNBC as \textit{contradicted} (not merely unsupported) via the explicit \texttt{contraindicated\_for} relation with the note ``TNBC lacks hormone receptors.'' Chemotherapy / HER2+ is verified as \textit{supported} with context that it is combined with targeted therapy---addressing the misconception that targeted therapy replaces chemotherapy.

## 3.3 Misinformation Detection

Testing against patient-style statements: ``Underwire bras cause breast cancer'' $\rightarrow$ \textbf{False}; ``No family history means no risk'' $\rightarrow$ \textbf{Misleading} (85\% of cases lack family history); ``Natural remedies cure cancer'' $\rightarrow$ \textbf{False}; ``Chemo for HER2+'' $\rightarrow$ no flag (correct); ``Mammograms cause cancer'' $\rightarrow$ \textbf{Misleading} (minimal dose). The system correctly avoids flagging the factually correct chemotherapy statement.

# 4. Discussion

\textbf{Contributions.} BCMO v2.0 is the first breast cancer ontology designed for misinformation detection combining data-grounded MedQuAD extraction with expert clinical knowledge and provenance tracking. Unlike general KGs (FAITH, CER), it includes contraindication reasoning and biomarker-subtype specificity critical for breast cancer.

\textbf{Multi-subset value.} Expanding from CancerGov-only (5 docs) to 6 subsets (43 docs) added rare subtypes from GARD (metaplastic carcinoma, BRCA syndromes), screening procedures from MedlinePlus ADAM, and genetic context from GHR.

\textbf{Limitations.} Regex extraction limits recall for complex relations. The misinformation matcher uses word overlap, which may miss semantic paraphrases. Future work will integrate LLM-based extraction, evaluate against CancerMyth and expert-annotated Reddit data, and compare ontology-grounded verification against LLM-only and RAG baselines.

# References

\small
\begin{enumerate}[leftmargin=14pt]
\item Ben Abacha, A. \& Demner-Fushman, D. (2019). A Question-Entailment Approach to QA. \textit{BMC Bioinformatics}, 20(511).
\item Zhou, K. et al. (2026). FAITH: Fact-Aligned Inference with Health KGs. \textit{AAAI 2026}.
\item Barone, F. et al. (2025). CER: Claim, Evidence, and Reasoning. \textit{SIGIR 2025}.
\item Wu, J. et al. (2024). MedGraphRAG. \textit{arXiv:2408.04187}.
\item Ming, D. et al. (2026). ANCHOR-RE: Ontology-Constrained Biomedical RE. \textit{arXiv:2608.03154}.
\item Vladika, J. et al. (2025). Atomic Fact-Checking for Medical QA. \textit{arXiv:2505.24830}.
\item Hang, C. et al. (2025). TrumorGPT. \textit{arXiv:2505.07891}.
\item NCCN Guidelines: Breast Cancer, v4.2026.
\item NCI Thesaurus, OBO Edition, 2026.
\end{enumerate}
