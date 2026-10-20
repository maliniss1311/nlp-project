---
geometry: margin=0.5in
header-includes:
  - \usepackage{titlesec}
  - \usepackage{enumitem}
  - \usepackage{multicol}
  - \titleformat{\section}{\normalsize\bfseries}{}{0pt}{}
  - \titleformat{\subsection}{\small\bfseries}{}{0pt}{}
  - \titlespacing*{\section}{0pt}{5pt}{2pt}
  - \titlespacing*{\subsection}{0pt}{4pt}{1pt}
  - \setlength{\parskip}{2pt}
  - \setlength{\parindent}{0pt}
  - \setlist{nosep,leftmargin=12pt}
  - \pagenumbering{gobble}
  - \small
---

\begin{center}
\large\textbf{Topic 4: Ontology-Grounded LLMs for Detecting Breast Cancer Misinformation in Patient Conversations}
\end{center}

\vspace{-4pt}

# Problem & Hypothesis

Patients use online communities and LLMs for cancer information, but discussions contain misconceptions expressed indirectly---embedded in personal experiences or as questions with false presuppositions. The CancerMyth benchmark (585 questions) shows frontier LLMs frequently fail to identify these, and cautious prompting increases false positives.

\textbf{Hypothesis:} Explicit ontology grounding makes LLM-based misinformation detection more accurate, robust, and explainable than parametric knowledge or RAG alone.

# Approach: Structured Claim Verification

The system (1) converts patient statements into structured claims (subject--relation--object triples), (2) grounds claims against a purpose-built Breast Cancer Misinformation Ontology (BCMO), and (3) verifies each against ontology constraints and retrieved evidence.

\textit{Example:} ``Chemotherapy won't work for HER2-positive'' $\rightarrow$ \texttt{HER2+ $\rightarrow$ incompatible\_with $\rightarrow$ chemo} $\rightarrow$ flagged as misleading (both therapies are often combined).

# Research Questions

\textbf{RQ1:} Does domain fine-tuning improve concept/relation extraction from patient text?
\textbf{RQ2:} Does ontology grounding beat LLM-only for misinformation detection?
\textbf{RQ3:} Does the ontology add value beyond standard RAG?
\textbf{RQ4:} Does grounding improve robustness to paraphrases and implicit misinformation?
\textbf{RQ5:} Does grounding produce more faithful explanations?

# BCMO Ontology & Data Resources

\textbf{Concepts:} Cancer subtypes, biomarkers (ER/PR/HER2), treatments, side effects, risk factors, staging, metastasis, prognosis.
\textbf{Relations:} causes, treats, contraindicated\_for, increases\_risk\_of, has\_side\_effect, improves\_survival.
\textbf{Key feature:} Models negation, uncertainty, and relation strength (distinguishing ``can reduce risk'' from ``always prevents'').

\textbf{Data:} Breastcancer.org (\textasciitilde362K posts), MedQuAD (47K QA pairs from NIH), CancerMyth (585+150 adversarial Qs), expert-annotated Reddit cancer misinformation dataset.

# Related Work & Novelty Gap

\begin{small}
\begin{itemize}
\item \textbf{FAITH (AAAI'26):} KG-grounded atomic claim verification for medical LLMs---uses general KGs, not cancer-specific.
\item \textbf{ANCHOR-RE (2026):} Ontology-constrained biomedical relation extraction---targets scientific literature, not patient language.
\item \textbf{CER (SIGIR'25):} Evidence retrieval + LLM reasoning for biomedical fact-checking---verifies news/web, not patient questions.
\item \textbf{TrumorGPT / MedGraphRAG:} Graph-RAG for health fact-checking---no negation/uncertainty modeling.
\item \textbf{OncoNet Ontology:} Cancer KG for biomarker discovery---no misinformation-specific relations.
\item \textbf{GraphCheck / FactKG:} KG + GNN for multi-hop reasoning---general domain, no patient discourse focus.
\item \textbf{Medical Sycophancy (2026):} Shows LLMs yield to patient misinformation pressure---no structured verification solution.
\end{itemize}
\end{small}

# Our Unique Contribution

A purpose-built BCMO with negation/uncertainty/strength modeling, combined with structured claim decomposition from informal patient text---the first system bridging ontology-grounded verification with domain-specific breast cancer patient misinformation detection.
