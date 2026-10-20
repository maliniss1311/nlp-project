---
geometry: margin=0.6in
header-includes:
  - \usepackage{titlesec}
  - \usepackage{enumitem}
  - \titleformat{\section}{\large\bfseries}{}{0pt}{}
  - \titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
  - \titlespacing*{\section}{0pt}{8pt}{4pt}
  - \titlespacing*{\subsection}{0pt}{6pt}{2pt}
  - \setlength{\parskip}{4pt}
  - \setlength{\parindent}{0pt}
  - \setlist{nosep,leftmargin=14pt,itemsep=2pt}
  - \pagenumbering{gobble}
---

\begin{center}
\Large\textbf{Two Key Related Papers for Topic 4}\\[4pt]
\large\textit{Ontology-Grounded LLMs for Breast Cancer Misinformation Detection}
\end{center}

\vspace{6pt}
\hrule
\vspace{8pt}

# Paper 1: FAITH --- KG-Based Automated Fact-Checking for Medical LLMs

\textbf{Full Title:} Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs

\textbf{Authors:} Shasha Zhou, Mingyu Huang, Jack Cole, Charles Britton, Ming Yin, Jan Wolber, Ke Li

\textbf{Affiliations:} University of Electronic Science and Technology of China; University of Exeter

\textbf{Venue:} AAAI 2026 (Conference Paper)

\textbf{arXiv:} 2511.12817v2, December 2025

## Abstract

The recent proliferation of large language models (LLMs) holds the potential to revolutionize healthcare, with strong capabilities in diverse medical tasks. Yet, deploying LLMs in high-stakes healthcare settings requires rigorous verification and validation to understand any potential harm. This paper investigates the reliability and viability of using medical knowledge graphs (KGs) for the automated factuality evaluation of LLM-generated responses. To ground this investigation, we introduce FAITH, a framework designed to systematically probe the strengths and limitations of this KG-based approach. FAITH operates without reference answers by decomposing responses into atomic claims, linking them to a medical KG, and scoring them based on evidence paths. Experiments on diverse medical tasks with human subjective evaluations demonstrate that KG-grounded evaluation achieves considerably higher correlations with clinician judgments and can effectively distinguish LLMs with varying capabilities. It is also robust to textual variances. The inherent explainability of its scoring can further help users understand and mitigate the limitations of current LLMs.

## Methodology (FAITH Pipeline)

\begin{enumerate}
\item \textbf{Claim Decomposition:} LLM-generated medical text is decomposed into atomic claims --- minimal, self-contained factual statements that can be independently verified.
\item \textbf{Entity Linking:} Medical entities in each claim are mapped to nodes in a medical knowledge graph (using UMLS-based concept matching).
\item \textbf{Evidence Path Scoring:} For each claim, the system searches for evidence paths in the KG connecting the identified entities. Claims are scored based on whether supporting, contradicting, or no evidence paths exist.
\item \textbf{Aggregation:} Individual claim scores are aggregated into an overall factuality score for the full response.
\end{enumerate}

## Key Results

\begin{itemize}
\item KG-grounded evaluation achieves \textbf{higher correlation with clinician judgments} than text-based metrics (BERTScore, etc.)
\item Can effectively distinguish between LLMs of varying capabilities without needing ground-truth answers
\item \textbf{Robust to textual variances} --- paraphrased claims still correctly verified
\item Provides inherent \textbf{explainability} via evidence paths (traceable reasoning)
\item Limitations: coverage gaps in KG; some claims too nuanced for simple path matching
\end{itemize}

## Relevance to Our Work

FAITH validates the \textit{core architecture} we propose: decompose patient text into claims, link to structured knowledge, verify via evidence. Key differences in our work:

\begin{itemize}
\item We build a \textbf{domain-specific ontology} (BCMO) rather than using a general medical KG
\item We target \textbf{patient-generated informal language} rather than LLM-generated clinical text
\item We model \textbf{negation, uncertainty, and relation strength} --- which FAITH does not address
\item We focus on \textbf{implicit misinformation} (false presuppositions in questions) rather than explicit factual claims
\end{itemize}

\newpage

\hrule
\vspace{8pt}

# Paper 2: ANCHOR-RE --- Neuro-Symbolic Ontology-Grounded Biomedical RE

\textbf{Full Title:} ANCHOR-RE: An Agentic Neuro-Symbolic Framework for Grounded Biomedical Relation Extraction

\textbf{Authors:} Shufan Ming, Yikun Han, Gibong Hong, Rui Zhang, Halil Kilicoglu

\textbf{Affiliations:} University of Illinois Urbana-Champaign; University of Minnesota

\textbf{Venue:} Submitted to Journal of Biomedical Informatics (under review, Aug 2026)

\textbf{arXiv:} 2608.03154v1, August 2026

## Abstract

Biomedical relation extraction (BioRE) extracts structured knowledge from biomedical literature for applications such as knowledge base construction and hypothesis generation. In real-world settings, BioRE systems must distinguish true relations from numerous candidate entity pairs, most of which are unrelated. Traditional symbolic systems, such as SemRep, provide high precision and interpretability but have limited recall under linguistic variability and complex discourse structures. In contrast, large language models (LLMs) offer strong contextual reasoning capabilities but can produce false-positive relation predictions. This study aims to develop an agentic, neuro-symbolic framework that combines symbolic biomedical knowledge with LLM reasoning to improve BioRE reliability. We developed ANCHOR-RE, a framework that integrates ontology-guided reasoning, external knowledge grounding, and data-driven verification rules into LLM inference. We evaluated it on three BioRE benchmarks (SemRepGS, DDI, and ChemProt) using both proprietary and open-weight LLMs. Results across multiple benchmarks, model families, and post-cutoff literature support ANCHOR-RE as a practical training-free approach to biomedical literature mining.

## Methodology (ANCHOR-RE Pipeline)

\begin{enumerate}
\item \textbf{Ontology-Guided Candidate Generation:} Uses UMLS Semantic Network and domain ontologies to constrain which entity pairs are valid candidates for specific relation types. Eliminates ontologically impossible relations before LLM processing.
\item \textbf{LLM-Based Relation Extraction:} Prompts the LLM with structured evidence blocks from external knowledge bases (UMLS Metathesaurus, DrugBank, CTD) to ground its reasoning.
\item \textbf{Data-Driven Verification Rules:} Post-hoc rules derived from training data patterns to filter false positives (e.g., certain entity type combinations that never co-occur in valid relations).
\item \textbf{Agentic Architecture:} The system operates as an agent that autonomously decides when to query external KBs, what evidence to present to the LLM, and how to verify outputs.
\end{enumerate}

## Key Results

\begin{itemize}
\item \textbf{F1 improvements without fine-tuning:}
  \begin{itemize}
  \item SemRepGS: 0.654 \textrightarrow\ 0.676 (general biomedical)
  \item DDI (drug-drug interactions): 0.769 \textrightarrow\ 0.872 (+13.4\%)
  \item ChemProt: 0.939 \textrightarrow\ 0.941
  \end{itemize}
\item Outperforms direct LLM prompting across \textbf{all benchmarks and model families}
\item On DDI/ChemProt: approaches fine-tuned systems \textbf{without any parameter updates}
\item \textbf{Temporal evaluation} on 100 articles from 2026 (post-training-cutoff): 69\% precision on 500 sampled predictions --- maintains performance on unseen literature
\item Works with both proprietary (GPT-4) and open-weight (Llama) LLMs
\end{itemize}

## Relevance to Our Work

ANCHOR-RE provides the \textit{methodological blueprint} for our ontology-grounded verification system:

\begin{itemize}
\item Demonstrates that \textbf{ontology constraints improve LLM reliability without fine-tuning} --- directly supports our neuro-symbolic approach
\item Their use of \textbf{structured evidence blocks from external KBs} parallels how we would present BCMO evidence to the LLM during claim verification
\item The \textbf{agentic architecture} (autonomous KB querying) is applicable to our multi-step claim decomposition and verification pipeline
\item Key differences in our work:
  \begin{itemize}
  \item They extract relations from \textbf{scientific literature}; we verify claims from \textbf{patient conversations}
  \item They focus on relation extraction (finding true relations); we focus on \textbf{misinformation detection} (finding false/misleading claims)
  \item They use existing ontologies (UMLS); we \textbf{construct a new domain-specific ontology} (BCMO) tailored to misinformation patterns
  \item We additionally model \textbf{claim strength and uncertainty} (``can reduce'' vs. ``always prevents'')
  \end{itemize}
\end{itemize}

\vspace{8pt}
\hrule
\vspace{8pt}

# How Both Papers Inform Our Approach

\begin{center}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Component} & \textbf{FAITH} & \textbf{ANCHOR-RE} \\
\hline
Claim decomposition & Core method & --- \\
Ontology constraints & --- & Core method \\
Evidence path scoring & Core method & Supporting evidence blocks \\
Training-free & Yes & Yes \\
Explainability & Via KG paths & Via ontology rules \\
Target text & LLM outputs & Scientific literature \\
\hline
\end{tabular}
\end{center}

\textbf{Our contribution unifies both:} FAITH's claim-decomposition-and-verify architecture + ANCHOR-RE's ontology-constrained neuro-symbolic reasoning, applied to a new domain (breast cancer patient misinformation) with a purpose-built ontology (BCMO) that handles negation, uncertainty, and implicit false presuppositions.
