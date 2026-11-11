# BCMO Ontology Gap Analysis

How gaps found during fact-checking were identified and filled, and how this analysis plugs into the study to test whether ontology grounding actually works.

Tool: `src/ontology/gap_filler.py` · Gap log: `data/ontology_gaps.json` · Output ontology: `data/bcmo_v2.1.json`

---

## 1. What is a "gap"?

A gap is any claim the verifier cannot judge because BCMO is missing the knowledge, not because the claim is genuinely unknowable. The pipeline (`entity_linker.py` → `bcmo_queries.verify_claim`) exposes three kinds:

| Gap type | Detected when | Fix |
|---|---|---|
| **Alias** | The entity doesn't link, but every token of it appears in an existing concept name | Add the surface form as an alias |
| **Concept** | The entity matches no concept at all | Add a new concept instance (class + UMLS/NCI code) |
| **Relation** | Both entities link, but the triple is absent, so the verdict is `unsupported` | Add the triple (positive or negated) with a source |

Without this step, all three end up as `unsupported`. That verdict mixes up "the ontology doesn't know" with "this claim is unverifiable", which hides the ontology's real coverage limits.

## 2. Gap-filling method

1. **Detect.** Run claims through linking and verification, and log each failure with its claim text and frequency.
2. **Find evidence.**
   - Aliases: lexical containment against existing concepts.
   - Concepts: match against BCMO's curated `common_misinformation` list (10 entries).
   - Relations: sentences in the 43 MedQuAD breast-cancer documents that mention both entities plus a relation cue. Negation cues ("no evidence", "not linked", …) mark a sentence as contradicting.
3. **Label.** Each gap gets `supported`, `contradicted`, `known_misinformation`, or `unverifiable`.
4. **Human review.** Approve or reject each proposal. Nothing without evidence is merged.
5. **Merge.** Approved items go into a new version with provenance (`gap_id`, evidence label, source document or QA ID).
6. **Second pass.** Relations that were blocked by a missing entity are re-checked against the new version.

## 3. Gaps found (pilot: 8 sample claims)

| Claim | Gap | Evidence | Action | v2.0 verdict | v2.1 verdict |
|---|---|---|---|---|---|
| "Drinking alcohol has nothing to do with breast cancer" | Alias: `alcohol` → Alcohol Consumption | Lexical | Alias added | unsupported | **supported**¹ |
| "Exercise lowers your chance of breast cancer" | Alias: `exercise` → Physical Exercise | Lexical | Alias added | unsupported | **supported** |
| "Herceptin is used for HER2-positive tumors" | Alias: `HER2-positive` → HER2-Positive Breast Cancer | Lexical | Alias added | unsupported | **supported** |
| "Wearing an underwire bra causes breast cancer" | Concept + relation | ACS (curated myth list) | New `MisinformationTopic` "Underwire Bra"; negated `causes` triple | unsupported | **contradicted** |
| "High-dose vitamin C cures breast cancer" | Concept | None | Held for review (class guessed as `Medication`, no CUI) | unsupported | unsupported |
| "Radiation is used after lumpectomy for DCIS" | Relation | None in MedQuAD | Held; needs an NCCN source | unsupported | unsupported |
| "Being overweight raises breast cancer risk" | none | — | — | supported | supported |
| "Mammograms can find breast cancer early" | none | — | — | supported | supported |

¹ The triple is supported. Because the claim is phrased as a denial, the claim extractor must set `negated=True` for the final label to be *contradicted*. This is a claim-extraction issue, not an ontology issue (see §6).

**Ontology change (v2.0 → v2.1):** 3 aliases, 1 concept (a new `MisinformationTopic` class, 15 → 16 classes), and 1 relation (136 → 137, negated, ACS-sourced). Each addition carries a provenance record.

**Coverage result:**

| Metric | v2.0 | v2.1 |
|---|---|---|
| Entities linked | 11/16 (69%) | 15/16 (94%) |
| Claims with a non-`unsupported` verdict | 2/8 (25%) | 6/8 (75%) |

**What this shows:**
- Most failures (3 of 6) were alias gaps: the knowledge was already in BCMO but couldn't be reached from patient wording. Lay vocabulary, not missing medical facts, is the main bottleneck.
- Myth-type claims need their own concept class. Without one, the most dangerous claims get the weakest verdict (`unsupported` rather than `contradicted`).
- Some gaps are genuine evidence gaps in the source corpus (MedQuAD has no DCIS-plus-radiation sentence). Gap filling should report these, not paper over them.

> Caveat: the 8 claims are a hand-built pilot chosen to exercise each gap type. The numbers show the mechanism works. They are **not** an estimate of real-world coverage. That requires the evaluation set (§4).

## 4. Using gaps to test whether the ontology works

The central hypothesis is that explicit ontology grounding beats LLM-only and RAG. Gap analysis turns this into a falsifiable test. If grounding helps, its errors should be concentrated in identifiable, fixable gaps, and fixing those gaps should improve accuracy without hurting precision.

### 4.1 Experimental conditions

| System | Description |
|---|---|
| LLM-only | Llama-3-8B judges the claim directly |
| RAG | LLM plus retrieved Cancer.gov passages |
| BCMO v2.0 | Ontology-grounded verifier, original ontology |
| BCMO v2.1 | Same verifier after gap filling |
| BCMO v2.1 + LLM fallback | Ontology verdict when one exists, otherwise defer to the LLM |

**Important:** to avoid leakage, gaps must be mined from a development split only. Report v2.0 vs v2.1 on a held-out test split. Filling gaps on the test set would inflate the results.

### 4.2 Metrics

1. **Coverage:** percentage of entities linked, and percentage of claims with a verdict (as in §3).
2. **Accuracy and macro-F1** over supported / contradicted / unsupported / misleading.
3. **Misinformation recall:** the percentage of false claims labelled *contradicted*. This is the clinically important number.
4. **Selective accuracy:** accuracy on the claims the ontology does answer, compared with the LLM on the same claims. This separates "the ontology is right when it speaks" from "the ontology speaks often".
5. **Gap-attributed error rate:** the percentage of ontology-system errors explained by a logged gap, broken down by type.

### 4.3 Mapping to research questions

| RQ | How gap analysis contributes |
|---|---|
| **RQ2** (ontology vs LLM) | Selective accuracy shows whether ontology verdicts beat the LLM where they apply. Coverage shows how often they apply. |
| **RQ3** (ontology vs RAG) | Relation gaps with **no** MedQuAD evidence but a correct ontology verdict (expert-curated facts) are cases RAG over the same corpus cannot answer. Count them. |
| **RQ4** (robustness) | Paraphrases mostly produce **alias** gaps. Measure verdict consistency across paraphrase sets for v2.0 vs v2.1. Robustness should rise after alias filling. |
| **RQ5** (explanations) | Every ontology verdict cites a triple and its provenance. Human raters can check that the cited source supports the verdict, which is not possible for LLM-only output. |

### 4.4 Decision criteria: works vs doesn't work

**Evidence the ontology works:**
- BCMO v2.1 selective accuracy is higher than LLM-only and RAG on the same claims.
- Misinformation recall for v2.1 is higher than for the baselines.
- v2.0 → v2.1 improves test-set accuracy and paraphrase consistency, which means the gaps were the real limitation.
- Most ontology errors are gap-attributed (fixable), not cases where the ontology confidently gives the wrong verdict.

**Evidence the ontology doesn't work (or isn't worth it):**
- Selective accuracy is no better than the LLM on the claims the ontology covers.
- Gap filling raises coverage but lowers precision, meaning the added facts or aliases introduce wrong links.
- The gap log keeps growing with new **concept** gaps on every new batch. That would mean the ontology can't keep up with patient language, and the approach doesn't scale.
- Most errors come from claim extraction, not the ontology. In that case the bottleneck is elsewhere, and the result should be reported as such.
- RAG matches the ontology on the cases in the RQ3 row above, so the ontology adds nothing beyond retrieval.

**Mixed outcome:** the ontology is precise but covers too little. Then the hybrid (v2.1 + LLM fallback) is the expected best system. That is still a positive finding for grounding, framed as "ontology as a high-precision verifier".

## 5. Proposed paper material

- **Table:** gap type distribution (alias / concept / relation) on the dev split, with examples.
- **Table:** v2.0 vs v2.1 vs baselines on the test split, using the metrics in §4.2.
- **Figure:** coverage over gap-filling iterations, to check whether new gaps level off or keep growing.
- **Discussion:** unverifiable gaps such as vitamin C and DCIS radiation, reported honestly as corpus limitations.

## 6. Limitations

- Evidence matching is keyword and negation-cue based, so it can mislabel hedged sentences. Human review is required before merging.
- The class for a new concept is a keyword guess and needs confirmation, plus UMLS/NCI codes.
- The claim extractor's negation handling decides the final verdict for denials ("alcohol has nothing to do with…"). Its errors should be reported separately from ontology errors.
- `validate_bcmo.py` currently validates only v2.0. It should be run on v2.1 before results are reported.

## 7. Reproduce

```bash
python3 src/ontology/gap_filler.py demo            # pilot shown in §3
python3 src/ontology/gap_filler.py detect --claims <dev_claims.json>
python3 src/ontology/gap_filler.py propose
python3 src/ontology/gap_filler.py review --approve <ids>
python3 src/ontology/gap_filler.py merge --version 2.1
python3 src/ontology/gap_filler.py report --claims <test_claims.json>
```
