"""
Extract semantic relations from MedQuAD breast cancer QA pairs.

Uses question type (qtype) as a semantic signal combined with pattern
matching on answer text to produce (subject, relation, object) triples
with provenance tracking back to source documents.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"


# Question-type to relation-type mapping
QTYPE_RELATION_MAP = {
    "treatment": ["treats", "has_side_effect"],
    "susceptibility": ["increases_risk_of", "decreases_risk_of"],
    "causes": ["increases_risk_of", "causes"],
    "symptoms": ["has_symptom"],
    "exams and tests": ["detects", "diagnoses"],
    "prevention": ["decreases_risk_of", "prevents"],
    "stages": ["has_stage"],
    "outlook": ["affects_prognosis"],
    "inheritance": ["has_genetic_factor"],
    "complications": ["has_side_effect", "causes"],
}

# Patterns to extract relation triples from answer text.
# Each entry: (regex_pattern, subject_group, relation, object_group_or_literal)
RELATION_PATTERNS = [
    # Treatment relations
    (r"((?:chemotherapy|radiation therapy|surgery|mastectomy|lumpectomy|"
     r"targeted therapy|hormone therapy|endocrine therapy|immunotherapy|"
     r"trastuzumab|pertuzumab|tamoxifen|anastrozole|letrozole|exemestane|"
     r"fulvestrant|palbociclib|lapatinib|aromatase inhibitor[s]?|"
     r"selective estrogen receptor modulator[s]?|oophorectomy|ovarian ablation|"
     r"breast-conserving surgery|sentinel lymph node biopsy|"
     r"ado-trastuzumab emtansine|neoadjuvant|adjuvant))"
     r"\s+(?:is|are|may be)\s+(?:used|given)\s+(?:to\s+treat|for)\b",
     "treats", None),

    # "treatment of X may include Y"
    (r"[Tt]reatment\s+(?:of|for)\s+([\w\s-]+?)\s+may\s+include",
     "treated_by", None),

    # Detection patterns
    (r"((?:mammogra(?:m|phy)|ultrasound|MRI|biopsy|CT scan|PET scan|"
     r"bone scan|clinical breast exam|physical exam|"
     r"sentinel lymph node biopsy|chest x-ray|genetic test(?:ing)?))"
     r"[^.]*?(?:used to|can)\s+(?:detect|find|diagnose|check for)",
     "detects", None),

    # Risk factor patterns
    (r"(?:risk factors?\s+(?:for|include)|increases?\s+(?:the\s+)?risk\s+of)"
     r"[^.]*?(family history|age|obesity|alcohol|radiation exposure|"
     r"dense breast|hormone.{0,20}therapy|menstruat\w+.{0,20}early|"
     r"menopause.{0,20}late|never having given birth|BRCA[12]|"
     r"personal history|estrogen|gene mutation)",
     "increases_risk_of", "Breast Cancer"),

    # Side effect patterns
    (r"(?:side effects?\s+(?:of|include|may include))[^.]*?"
     r"(hair loss|nausea|fatigue|heart failure|blood clots|"
     r"premature menopause|lymphedema|bone loss|hot flashes|"
     r"inflammation|infertility|vomiting|weight (?:gain|loss))",
     "has_side_effect", None),

    # Biomarker patterns
    (r"(HER2[/-]?positive|triple[- ]negative|hormone receptor[- ]positive|"
     r"estrogen receptor[- ]positive|progesterone receptor[- ]positive)"
     r"[^.]*?(?:respond|treated|target)",
     "has_biomarker_implication", None),

    # Protective factor patterns
    (r"(?:protective factors?|decrease\s+(?:the\s+)?risk)[^.]*?"
     r"(breastfeeding|exercise|early pregnancy|"
     r"selective estrogen receptor modulator|aromatase inhibitor|"
     r"oophorectomy|ovarian ablation)",
     "decreases_risk_of", "Breast Cancer"),
]

# Canonical mapping for normalizing extracted entities
ENTITY_CANONICAL = {
    "chemo": "Chemotherapy", "chemotherapy": "Chemotherapy",
    "radiation": "Radiation Therapy", "radiation therapy": "Radiation Therapy",
    "surgery": "Surgery", "mastectomy": "Mastectomy",
    "lumpectomy": "Lumpectomy",
    "breast-conserving surgery": "Breast-Conserving Surgery",
    "targeted therapy": "Targeted Therapy",
    "hormone therapy": "Hormone Therapy",
    "endocrine therapy": "Endocrine Therapy",
    "immunotherapy": "Immunotherapy",
    "trastuzumab": "Trastuzumab", "pertuzumab": "Pertuzumab",
    "tamoxifen": "Tamoxifen",
    "anastrozole": "Anastrozole", "letrozole": "Letrozole",
    "exemestane": "Exemestane", "fulvestrant": "Fulvestrant",
    "palbociclib": "Palbociclib", "lapatinib": "Lapatinib",
    "mammogram": "Mammogram", "mammography": "Mammogram",
    "ultrasound": "Ultrasound", "mri": "MRI",
    "biopsy": "Biopsy", "ct scan": "CT Scan", "pet scan": "PET Scan",
    "bone scan": "Bone Scan", "chest x-ray": "Chest X-Ray",
    "clinical breast exam": "Clinical Breast Exam",
    "physical exam": "Physical Exam",
    "genetic testing": "Genetic Testing", "genetic test": "Genetic Testing",
    "sentinel lymph node biopsy": "Sentinel Lymph Node Biopsy",
    "family history": "Family History", "age": "Age",
    "obesity": "Obesity", "alcohol": "Alcohol Consumption",
    "dense breast": "Dense Breast Tissue",
    "brca1": "BRCA1 Gene Mutation", "brca2": "BRCA2 Gene Mutation",
    "breastfeeding": "Breastfeeding", "exercise": "Physical Exercise",
    "early pregnancy": "Early Pregnancy",
    "hair loss": "Hair Loss", "nausea": "Nausea",
    "fatigue": "Fatigue", "heart failure": "Heart Failure",
    "blood clots": "Blood Clots", "lymphedema": "Lymphedema",
    "premature menopause": "Premature Menopause",
    "bone loss": "Bone Loss", "hot flashes": "Hot Flashes",
    "vomiting": "Vomiting", "infertility": "Infertility",
    "oophorectomy": "Oophorectomy", "ovarian ablation": "Ovarian Ablation",
    "aromatase inhibitor": "Aromatase Inhibitor",
    "aromatase inhibitors": "Aromatase Inhibitor",
    "selective estrogen receptor modulator": "Selective Estrogen Receptor Modulator",
    "selective estrogen receptor modulators": "Selective Estrogen Receptor Modulator",
    "neoadjuvant": "Neoadjuvant Therapy", "adjuvant": "Adjuvant Therapy",
    "ado-trastuzumab emtansine": "Ado-Trastuzumab Emtansine",
}


def normalize_entity(raw):
    """Normalize an extracted entity to its canonical name."""
    key = raw.lower().strip()
    return ENTITY_CANONICAL.get(key, raw.strip().title())


CONDITION_CANONICAL = {
    "early, localized, or operable breast cancer": "Breast Cancer",
    "early, localized, or operable breast": "Breast Cancer",
    "locally advanced or inflammatory breast cancer": "Inflammatory Breast Cancer",
    "locally advanced or inflammatory breast": "Inflammatory Breast Cancer",
    "locoregional recurrent breast cancer": "Breast Cancer",
    "metastatic breast cancer": "Metastatic Breast Cancer",
    "ductal carcinoma in situ": "Ductal Carcinoma In Situ",
    "recurrent breast cancer": "Breast Cancer",
    "breast cancer": "Breast Cancer",
    "male breast cancer": "Breast Cancer",
}


def normalize_condition(raw):
    """Normalize a condition name extracted from treatment context."""
    key = raw.lower().strip()
    return CONDITION_CANONICAL.get(key, raw.strip())


def extract_treatment_relations(text, doc_id, qid):
    """Extract treatment-specific relations from answer text."""
    relations = []

    # "Treatment of [condition] may include [treatment]"
    blocks = re.split(r"(?:Treatment of|treatment for)\s+", text)
    for block in blocks[1:]:
        condition_m = re.match(r"([\w\s,'-]+?)(?:\s+may include|\s+is\s+a)", block)
        if not condition_m:
            continue
        condition = normalize_condition(condition_m.group(1).strip().rstrip(",."))

        treatments_found = re.findall(
            r"(chemotherapy|radiation therapy|surgery|mastectomy|lumpectomy|"
            r"targeted therapy|hormone therapy|immunotherapy|trastuzumab|"
            r"pertuzumab|tamoxifen|breast-conserving surgery|"
            r"sentinel lymph node biopsy|aromatase inhibitor|"
            r"selective estrogen receptor modulator|oophorectomy|"
            r"ovarian ablation|endocrine therapy|lapatinib|palbociclib|"
            r"ado-trastuzumab emtansine|fulvestrant)",
            block, re.IGNORECASE
        )
        for t in treatments_found:
            relations.append({
                "subject": normalize_entity(t),
                "relation": "treats",
                "object": condition,
                "source_doc": doc_id,
                "source_qa": qid,
                "data_grounded": True,
            })

    # "[Drug] is used to treat" or "[Treatment] may be given"
    treat_patterns = [
        (r"(\b(?:trastuzumab|pertuzumab|tamoxifen|anastrozole|letrozole|"
         r"exemestane|fulvestrant|palbociclib|lapatinib|"
         r"ado-trastuzumab emtansine)\b)[^.]*?"
         r"(?:used to treat|treat|given for|given to)\s+"
         r"([\w\s-]+?)(?:\.|,|;|\band\b)",),
    ]
    for pat_tuple in treat_patterns:
        for m in re.finditer(pat_tuple[0], text, re.IGNORECASE):
            relations.append({
                "subject": normalize_entity(m.group(1)),
                "relation": "treats",
                "object": m.group(2).strip().rstrip(".,;"),
                "source_doc": doc_id,
                "source_qa": qid,
                "data_grounded": True,
            })

    return relations


def extract_side_effect_relations(text, doc_id, qid):
    """Extract side effect relations from treatment-related text."""
    relations = []

    side_effects = [
        "hair loss", "nausea", "vomiting", "fatigue", "heart failure",
        "blood clots", "premature menopause", "lymphedema", "bone loss",
        "hot flashes", "inflammation of the lung", "infertility",
        "weight gain", "joint pain", "mouth sores", "diarrhea",
        "second cancer", "heart problems",
    ]

    treatments = [
        "chemotherapy", "radiation therapy", "trastuzumab", "tamoxifen",
        "aromatase inhibitor", "hormone therapy", "targeted therapy",
        "pertuzumab", "lapatinib", "surgery", "mastectomy",
    ]

    for treatment in treatments:
        t_pattern = re.escape(treatment)
        # Find sentences mentioning treatment + side effects
        for m in re.finditer(
            rf"(?:side effects?\s+of\s+)?{t_pattern}[^.]*?(?:may (?:include|cause)|"
            r"side effects?[^.]*?include)[^.]+\.",
            text, re.IGNORECASE
        ):
            sentence = m.group()
            for se in side_effects:
                if re.search(re.escape(se), sentence, re.IGNORECASE):
                    relations.append({
                        "subject": normalize_entity(treatment),
                        "relation": "has_side_effect",
                        "object": normalize_entity(se),
                        "source_doc": doc_id,
                        "source_qa": qid,
                        "data_grounded": True,
                    })

    # "Late effects of [treatment] ... may include [effect]"
    for m in re.finditer(
        r"(?:Late effects? of|effects? of)\s+([\w\s]+?)\s+"
        r"(?:may include|include|are)[^.]*?\.",
        text, re.IGNORECASE
    ):
        treatment_name = m.group(1).strip()
        sentence = m.group()
        for se in side_effects:
            if re.search(re.escape(se), sentence, re.IGNORECASE):
                relations.append({
                    "subject": normalize_entity(treatment_name),
                    "relation": "has_side_effect",
                    "object": normalize_entity(se),
                    "source_doc": doc_id,
                    "source_qa": qid,
                    "data_grounded": True,
                })

    return relations


def extract_risk_factor_relations(text, doc_id, qid):
    """Extract risk factor relations from susceptibility/causes text."""
    relations = []

    risk_factors = {
        r"family history": "Family History",
        r"\bage\b": "Age",
        r"obesity": "Obesity",
        r"alcohol": "Alcohol Consumption",
        r"radiation.{0,10}(?:exposure|therapy|treatment)": "Radiation Exposure",
        r"dense breast": "Dense Breast Tissue",
        r"menstruat\w+\s+at\s+an?\s+early": "Early Menarche",
        r"menopause\s+at\s+a\s+late": "Late Menopause",
        r"never having given birth": "Nulliparity",
        r"older age at first birth": "Late First Pregnancy",
        r"hormone.{0,15}(?:replacement|therapy)": "Hormone Replacement Therapy",
        r"estrogen.{0,15}(?:exposure|made by the body)": "Exogenous Estrogen",
        r"personal history.{0,30}breast": "Personal History of Breast Disease",
        r"BRCA1": "BRCA1 Gene Mutation",
        r"BRCA2": "BRCA2 Gene Mutation",
        r"inherited.{0,15}gene": "Inherited Gene Mutation",
        r"benign.{0,10}breast disease": "Benign Breast Disease",
    }

    # Check if text discusses risk factors
    risk_context = re.search(
        r"risk factors?|increase.{0,10}risk|at risk", text, re.IGNORECASE
    )
    if risk_context:
        for pat, canonical in risk_factors.items():
            if re.search(pat, text, re.IGNORECASE):
                relations.append({
                    "subject": canonical,
                    "relation": "increases_risk_of",
                    "object": "Breast Cancer",
                    "source_doc": doc_id,
                    "source_qa": qid,
                    "data_grounded": True,
                })

    # Protective factors
    protective = {
        r"breastfeeding": "Breastfeeding",
        r"exercise": "Physical Exercise",
        r"early pregnancy": "Early Pregnancy",
        r"estrogen-only hormone therapy after.{0,20}hysterectomy":
            "Estrogen-Only HRT Post-Hysterectomy",
    }
    protect_context = re.search(
        r"protective factors?|decrease.{0,10}risk|reduce.{0,10}risk|prevent",
        text, re.IGNORECASE
    )
    if protect_context:
        for pat, canonical in protective.items():
            if re.search(pat, text, re.IGNORECASE):
                relations.append({
                    "subject": canonical,
                    "relation": "decreases_risk_of",
                    "object": "Breast Cancer",
                    "source_doc": doc_id,
                    "source_qa": qid,
                    "data_grounded": True,
                })

    return relations


def extract_detection_relations(text, doc_id, qid):
    """Extract diagnostic detection relations."""
    relations = []

    tests = {
        r"mammogra(?:m|phy)": "Mammogram",
        r"ultrasound": "Ultrasound",
        r"\bMRI\b": "MRI",
        r"\bbiopsy\b": "Biopsy",
        r"CT scan": "CT Scan",
        r"PET scan": "PET Scan",
        r"bone scan": "Bone Scan",
        r"clinical breast exam": "Clinical Breast Exam",
        r"physical exam": "Physical Exam",
        r"chest x-ray": "Chest X-Ray",
        r"sentinel lymph node biopsy": "Sentinel Lymph Node Biopsy",
        r"fine-needle aspiration": "Fine-Needle Aspiration Biopsy",
        r"core biopsy": "Core Biopsy",
        r"blood chemistry": "Blood Chemistry Studies",
        r"complete blood count": "Complete Blood Count",
        r"genetic test(?:ing)?": "Genetic Testing",
        r"Oncotype DX": "Oncotype DX",
        r"MammaPrint": "MammaPrint",
    }

    detection_context = re.search(
        r"(?:used to|may be used to|can)\s+(?:detect|find|diagnose|check|"
        r"determine|stage|look for)|tests?\s+(?:and procedures?\s+)?(?:may be "
        r"used|are used|include)",
        text, re.IGNORECASE
    )
    if detection_context:
        for pat, canonical in tests.items():
            if re.search(pat, text, re.IGNORECASE):
                relations.append({
                    "subject": canonical,
                    "relation": "detects",
                    "object": "Breast Cancer",
                    "source_doc": doc_id,
                    "source_qa": qid,
                    "data_grounded": True,
                })

    return relations


def extract_subtype_biomarker_relations(text, doc_id, qid):
    """Extract subtype-biomarker and subtype-treatment relations."""
    relations = []

    # HER2-positive → specific treatments
    if re.search(r"HER2[/-]?(?:positive|neu positive)", text, re.IGNORECASE):
        for drug in ["trastuzumab", "pertuzumab", "lapatinib",
                      "ado-trastuzumab emtansine"]:
            if re.search(drug, text, re.IGNORECASE):
                relations.append({
                    "subject": normalize_entity(drug),
                    "relation": "treats",
                    "object": "HER2-Positive Breast Cancer",
                    "source_doc": doc_id,
                    "source_qa": qid,
                    "data_grounded": True,
                })

    # Hormone receptor-positive → endocrine treatments
    if re.search(r"hormone receptor[- ]positive|ER[- ]positive", text, re.IGNORECASE):
        for drug in ["tamoxifen", "aromatase inhibitor", "fulvestrant",
                      "selective estrogen receptor modulator", "hormone therapy",
                      "endocrine therapy"]:
            if re.search(drug, text, re.IGNORECASE):
                relations.append({
                    "subject": normalize_entity(drug),
                    "relation": "treats",
                    "object": "Hormone Receptor-Positive Breast Cancer",
                    "source_doc": doc_id,
                    "source_qa": qid,
                    "data_grounded": True,
                })

    # Triple-negative → limited targeted options, relies on chemo
    if re.search(r"triple[- ]negative", text, re.IGNORECASE):
        if re.search(r"chemotherapy", text, re.IGNORECASE):
            relations.append({
                "subject": "Chemotherapy",
                "relation": "treats",
                "object": "Triple-Negative Breast Cancer",
                "source_doc": doc_id,
                "source_qa": qid,
                "data_grounded": True,
            })
        if re.search(r"immunotherapy", text, re.IGNORECASE):
            relations.append({
                "subject": "Immunotherapy",
                "relation": "treats",
                "object": "Triple-Negative Breast Cancer",
                "source_doc": doc_id,
                "source_qa": qid,
                "data_grounded": True,
            })

    # Biomarker characterization
    biomarker_subtype_pairs = [
        (r"HER2[/-]?positive", "HER2-Positive Breast Cancer", "HER2"),
        (r"estrogen receptor[- ]positive", "ER-Positive Breast Cancer",
         "Estrogen Receptor"),
        (r"progesterone receptor[- ]positive", "PR-Positive Breast Cancer",
         "Progesterone Receptor"),
        (r"triple[- ]negative", "Triple-Negative Breast Cancer", None),
    ]
    for pat, subtype, biomarker in biomarker_subtype_pairs:
        if re.search(pat, text, re.IGNORECASE) and biomarker:
            relations.append({
                "subject": subtype,
                "relation": "has_biomarker",
                "object": biomarker,
                "source_doc": doc_id,
                "source_qa": qid,
                "data_grounded": True,
            })

    return relations


def extract_relations_from_doc(doc):
    """Extract all relations from a single MedQuAD document."""
    relations = []

    for qa in doc["qa_pairs"]:
        doc_id = doc["id"]
        qid = qa["qid"]
        qtype = qa["qtype"]
        text = qa["answer"]

        if not text:
            continue

        # Always run subtype-biomarker extraction
        relations.extend(extract_subtype_biomarker_relations(text, doc_id, qid))

        # Use qtype to guide which extractors to apply
        if qtype in ("treatment", "research"):
            relations.extend(extract_treatment_relations(text, doc_id, qid))
            relations.extend(extract_side_effect_relations(text, doc_id, qid))
        elif qtype in ("susceptibility", "causes", "inheritance"):
            relations.extend(extract_risk_factor_relations(text, doc_id, qid))
        elif qtype in ("exams and tests",):
            relations.extend(extract_detection_relations(text, doc_id, qid))
        elif qtype == "symptoms":
            # Extract symptom-disease associations
            from parse_medquad import extract_concepts_from_text
            concepts = extract_concepts_from_text(text)
            for symptom in concepts.get("symptom", []):
                relations.append({
                    "subject": "Breast Cancer",
                    "relation": "has_symptom",
                    "object": symptom,
                    "source_doc": doc_id,
                    "source_qa": qid,
                    "data_grounded": True,
                })
        elif qtype == "prevention":
            relations.extend(extract_risk_factor_relations(text, doc_id, qid))
        elif qtype == "stages":
            from parse_medquad import extract_concepts_from_text
            concepts = extract_concepts_from_text(text)
            for stage in concepts.get("stage", []):
                relations.append({
                    "subject": "Breast Cancer",
                    "relation": "has_stage",
                    "object": stage,
                    "source_doc": doc_id,
                    "source_qa": qid,
                    "data_grounded": True,
                })
        elif qtype == "information":
            # General text -- run all extractors
            relations.extend(extract_treatment_relations(text, doc_id, qid))
            relations.extend(extract_risk_factor_relations(text, doc_id, qid))
            relations.extend(extract_detection_relations(text, doc_id, qid))
            relations.extend(extract_side_effect_relations(text, doc_id, qid))

    return relations


def deduplicate_relations(relations):
    """Remove duplicate relations, keeping the first occurrence."""
    seen = set()
    unique = []
    for r in relations:
        key = (r["subject"], r["relation"], r["object"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def main():
    docs_path = DATA_DIR / "medquad_breast_cancer.json"
    with open(docs_path) as f:
        docs = json.load(f)

    print(f"Processing {len(docs)} breast cancer documents...")

    all_relations = []
    for doc in docs:
        relations = extract_relations_from_doc(doc)
        all_relations.extend(relations)

    all_relations = deduplicate_relations(all_relations)

    # Group by relation type for summary
    by_type = defaultdict(list)
    for r in all_relations:
        by_type[r["relation"]].append(r)

    print(f"\nExtracted {len(all_relations)} unique relations:")
    for rtype, rels in sorted(by_type.items()):
        print(f"  {rtype}: {len(rels)}")
        for r in sorted(rels, key=lambda x: (x["subject"], x["object"]))[:5]:
            print(f"    {r['subject']} --{rtype}--> {r['object']}")
        if len(rels) > 5:
            print(f"    ... and {len(rels) - 5} more")

    output_path = DATA_DIR / "extracted_relations.json"
    with open(output_path, "w") as f:
        json.dump(all_relations, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
