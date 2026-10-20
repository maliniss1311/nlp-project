"""
Parse MedQuAD XML files across all subsets to extract breast-cancer concepts,
relations, and question-answer pairs for ontology construction.

Scans all 12 MedQuAD subdirectories with strict Focus-based filtering
to ensure only breast-cancer-relevant documents are included.
"""

import xml.etree.ElementTree as ET
import json
import os
import re
from pathlib import Path
from collections import defaultdict


PROJECT_DIR = Path(__file__).resolve().parents[2]
MEDQUAD_DIR = PROJECT_DIR / "MedQuAD"
OUTPUT_DIR = PROJECT_DIR / "data"

MEDQUAD_SUBSETS = [
    "1_CancerGov_QA",
    "2_GARD_QA",
    "3_GHR_QA",
    "4_MPlus_Health_Topics_QA",
    "5_NIDDK_QA",
    "6_NINDS_QA",
    "7_SeniorHealth_QA",
    "8_NHLBI_QA_XML",
    "9_CDC_QA",
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements",
]

# Focus values that indicate breast-cancer-relevant documents.
# Matched case-insensitively against the <Focus> tag.
BREAST_FOCUS_PATTERNS = [
    r"breast cancer",
    r"male breast cancer",
    r"inflammatory breast cancer",
    r"metaplastic carcinoma of the breast",
    r"breast cancer screen",
    r"breast cancer stag",
    r"breast cancer risk",
    r"breast self.exam",
    r"breast biopsy",
    r"breast MRI",
    r"breast PET",
    r"breast ultrasound",
    r"breast skin and nipple",
    r"breast lump$",
    r"breast lump removal",
    r"breast pain",
    r"breast radiation",
    r"breast reconstruct",
    r"mastectomy",
    r"mammogra",
    r"hormone therapy for breast",
    r"BRCA1 hereditary breast",
    r"BRCA2 hereditary breast",
    r"fibroadenoma.+breast",
    r"fibrocystic breast",
    r"nipple discharge",
    r"ductal carcinoma",
    r"lobular carcinoma",
    r"Paget disease of the nipple",
]

# Canonical forms for extracted concepts -- maps raw regex matches to
# a normalized name to eliminate duplicates like "Age"/"age".
CONCEPT_CANONICAL = {
    # Subtypes
    "ductal carcinoma in situ (dcis)": "Ductal Carcinoma In Situ",
    "dcis": "Ductal Carcinoma In Situ",
    "ductal carcinoma": "Invasive Ductal Carcinoma",
    "lobular carcinoma in situ (lcis)": "Lobular Carcinoma In Situ",
    "lcis": "Lobular Carcinoma In Situ",
    "lobular carcinoma": "Invasive Lobular Carcinoma",
    "inflammatory breast cancer": "Inflammatory Breast Cancer",
    "triple-negative": "Triple-Negative Breast Cancer",
    "triple negative": "Triple-Negative Breast Cancer",
    "invasive breast cancer": "Invasive Breast Cancer",
    "metastatic breast cancer": "Metastatic Breast Cancer",
    "paget disease of the nipple": "Paget Disease of the Nipple",
    # Biomarkers
    "her2": "HER2",
    "her2/neu": "HER2/neu",
    "estrogen receptor": "Estrogen Receptor",
    "progesterone receptor": "Progesterone Receptor",
    "brca1": "BRCA1",
    "brca2": "BRCA2",
    "oncotype dx": "Oncotype DX",
    "mammaprint": "MammaPrint",
    # Treatments
    "chemotherapy": "Chemotherapy",
    "radiation therapy": "Radiation Therapy",
    "surgery": "Surgery",
    "mastectomy": "Mastectomy",
    "lumpectomy": "Lumpectomy",
    "targeted therapy": "Targeted Therapy",
    "hormone therapy": "Hormone Therapy",
    "endocrine therapy": "Endocrine Therapy",
    "immunotherapy": "Immunotherapy",
    "trastuzumab": "Trastuzumab",
    "pertuzumab": "Pertuzumab",
    "tamoxifen": "Tamoxifen",
    "aromatase inhibitor": "Aromatase Inhibitor",
    "selective estrogen receptor modulator": "Selective Estrogen Receptor Modulator",
    "oophorectomy": "Oophorectomy",
    "ovarian ablation": "Ovarian Ablation",
    "stem cell transplant": "Stem Cell Transplant",
    "sentinel lymph node biopsy": "Sentinel Lymph Node Biopsy",
    "biologic therapy": "Biologic Therapy",
    # Diagnostics
    "mammogram": "Mammogram",
    "mammography": "Mammogram",
    "ultrasound": "Ultrasound",
    "mri": "MRI",
    "biopsy": "Biopsy",
    "ct scan": "CT Scan",
    "pet scan": "PET Scan",
    "bone scan": "Bone Scan",
    "clinical breast exam": "Clinical Breast Exam",
    "physical exam": "Physical Exam",
    "blood chemistry": "Blood Chemistry Studies",
    "complete blood count": "Complete Blood Count",
    # Symptoms
    "lump": "Breast Lump",
    "thickening": "Breast Thickening",
    "dimple": "Skin Dimpling",
    "puckering": "Skin Puckering",
    "nipple turned inward": "Nipple Retraction",
    "nipple discharge": "Nipple Discharge",
    "swollen skin": "Skin Swelling",
    "peau d'orange": "Peau d'Orange",
    "pain": "Breast Pain",
    "fatigue": "Fatigue",
    "weight loss": "Weight Loss",
    # Risk factors
    "family history": "Family History",
    "obesity": "Obesity",
    "alcohol": "Alcohol Consumption",
    "radiation exposure": "Radiation Exposure",
    "dense breast": "Dense Breast Tissue",
    "never having given birth": "Nulliparity",
    # Stages
    "carcinoma in situ": "Stage 0 (Carcinoma In Situ)",
    "locally advanced": "Locally Advanced",
    "metastatic": "Metastatic",
}


def normalize_concept(raw_text):
    """Normalize a raw regex match to a canonical concept name."""
    key = raw_text.lower().strip()
    if key in CONCEPT_CANONICAL:
        return CONCEPT_CANONICAL[key]
    # Title case fallback for unknown terms
    return raw_text.strip().title() if raw_text[0].islower() else raw_text.strip()


def parse_xml_file(filepath):
    """Parse a single MedQuAD XML file and return structured data."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    doc = {
        "id": root.get("id", ""),
        "source": root.get("source", ""),
        "url": root.get("url", ""),
        "focus": root.findtext("Focus", ""),
        "subset": filepath.parent.name,
        "umls_cuis": [],
        "semantic_types": [],
        "semantic_group": "",
        "qa_pairs": [],
    }

    umls = root.find(".//UMLS")
    if umls is not None:
        doc["umls_cuis"] = [c.text for c in umls.findall(".//CUI") if c.text]
        doc["semantic_types"] = [
            s.text for s in umls.findall(".//SemanticType") if s.text
        ]
        doc["semantic_group"] = umls.findtext("SemanticGroup", "")

    for qp in root.findall(".//QAPair"):
        q_elem = qp.find("Question")
        a_elem = qp.find("Answer")
        if q_elem is not None and a_elem is not None:
            doc["qa_pairs"].append({
                "qid": q_elem.get("qid", ""),
                "qtype": q_elem.get("qtype", ""),
                "question": (q_elem.text or "").strip(),
                "answer": (a_elem.text or "").strip(),
            })

    return doc


def is_breast_cancer_focused(doc):
    """Strict filter: only documents whose Focus tag matches breast cancer topics."""
    focus = doc["focus"].strip()
    if not focus:
        return False
    for pattern in BREAST_FOCUS_PATTERNS:
        if re.search(pattern, focus, re.IGNORECASE):
            return True
    return False


def extract_concepts_from_text(text):
    """Extract and normalize medical concepts from answer text."""
    concepts = defaultdict(set)

    # Breast cancer subtypes -- (pattern, canonical_name) pairs
    subtype_patterns = [
        (r"ductal carcinoma in situ\s*\(DCIS\)", "Ductal Carcinoma In Situ"),
        (r"ductal carcinoma(?! in situ)", "Invasive Ductal Carcinoma"),
        (r"lobular carcinoma in situ\s*\(LCIS\)", "Lobular Carcinoma In Situ"),
        (r"lobular carcinoma(?! in situ)", "Invasive Lobular Carcinoma"),
        (r"inflammatory breast cancer", "Inflammatory Breast Cancer"),
        (r"triple[- ]negative(?:\s+breast\s+cancer)?", "Triple-Negative Breast Cancer"),
        (r"HER2[/-]?positive(?:\s+breast\s+cancer)?", "HER2-Positive Breast Cancer"),
        (r"HER2[/-]?negative(?:\s+breast\s+cancer)?", "HER2-Negative Breast Cancer"),
        (r"hormone receptor[- ]positive", "Hormone Receptor-Positive Breast Cancer"),
        (r"hormone receptor[- ]negative", "Hormone Receptor-Negative Breast Cancer"),
        (r"estrogen receptor[- ]positive", "ER-Positive Breast Cancer"),
        (r"estrogen receptor[- ]negative", "ER-Negative Breast Cancer"),
        (r"progesterone receptor[- ]positive", "PR-Positive Breast Cancer"),
        (r"progesterone receptor[- ]negative", "PR-Negative Breast Cancer"),
        (r"Paget disease of the nipple", "Paget Disease of the Nipple"),
        (r"invasive breast cancer", "Invasive Breast Cancer"),
        (r"metastatic breast cancer", "Metastatic Breast Cancer"),
        (r"metaplastic carcinoma", "Metaplastic Carcinoma"),
        (r"medullary carcinoma", "Medullary Carcinoma"),
        (r"tubular carcinoma", "Tubular Carcinoma"),
        (r"mucinous carcinoma", "Mucinous Carcinoma"),
        (r"papillary carcinoma", "Papillary Carcinoma"),
    ]
    for pat, canonical in subtype_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["breast_cancer_subtype"].add(canonical)

    # Biomarkers
    biomarker_patterns = [
        (r"estrogen receptor(?!\s*[- ](?:positive|negative))", "Estrogen Receptor"),
        (r"progesterone receptor(?!\s*[- ](?:positive|negative))", "Progesterone Receptor"),
        (r"HER2/neu", "HER2/neu"),
        (r"HER2(?!\s*[/-]?(?:positive|negative|/neu))", "HER2"),
        (r"BRCA1", "BRCA1"), (r"BRCA2", "BRCA2"), (r"Ki-?67", "Ki-67"),
        (r"Oncotype DX", "Oncotype DX"), (r"MammaPrint", "MammaPrint"),
        (r"tumor protein p53|TP53", "TP53"),
    ]
    for pat, canonical in biomarker_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["biomarker"].add(canonical)

    # Treatments
    treatment_patterns = [
        (r"chemotherapy", "Chemotherapy"),
        (r"radiation therapy", "Radiation Therapy"),
        (r"(?<!\w)surgery", "Surgery"),
        (r"mastectomy", "Mastectomy"),
        (r"lumpectomy", "Lumpectomy"),
        (r"breast-conserving surgery", "Breast-Conserving Surgery"),
        (r"sentinel lymph node biopsy", "Sentinel Lymph Node Biopsy"),
        (r"lymph node dissection", "Lymph Node Dissection"),
        (r"stem cell transplant", "Stem Cell Transplant"),
        (r"targeted therapy", "Targeted Therapy"),
        (r"hormone therapy", "Hormone Therapy"),
        (r"endocrine therapy", "Endocrine Therapy"),
        (r"immunotherapy", "Immunotherapy"),
        (r"biologic therapy", "Biologic Therapy"),
        (r"trastuzumab", "Trastuzumab"),
        (r"pertuzumab", "Pertuzumab"),
        (r"tamoxifen", "Tamoxifen"),
        (r"anastrozole", "Anastrozole"),
        (r"letrozole", "Letrozole"),
        (r"exemestane", "Exemestane"),
        (r"fulvestrant", "Fulvestrant"),
        (r"palbociclib", "Palbociclib"),
        (r"lapatinib", "Lapatinib"),
        (r"aromatase inhibitor", "Aromatase Inhibitor"),
        (r"selective estrogen receptor modulator", "Selective Estrogen Receptor Modulator"),
        (r"neoadjuvant", "Neoadjuvant Therapy"),
        (r"adjuvant(?!\s)", "Adjuvant Therapy"),
        (r"oophorectomy", "Oophorectomy"),
        (r"ovarian ablation", "Ovarian Ablation"),
        (r"ado-trastuzumab emtansine", "Ado-Trastuzumab Emtansine"),
    ]
    for pat, canonical in treatment_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["treatment"].add(canonical)

    # Diagnostic tests
    diagnostic_patterns = [
        (r"mammogra(?:m|phy)", "Mammogram"),
        (r"ultrasound", "Ultrasound"),
        (r"\bMRI\b", "MRI"),
        (r"\bbiopsy\b(?!.*(?:sentinel|core|excisional|fine))", "Biopsy"),
        (r"CT scan", "CT Scan"),
        (r"PET scan", "PET Scan"),
        (r"bone scan", "Bone Scan"),
        (r"clinical breast exam", "Clinical Breast Exam"),
        (r"physical exam", "Physical Exam"),
        (r"blood chemistry", "Blood Chemistry Studies"),
        (r"complete blood count", "Complete Blood Count"),
        (r"chest x-ray", "Chest X-Ray"),
        (r"fine-needle aspiration", "Fine-Needle Aspiration Biopsy"),
        (r"core biopsy", "Core Biopsy"),
        (r"excisional biopsy", "Excisional Biopsy"),
        (r"genetic test(?:ing)?", "Genetic Testing"),
    ]
    for pat, canonical in diagnostic_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["diagnostic_test"].add(canonical)

    # Symptoms and signs
    symptom_patterns = [
        (r"\blump\b", "Breast Lump"),
        (r"thickening", "Breast Thickening"),
        (r"dimpl(?:e|ing)", "Skin Dimpling"),
        (r"puckering", "Skin Puckering"),
        (r"nipple\s+turned\s+inward", "Nipple Retraction"),
        (r"nipple\s+retract", "Nipple Retraction"),
        (r"nipple\s+discharge", "Nipple Discharge"),
        (r"swollen\s+skin", "Skin Swelling"),
        (r"peau d.orange", "Peau d'Orange"),
        (r"\bbreast\s+pain\b", "Breast Pain"),
        (r"\bfatigue\b", "Fatigue"),
        (r"weight loss", "Weight Loss"),
        (r"lymphedema", "Lymphedema"),
        (r"scaly.{0,6}skin", "Scaly Skin"),
        (r"redness", "Breast Redness"),
        (r"warmth", "Breast Warmth"),
        (r"swelling of.{0,10}breast", "Breast Swelling"),
    ]
    for pat, canonical in symptom_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["symptom"].add(canonical)

    # Risk factors
    risk_patterns = [
        (r"family history", "Family History"),
        (r"\bage\b", "Age"),
        (r"obesity", "Obesity"),
        (r"alcohol", "Alcohol Consumption"),
        (r"radiation exposure", "Radiation Exposure"),
        (r"dense breast", "Dense Breast Tissue"),
        (r"menstruat\w+\s+at\s+an?\s+early", "Early Menarche"),
        (r"menopause\s+at\s+a\s+late", "Late Menopause"),
        (r"never having given birth", "Nulliparity"),
        (r"older age at first birth", "Late First Pregnancy"),
        (r"hormone replacement therapy", "Hormone Replacement Therapy"),
        (r"taking hormones such as estrogen", "Exogenous Estrogen"),
        (r"personal history of.{0,30}breast", "Personal History of Breast Disease"),
        (r"benign.{0,10}breast disease", "Benign Breast Disease"),
    ]
    for pat, canonical in risk_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["risk_factor"].add(canonical)

    # Protective factors
    protective_patterns = [
        (r"breastfeeding", "Breastfeeding"),
        (r"exercise", "Physical Exercise"),
        (r"early pregnancy", "Early Pregnancy"),
    ]
    for pat, canonical in protective_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["protective_factor"].add(canonical)

    # Genetic factors
    genetic_patterns = [
        (r"BRCA1", "BRCA1 Gene Mutation"),
        (r"BRCA2", "BRCA2 Gene Mutation"),
        (r"inherited\s+(?:gene\s+)?mutation", "Inherited Gene Mutation"),
        (r"hereditary\s+breast", "Hereditary Breast Cancer"),
        (r"Li-Fraumeni", "Li-Fraumeni Syndrome"),
        (r"Cowden syndrome", "Cowden Syndrome"),
    ]
    for pat, canonical in genetic_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["genetic_factor"].add(canonical)

    # Stages
    stage_patterns = [
        (r"Stage\s+0", "Stage 0"),
        (r"Stage\s+IA\b", "Stage IA"),
        (r"Stage\s+IB\b", "Stage IB"),
        (r"Stage\s+I\b(?!I)", "Stage I"),
        (r"Stage\s+IIA\b", "Stage IIA"),
        (r"Stage\s+IIB\b", "Stage IIB"),
        (r"Stage\s+II\b(?!I)", "Stage II"),
        (r"Stage\s+IIIA\b", "Stage IIIA"),
        (r"Stage\s+IIIB\b", "Stage IIIB"),
        (r"Stage\s+IIIC\b", "Stage IIIC"),
        (r"Stage\s+III\b(?![\w])", "Stage III"),
        (r"Stage\s+IV\b", "Stage IV"),
        (r"carcinoma in situ", "Stage 0 (Carcinoma In Situ)"),
        (r"locally advanced", "Locally Advanced"),
        (r"\bmetastatic\b(?!\s+breast)", "Metastatic"),
    ]
    for pat, canonical in stage_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["stage"].add(canonical)

    # Side effects
    side_effect_patterns = [
        (r"hair loss", "Hair Loss"),
        (r"nausea", "Nausea"),
        (r"vomiting", "Vomiting"),
        (r"heart failure", "Heart Failure"),
        (r"blood clots", "Blood Clots"),
        (r"premature menopause", "Premature Menopause"),
        (r"lymphedema", "Lymphedema"),
        (r"inflammation of the lung", "Lung Inflammation"),
        (r"infertility", "Infertility"),
        (r"bone loss|osteoporosis", "Bone Loss"),
        (r"hot flashes", "Hot Flashes"),
    ]
    for pat, canonical in side_effect_patterns:
        if re.search(pat, text, re.IGNORECASE):
            concepts["side_effect"].add(canonical)

    return {k: sorted(v) for k, v in concepts.items()}


def parse_all_medquad():
    """Parse all MedQuAD XML files across subsets and extract breast-cancer data."""
    breast_docs = []
    all_concepts = defaultdict(set)
    subset_stats = {}

    for subset in MEDQUAD_SUBSETS:
        subset_dir = MEDQUAD_DIR / subset
        if not subset_dir.exists():
            continue

        xml_files = sorted(subset_dir.glob("*.xml"))
        subset_count = 0

        for f in xml_files:
            try:
                doc = parse_xml_file(f)
                if is_breast_cancer_focused(doc):
                    breast_docs.append(doc)
                    subset_count += 1
                    for qa in doc["qa_pairs"]:
                        concepts = extract_concepts_from_text(qa["answer"])
                        for category, items in concepts.items():
                            all_concepts[category].update(items)
            except ET.ParseError as e:
                print(f"  Parse error in {subset}/{f.name}: {e}")

        if subset_count > 0:
            subset_stats[subset] = subset_count

    print(f"Breast-cancer-focused documents: {len(breast_docs)}")
    print(f"\nDocuments by MedQuAD subset:")
    for subset, count in sorted(subset_stats.items()):
        print(f"  {subset}: {count}")

    print(f"\nExtracted concept categories:")
    for cat, items in sorted(all_concepts.items()):
        print(f"  {cat}: {len(items)} unique concepts")
        for item in sorted(items):
            print(f"    - {item}")

    return breast_docs, {k: sorted(v) for k, v in all_concepts.items()}


def extract_question_types(breast_docs):
    """Summarize question types found in breast cancer documents."""
    qtypes = defaultdict(int)
    for doc in breast_docs:
        for qa in doc["qa_pairs"]:
            qtype = qa["qtype"].strip()
            if qtype:
                qtypes[qtype] += 1
    return dict(sorted(qtypes.items(), key=lambda x: -x[1]))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    breast_docs, concepts = parse_all_medquad()

    output_path = OUTPUT_DIR / "medquad_breast_cancer.json"
    with open(output_path, "w") as f:
        json.dump(breast_docs, f, indent=2)
    print(f"\nSaved {len(breast_docs)} breast cancer docs to {output_path}")

    concepts_path = OUTPUT_DIR / "extracted_concepts.json"
    with open(concepts_path, "w") as f:
        json.dump(concepts, f, indent=2)
    print(f"Saved extracted concepts to {concepts_path}")

    qtypes = extract_question_types(breast_docs)
    print(f"\nQuestion types in breast cancer docs:")
    for qt, count in qtypes.items():
        print(f"  {qt}: {count}")


if __name__ == "__main__":
    main()
