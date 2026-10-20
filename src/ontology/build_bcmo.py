"""
Build the Breast Cancer Misinformation Ontology (BCMO) v2.

Merges MedQuAD-extracted concepts and relations with expert-curated
knowledge from NCI Thesaurus, UMLS, and NCCN guidelines. Every fact
tracks whether it came from data extraction or expert curation.

Output: data/bcmo_v2.json
"""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_extracted_data():
    """Load concepts and relations extracted from MedQuAD."""
    concepts, relations = {}, []
    cp = DATA_DIR / "extracted_concepts.json"
    rp = DATA_DIR / "extracted_relations.json"
    if cp.exists():
        with open(cp) as f:
            concepts = json.load(f)
    if rp.exists():
        with open(rp) as f:
            relations = json.load(f)
    return concepts, relations


def build_concept_classes(extracted_concepts):
    """Build concept classes merging expert definitions with extracted data."""

    def _merge_instances(expert_instances, extracted_names):
        """Add extracted names not already in expert list."""
        existing = set()
        for inst in expert_instances:
            if isinstance(inst, str):
                existing.add(inst.lower())
            elif isinstance(inst, dict):
                existing.add(inst["name"].lower())
                for a in inst.get("aliases", []):
                    existing.add(a.lower())
        added = []
        for name in extracted_names:
            if name.lower() not in existing:
                added.append({"name": name, "source": "MedQuAD"})
        return expert_instances + added

    # Expert-curated concept definitions
    classes = {
        "BreastCancer": {
            "description": "The primary disease entity",
            "umls_cui": "C0006142",
            "nci_code": "C4872",
            "instances": [{"name": "Breast Cancer", "source": "expert"}],
        },
        "BreastCancerSubtype": {
            "description": "Histological and molecular subtypes of breast cancer",
            "instances": [
                {"name": "Invasive Ductal Carcinoma", "aliases": ["IDC", "ductal carcinoma", "infiltrating ductal carcinoma"], "umls_cui": "C0007124", "description": "Most common type (~70-80%), begins in milk duct lining", "source": "expert"},
                {"name": "Invasive Lobular Carcinoma", "aliases": ["ILC", "lobular carcinoma"], "umls_cui": "C0279565", "description": "Second most common (~10-15%), begins in lobules", "source": "expert"},
                {"name": "Ductal Carcinoma In Situ", "aliases": ["DCIS"], "umls_cui": "C0007124", "description": "Non-invasive; abnormal cells in duct lining. Stage 0.", "source": "expert"},
                {"name": "Lobular Carcinoma In Situ", "aliases": ["LCIS"], "umls_cui": "C0279563", "description": "Abnormal cells in lobules; increases risk", "source": "expert"},
                {"name": "Inflammatory Breast Cancer", "aliases": ["IBC"], "umls_cui": "C0278601", "description": "Rare, aggressive; warm, red, swollen breast", "source": "expert"},
                {"name": "Triple-Negative Breast Cancer", "aliases": ["TNBC", "triple negative"], "umls_cui": "C3539878", "description": "ER-, PR-, HER2-. No targeted hormone/HER2 therapy.", "source": "expert"},
                {"name": "HER2-Positive Breast Cancer", "aliases": ["HER2+"], "description": "Overexpresses HER2 protein", "source": "expert"},
                {"name": "Hormone Receptor-Positive Breast Cancer", "aliases": ["HR+", "ER+/PR+"], "description": "ER+ and/or PR+. Responds to endocrine therapy.", "source": "expert"},
                {"name": "Paget Disease of the Nipple", "aliases": ["Paget's disease"], "umls_cui": "C0030185", "description": "Rare form affecting nipple skin", "source": "expert"},
                {"name": "Metastatic Breast Cancer", "aliases": ["Stage IV breast cancer", "advanced breast cancer"], "description": "Spread to distant organs", "source": "expert"},
            ],
        },
        "StageAndGrade": {
            "description": "Cancer staging (TNM) and histologic grading",
            "instances": [
                {"name": "Stage 0", "aliases": ["carcinoma in situ"], "description": "Non-invasive (DCIS, LCIS, Paget)", "source": "expert"},
                {"name": "Stage I", "aliases": ["Stage IA", "Stage IB"], "description": "Small tumor (<=2cm), limited node involvement", "source": "expert"},
                {"name": "Stage II", "aliases": ["Stage IIA", "Stage IIB"], "description": "Larger tumor or limited lymph node spread", "source": "expert"},
                {"name": "Stage III", "aliases": ["Stage IIIA", "Stage IIIB", "Stage IIIC", "locally advanced"], "description": "Extensive local/regional disease", "source": "expert"},
                {"name": "Stage IV", "aliases": ["metastatic"], "description": "Distant metastasis", "source": "expert"},
                {"name": "Grade 1", "aliases": ["low grade", "well differentiated"], "description": "Slow-growing", "source": "expert"},
                {"name": "Grade 2", "aliases": ["intermediate grade"], "description": "Intermediate growth rate", "source": "expert"},
                {"name": "Grade 3", "aliases": ["high grade", "poorly differentiated"], "description": "Fast-growing", "source": "expert"},
            ],
        },
        "Symptom": {
            "description": "Signs and symptoms of breast cancer",
            "instances": [
                {"name": "Breast Lump", "aliases": ["lump", "mass", "thickening"], "source": "expert"},
                {"name": "Nipple Discharge", "aliases": ["fluid from nipple", "bloody discharge"], "source": "expert"},
                {"name": "Nipple Retraction", "aliases": ["nipple turned inward", "retracted nipple"], "source": "expert"},
                {"name": "Skin Dimpling", "aliases": ["dimple", "puckering", "peau d'orange"], "source": "expert"},
                {"name": "Breast Skin Changes", "aliases": ["scaly skin", "red skin", "swollen skin"], "source": "expert"},
                {"name": "Breast Pain", "aliases": ["mastalgia"], "source": "expert"},
                {"name": "Bone Pain", "description": "May indicate metastasis", "source": "expert"},
                {"name": "Axillary Lymphadenopathy", "aliases": ["swollen lymph nodes", "underarm lump"], "source": "expert"},
                {"name": "Fatigue", "aliases": ["tiredness"], "source": "expert"},
                {"name": "Weight Loss", "aliases": ["unexplained weight loss"], "source": "expert"},
            ],
        },
        "RiskFactor": {
            "description": "Factors that increase breast cancer risk",
            "instances": [
                {"name": "Family History", "aliases": ["family history of breast cancer"], "evidence_strength": "strong", "source": "expert"},
                {"name": "BRCA1/BRCA2 Mutation", "aliases": ["BRCA mutation", "hereditary breast cancer"], "evidence_strength": "strong", "source": "expert"},
                {"name": "Age", "aliases": ["older age", "aging"], "evidence_strength": "strong", "source": "expert"},
                {"name": "Dense Breast Tissue", "aliases": ["dense breasts"], "evidence_strength": "moderate", "source": "expert"},
                {"name": "Obesity", "aliases": ["overweight"], "evidence_strength": "moderate", "source": "expert"},
                {"name": "Alcohol Consumption", "aliases": ["drinking alcohol"], "evidence_strength": "moderate", "source": "expert"},
                {"name": "Hormone Replacement Therapy", "aliases": ["HRT", "estrogen + progestin"], "evidence_strength": "moderate", "source": "expert"},
                {"name": "Early Menarche", "aliases": ["menarche before 12", "menstruating at an early age"], "evidence_strength": "moderate", "source": "expert"},
                {"name": "Late Menopause", "aliases": ["menopause after 55"], "evidence_strength": "moderate", "source": "expert"},
                {"name": "Nulliparity", "aliases": ["never having given birth"], "evidence_strength": "moderate", "source": "expert"},
                {"name": "Radiation Exposure", "aliases": ["previous radiation to chest"], "evidence_strength": "strong", "source": "expert"},
                {"name": "Personal History of Breast Disease", "aliases": ["personal history"], "evidence_strength": "strong", "source": "expert"},
            ],
        },
        "ProtectiveFactor": {
            "description": "Factors that decrease breast cancer risk",
            "instances": [
                {"name": "Physical Exercise", "aliases": ["regular exercise", "physical activity"], "evidence_strength": "moderate", "source": "expert"},
                {"name": "Breastfeeding", "evidence_strength": "moderate", "source": "expert"},
                {"name": "Early Pregnancy", "aliases": ["first pregnancy before 30"], "evidence_strength": "moderate", "source": "expert"},
                {"name": "Prophylactic Mastectomy", "aliases": ["risk-reducing mastectomy"], "evidence_strength": "strong", "source": "expert"},
                {"name": "Prophylactic Oophorectomy", "aliases": ["ovarian ablation", "risk-reducing oophorectomy"], "evidence_strength": "strong", "source": "expert"},
                {"name": "Selective Estrogen Receptor Modulators", "aliases": ["SERMs", "tamoxifen for prevention"], "evidence_strength": "strong", "source": "expert"},
                {"name": "Aromatase Inhibitors for Prevention", "evidence_strength": "strong", "source": "expert"},
                {"name": "Estrogen-Only HRT Post-Hysterectomy", "description": "Estrogen-only therapy after hysterectomy may lower risk", "evidence_strength": "moderate", "source": "MedQuAD"},
            ],
        },
        "GeneticFactor": {
            "description": "Genes and mutations associated with breast cancer",
            "instances": [
                {"name": "BRCA1", "umls_cui": "C0376571", "description": "Tumor suppressor; mutations increase risk to 45-72%", "source": "expert"},
                {"name": "BRCA2", "umls_cui": "C0376572", "description": "Tumor suppressor; mutations increase risk to 45-69%", "source": "expert"},
                {"name": "TP53", "aliases": ["Li-Fraumeni syndrome"], "description": "Rare; increases risk of multiple cancers", "source": "expert"},
                {"name": "PALB2", "description": "Partner of BRCA2; moderate-to-high risk", "source": "expert"},
                {"name": "ATM", "description": "Moderate-risk breast cancer gene", "source": "expert"},
                {"name": "CHEK2", "description": "Moderate-risk breast cancer gene", "source": "expert"},
            ],
        },
        "Biomarker": {
            "description": "Molecular markers for diagnosis, prognosis, and treatment selection",
            "instances": [
                {"name": "Estrogen Receptor", "aliases": ["ER", "ER+", "ER-"], "umls_cui": "C0034804", "source": "expert"},
                {"name": "Progesterone Receptor", "aliases": ["PR", "PR+", "PR-", "PgR"], "umls_cui": "C0034833", "source": "expert"},
                {"name": "HER2/neu", "aliases": ["HER2", "ERBB2", "HER2+", "HER2-"], "umls_cui": "C0069515", "source": "expert"},
                {"name": "Ki-67", "description": "Proliferation marker", "source": "expert"},
                {"name": "Oncotype DX", "description": "21-gene recurrence score for HR+/HER2- early BC", "source": "expert"},
                {"name": "MammaPrint", "description": "70-gene signature for recurrence risk", "source": "expert"},
                {"name": "PD-L1", "description": "Immune checkpoint marker; guides immunotherapy in TNBC", "source": "expert"},
            ],
        },
        "DiagnosticTest": {
            "description": "Tests used to detect and diagnose breast cancer",
            "instances": [
                {"name": "Mammogram", "aliases": ["mammography"], "description": "X-ray imaging; primary screening tool", "source": "expert"},
                {"name": "Ultrasound", "aliases": ["breast ultrasound", "sonogram"], "description": "Sound waves; for dense breasts", "source": "expert"},
                {"name": "MRI", "aliases": ["breast MRI"], "description": "For high-risk screening", "source": "expert"},
                {"name": "Biopsy", "aliases": ["core biopsy", "needle biopsy", "excisional biopsy"], "description": "Tissue examination", "source": "expert"},
                {"name": "Fine-Needle Aspiration Biopsy", "aliases": ["FNA", "FNA biopsy"], "source": "expert"},
                {"name": "Sentinel Lymph Node Biopsy", "aliases": ["SLNB"], "description": "Checks lymph node spread", "source": "expert"},
                {"name": "Clinical Breast Exam", "aliases": ["CBE"], "source": "expert"},
                {"name": "PET Scan", "aliases": ["PET/CT"], "description": "Metabolically active cancer cells", "source": "expert"},
                {"name": "CT Scan", "aliases": ["CAT scan"], "description": "Cross-sectional imaging for staging", "source": "expert"},
                {"name": "Bone Scan", "description": "Checks for bone metastasis", "source": "expert"},
                {"name": "Genetic Testing", "aliases": ["BRCA testing"], "description": "Hereditary cancer gene tests", "source": "expert"},
                {"name": "Immunohistochemistry", "aliases": ["IHC"], "description": "Tests ER, PR, HER2 status", "source": "expert"},
                {"name": "FISH Test", "aliases": ["fluorescence in situ hybridization"], "description": "Confirms HER2 amplification", "source": "expert"},
            ],
        },
        "Treatment": {
            "description": "Therapeutic interventions for breast cancer",
            "instances": [
                {"name": "Surgery", "subcategories": ["Lumpectomy", "Mastectomy"], "source": "expert"},
                {"name": "Lumpectomy", "aliases": ["breast-conserving surgery", "partial mastectomy"], "description": "Removes tumor and margin", "source": "expert"},
                {"name": "Mastectomy", "aliases": ["total mastectomy", "modified radical mastectomy"], "description": "Removes entire breast", "source": "expert"},
                {"name": "Chemotherapy", "aliases": ["chemo"], "description": "Systemic drug treatment", "source": "expert"},
                {"name": "Radiation Therapy", "aliases": ["radiotherapy", "radiation"], "description": "High-energy rays to kill cancer cells", "source": "expert"},
                {"name": "Endocrine Therapy", "aliases": ["hormone therapy", "hormonal therapy"], "description": "Blocks hormones fueling HR+ growth", "source": "expert"},
                {"name": "Targeted Therapy", "description": "Drugs targeting specific molecular features", "source": "expert"},
                {"name": "Immunotherapy", "aliases": ["immune checkpoint inhibitor"], "description": "Stimulates immune attack on cancer", "source": "expert"},
                {"name": "Neoadjuvant Therapy", "description": "Treatment before surgery", "source": "expert"},
                {"name": "Adjuvant Therapy", "description": "Treatment after surgery", "source": "expert"},
                {"name": "Stem Cell Transplant", "aliases": ["bone marrow transplant"], "source": "expert"},
            ],
        },
        "Medication": {
            "description": "Specific drugs used in breast cancer treatment",
            "instances": [
                {"name": "Tamoxifen", "class": "SERM", "target": "ER", "source": "expert"},
                {"name": "Anastrozole", "aliases": ["Arimidex"], "class": "Aromatase Inhibitor", "source": "expert"},
                {"name": "Letrozole", "aliases": ["Femara"], "class": "Aromatase Inhibitor", "source": "expert"},
                {"name": "Exemestane", "aliases": ["Aromasin"], "class": "Aromatase Inhibitor", "source": "expert"},
                {"name": "Trastuzumab", "aliases": ["Herceptin"], "class": "Monoclonal Antibody", "target": "HER2", "source": "expert"},
                {"name": "Pertuzumab", "aliases": ["Perjeta"], "class": "Monoclonal Antibody", "target": "HER2", "source": "expert"},
                {"name": "Ado-Trastuzumab Emtansine", "aliases": ["Kadcyla", "T-DM1"], "class": "Antibody-Drug Conjugate", "target": "HER2", "source": "expert"},
                {"name": "Trastuzumab Deruxtecan", "aliases": ["Enhertu", "T-DXd"], "class": "Antibody-Drug Conjugate", "target": "HER2", "source": "expert"},
                {"name": "Palbociclib", "aliases": ["Ibrance"], "class": "CDK4/6 Inhibitor", "source": "expert"},
                {"name": "Ribociclib", "aliases": ["Kisqali"], "class": "CDK4/6 Inhibitor", "source": "expert"},
                {"name": "Pembrolizumab", "aliases": ["Keytruda"], "class": "PD-1 Inhibitor", "source": "expert"},
                {"name": "Olaparib", "aliases": ["Lynparza"], "class": "PARP Inhibitor", "source": "expert"},
                {"name": "Fulvestrant", "aliases": ["Faslodex"], "class": "ER Antagonist", "source": "expert"},
                {"name": "Lapatinib", "aliases": ["Tykerb"], "class": "Tyrosine Kinase Inhibitor", "target": "HER2", "source": "expert"},
            ],
        },
        "SideEffect": {
            "description": "Adverse effects of breast cancer treatments",
            "instances": [
                {"name": "Nausea and Vomiting", "source": "expert"},
                {"name": "Hair Loss", "aliases": ["alopecia"], "source": "expert"},
                {"name": "Fatigue", "source": "expert"},
                {"name": "Neutropenia", "aliases": ["low white blood cells"], "source": "expert"},
                {"name": "Neuropathy", "aliases": ["peripheral neuropathy", "numbness", "tingling"], "source": "expert"},
                {"name": "Lymphedema", "aliases": ["arm swelling"], "source": "expert"},
                {"name": "Hot Flashes", "source": "expert"},
                {"name": "Joint Pain", "aliases": ["arthralgia"], "source": "expert"},
                {"name": "Bone Loss", "aliases": ["osteoporosis"], "source": "expert"},
                {"name": "Cardiotoxicity", "aliases": ["heart damage", "heart failure", "heart problems"], "source": "expert"},
                {"name": "Skin Irritation", "aliases": ["radiation dermatitis"], "source": "expert"},
                {"name": "Cognitive Changes", "aliases": ["chemo brain", "brain fog"], "source": "expert"},
                {"name": "Infertility", "source": "expert"},
                {"name": "Blood Clots", "source": "expert"},
                {"name": "Premature Menopause", "source": "expert"},
                {"name": "Lung Inflammation", "source": "expert"},
            ],
        },
        "Recurrence": {
            "description": "Return of cancer after treatment",
            "instances": [
                {"name": "Local Recurrence", "aliases": ["recurrence"], "description": "Returns in same breast or chest wall", "source": "expert"},
                {"name": "Regional Recurrence", "description": "Returns in nearby lymph nodes", "source": "expert"},
                {"name": "Distant Recurrence", "aliases": ["metastatic recurrence"], "description": "Returns in distant organs", "source": "expert"},
            ],
        },
        "Metastasis": {
            "description": "Spread of cancer to distant sites",
            "instances": [
                {"name": "Bone Metastasis", "aliases": ["bone mets"], "description": "Most common site", "source": "expert"},
                {"name": "Lung Metastasis", "aliases": ["lung mets"], "source": "expert"},
                {"name": "Liver Metastasis", "aliases": ["liver mets"], "source": "expert"},
                {"name": "Brain Metastasis", "aliases": ["brain mets"], "source": "expert"},
            ],
        },
        "PreventionAndScreening": {
            "description": "Prevention strategies and screening guidelines",
            "instances": [
                {"name": "Mammography Screening", "description": "Recommended starting age 40-50", "source": "expert"},
                {"name": "Breast Self-Exam", "aliases": ["BSE"], "source": "expert"},
                {"name": "Genetic Counseling", "description": "For high-risk individuals", "source": "expert"},
                {"name": "Risk-Reducing Surgery", "aliases": ["prophylactic mastectomy", "prophylactic oophorectomy"], "source": "expert"},
                {"name": "Chemoprevention", "aliases": ["tamoxifen prevention", "SERM prevention"], "source": "expert"},
            ],
        },
    }

    # Merge extracted concepts into matching classes
    category_to_class = {
        "breast_cancer_subtype": "BreastCancerSubtype",
        "biomarker": "Biomarker",
        "treatment": "Treatment",
        "diagnostic_test": "DiagnosticTest",
        "symptom": "Symptom",
        "risk_factor": "RiskFactor",
        "protective_factor": "ProtectiveFactor",
        "genetic_factor": "GeneticFactor",
        "stage": "StageAndGrade",
        "side_effect": "SideEffect",
    }

    for category, class_name in category_to_class.items():
        if category in extracted_concepts and class_name in classes:
            classes[class_name]["instances"] = _merge_instances(
                classes[class_name]["instances"],
                extracted_concepts[category]
            )

    return classes


def build_relation_types():
    """Define relation types with semantics for misinformation detection."""
    return {
        "treats": {
            "description": "Treatment X is used to treat condition Y",
            "domain": ["Treatment", "Medication"],
            "range": ["BreastCancer", "BreastCancerSubtype", "SideEffect"],
            "strength_levels": ["first_line", "standard", "investigational", "off_label"],
        },
        "prevents": {
            "description": "Factor X reduces likelihood of condition Y developing",
            "domain": ["ProtectiveFactor", "Medication", "PreventionAndScreening"],
            "range": ["BreastCancer", "Recurrence"],
            "strength_levels": ["proven", "probable", "possible"],
        },
        "increases_risk_of": {
            "description": "Factor X increases probability of condition Y",
            "domain": ["RiskFactor", "GeneticFactor", "Medication"],
            "range": ["BreastCancer", "BreastCancerSubtype", "Recurrence"],
            "strength_levels": ["strong", "moderate", "weak"],
        },
        "decreases_risk_of": {
            "description": "Factor X reduces probability of condition Y",
            "domain": ["ProtectiveFactor", "Medication", "Treatment"],
            "range": ["BreastCancer", "Recurrence", "Metastasis"],
            "strength_levels": ["strong", "moderate", "weak"],
        },
        "detects": {
            "description": "Test X can identify or diagnose condition Y",
            "domain": ["DiagnosticTest"],
            "range": ["BreastCancer", "BreastCancerSubtype", "Biomarker", "Metastasis"],
        },
        "has_biomarker": {
            "description": "Cancer subtype X is characterized by biomarker Y status",
            "domain": ["BreastCancerSubtype"],
            "range": ["Biomarker"],
        },
        "has_side_effect": {
            "description": "Treatment X can cause adverse effect Y",
            "domain": ["Treatment", "Medication"],
            "range": ["SideEffect"],
            "strength_levels": ["common", "uncommon", "rare"],
        },
        "contraindicated_for": {
            "description": "Treatment X should NOT be used for condition Y",
            "domain": ["Treatment", "Medication"],
            "range": ["BreastCancerSubtype", "SideEffect"],
        },
        "has_symptom": {
            "description": "Disease X presents with symptom Y",
            "domain": ["BreastCancer", "BreastCancerSubtype"],
            "range": ["Symptom"],
        },
        "has_stage": {
            "description": "Cancer X has staging classification Y",
            "domain": ["BreastCancer"],
            "range": ["StageAndGrade"],
        },
        "associated_with": {
            "description": "Entity X is clinically associated with entity Y",
            "domain": ["Biomarker", "GeneticFactor", "Symptom"],
            "range": ["BreastCancerSubtype", "Treatment", "Metastasis"],
        },
        "causes": {
            "description": "Factor X directly causes outcome Y",
            "domain": ["GeneticFactor", "Treatment", "Medication"],
            "range": ["BreastCancer", "SideEffect", "Recurrence"],
            "note": "Use sparingly; most cancer relationships are probabilistic",
        },
        "reduces_recurrence_risk": {
            "description": "Intervention X lowers chance of cancer returning",
            "domain": ["Treatment", "Medication"],
            "range": ["Recurrence"],
            "strength_levels": ["strong", "moderate", "weak"],
        },
        "improves_survival": {
            "description": "Intervention X is associated with longer survival",
            "domain": ["Treatment", "Medication"],
            "range": ["BreastCancer", "BreastCancerSubtype"],
            "strength_levels": ["proven", "probable", "possible"],
        },
    }


def build_modifiers():
    """Define modifiers for nuanced claim representation."""
    return {
        "negation": {
            "description": "Negates a relation",
            "values": [True, False],
            "default": False,
        },
        "certainty": {
            "description": "Epistemic certainty of the claim",
            "values": ["definite", "probable", "possible", "uncertain", "false"],
            "default": "definite",
        },
        "temporality": {
            "description": "When the relation holds",
            "values": ["always", "during_treatment", "post_treatment", "pre_treatment", "conditional"],
            "default": "always",
        },
        "strength": {
            "description": "How strong the association is",
            "values": ["strong", "moderate", "weak", "negligible"],
            "default": "moderate",
        },
        "population_scope": {
            "description": "Which patient population the claim applies to",
            "values": ["all_patients", "subtype_specific", "stage_specific", "age_specific", "genotype_specific"],
            "default": "all_patients",
        },
        "source_quality": {
            "description": "Evidence quality",
            "values": ["clinical_guideline", "meta_analysis", "randomized_trial", "observational_study", "expert_opinion", "anecdotal"],
            "default": "clinical_guideline",
        },
    }


def build_expert_relations():
    """Expert-curated factual relations with clinical sources."""
    return [
        # Treatment-subtype (expert-curated with guideline sources)
        {"subject": "Trastuzumab", "relation": "treats", "object": "HER2-Positive Breast Cancer", "strength": "strong", "certainty": "definite", "source": "NCCN Guidelines", "data_grounded": False},
        {"subject": "Pertuzumab", "relation": "treats", "object": "HER2-Positive Breast Cancer", "strength": "strong", "certainty": "definite", "source": "NCCN Guidelines", "data_grounded": False},
        {"subject": "Endocrine Therapy", "relation": "treats", "object": "Hormone Receptor-Positive Breast Cancer", "strength": "strong", "certainty": "definite", "source": "NCCN Guidelines", "data_grounded": False},
        {"subject": "Tamoxifen", "relation": "treats", "object": "Hormone Receptor-Positive Breast Cancer", "strength": "strong", "certainty": "definite", "source": "NCCN Guidelines", "data_grounded": False},
        {"subject": "Chemotherapy", "relation": "treats", "object": "Triple-Negative Breast Cancer", "strength": "strong", "certainty": "definite", "source": "NCCN Guidelines", "data_grounded": False},
        {"subject": "Chemotherapy", "relation": "treats", "object": "HER2-Positive Breast Cancer", "strength": "strong", "certainty": "definite", "source": "NCCN Guidelines", "note": "Typically combined with HER2-targeted therapy", "data_grounded": False},
        {"subject": "Pembrolizumab", "relation": "treats", "object": "Triple-Negative Breast Cancer", "strength": "strong", "certainty": "definite", "source": "KEYNOTE-522", "note": "With chemotherapy for early-stage TNBC", "data_grounded": False},
        {"subject": "Olaparib", "relation": "treats", "object": "Breast Cancer", "strength": "strong", "certainty": "definite", "source": "OlympiA Trial", "note": "BRCA-mutated, HER2-negative early BC", "data_grounded": False},
        {"subject": "Palbociclib", "relation": "treats", "object": "Hormone Receptor-Positive Breast Cancer", "strength": "strong", "certainty": "definite", "source": "NCCN Guidelines", "note": "With endocrine therapy for advanced HR+/HER2-", "data_grounded": False},
        {"subject": "Radiation Therapy", "relation": "treats", "object": "Breast Cancer", "strength": "strong", "certainty": "definite", "source": "NCCN Guidelines", "data_grounded": False},

        # Contraindications (critical for misinformation detection)
        {"subject": "Endocrine Therapy", "relation": "contraindicated_for", "object": "Triple-Negative Breast Cancer", "certainty": "definite", "source": "NCCN Guidelines", "note": "TNBC lacks hormone receptors", "data_grounded": False},
        {"subject": "Trastuzumab", "relation": "contraindicated_for", "object": "Triple-Negative Breast Cancer", "certainty": "definite", "source": "NCCN Guidelines", "note": "TNBC is HER2-negative by definition", "data_grounded": False},
        {"subject": "Tamoxifen", "relation": "contraindicated_for", "object": "Triple-Negative Breast Cancer", "certainty": "definite", "source": "NCCN Guidelines", "data_grounded": False},

        # Biomarker-subtype
        {"subject": "HER2-Positive Breast Cancer", "relation": "has_biomarker", "object": "HER2/neu", "certainty": "definite", "data_grounded": False},
        {"subject": "Hormone Receptor-Positive Breast Cancer", "relation": "has_biomarker", "object": "Estrogen Receptor", "certainty": "definite", "data_grounded": False},
        {"subject": "Hormone Receptor-Positive Breast Cancer", "relation": "has_biomarker", "object": "Progesterone Receptor", "certainty": "definite", "data_grounded": False},
        {"subject": "Triple-Negative Breast Cancer", "relation": "has_biomarker", "object": "Estrogen Receptor", "certainty": "definite", "negated": True, "note": "ER-negative", "data_grounded": False},
        {"subject": "Triple-Negative Breast Cancer", "relation": "has_biomarker", "object": "Progesterone Receptor", "certainty": "definite", "negated": True, "note": "PR-negative", "data_grounded": False},
        {"subject": "Triple-Negative Breast Cancer", "relation": "has_biomarker", "object": "HER2/neu", "certainty": "definite", "negated": True, "note": "HER2-negative", "data_grounded": False},

        # Risk factors
        {"subject": "BRCA1", "relation": "increases_risk_of", "object": "Breast Cancer", "strength": "strong", "certainty": "definite", "data_grounded": False},
        {"subject": "BRCA2", "relation": "increases_risk_of", "object": "Breast Cancer", "strength": "strong", "certainty": "definite", "data_grounded": False},
        {"subject": "Family History", "relation": "increases_risk_of", "object": "Breast Cancer", "strength": "strong", "certainty": "definite", "data_grounded": False},
        {"subject": "Age", "relation": "increases_risk_of", "object": "Breast Cancer", "strength": "strong", "certainty": "definite", "data_grounded": False},
        {"subject": "Obesity", "relation": "increases_risk_of", "object": "Breast Cancer", "strength": "moderate", "certainty": "probable", "note": "Especially post-menopausal", "data_grounded": False},
        {"subject": "Alcohol Consumption", "relation": "increases_risk_of", "object": "Breast Cancer", "strength": "moderate", "certainty": "definite", "data_grounded": False},
        {"subject": "Hormone Replacement Therapy", "relation": "increases_risk_of", "object": "Breast Cancer", "strength": "moderate", "certainty": "definite", "data_grounded": False},
        {"subject": "Dense Breast Tissue", "relation": "increases_risk_of", "object": "Breast Cancer", "strength": "moderate", "certainty": "definite", "data_grounded": False},

        # Protective factors
        {"subject": "Physical Exercise", "relation": "decreases_risk_of", "object": "Breast Cancer", "strength": "moderate", "certainty": "probable", "data_grounded": False},
        {"subject": "Breastfeeding", "relation": "decreases_risk_of", "object": "Breast Cancer", "strength": "moderate", "certainty": "probable", "data_grounded": False},
        {"subject": "Prophylactic Mastectomy", "relation": "decreases_risk_of", "object": "Breast Cancer", "strength": "strong", "certainty": "definite", "note": "Reduces risk by ~90%", "data_grounded": False},
        {"subject": "Tamoxifen", "relation": "prevents", "object": "Breast Cancer", "strength": "strong", "certainty": "definite", "note": "Reduces risk of ER+ BC by ~50%", "data_grounded": False},

        # Side effects
        {"subject": "Chemotherapy", "relation": "has_side_effect", "object": "Hair Loss", "strength": "common", "certainty": "definite", "data_grounded": False},
        {"subject": "Chemotherapy", "relation": "has_side_effect", "object": "Nausea and Vomiting", "strength": "common", "certainty": "definite", "data_grounded": False},
        {"subject": "Chemotherapy", "relation": "has_side_effect", "object": "Neutropenia", "strength": "common", "certainty": "definite", "data_grounded": False},
        {"subject": "Chemotherapy", "relation": "has_side_effect", "object": "Neuropathy", "strength": "common", "certainty": "definite", "data_grounded": False},
        {"subject": "Chemotherapy", "relation": "has_side_effect", "object": "Cognitive Changes", "strength": "moderate", "certainty": "probable", "data_grounded": False},
        {"subject": "Trastuzumab", "relation": "has_side_effect", "object": "Cardiotoxicity", "strength": "uncommon", "certainty": "definite", "data_grounded": False},
        {"subject": "Tamoxifen", "relation": "has_side_effect", "object": "Hot Flashes", "strength": "common", "certainty": "definite", "data_grounded": False},
        {"subject": "Radiation Therapy", "relation": "has_side_effect", "object": "Skin Irritation", "strength": "common", "certainty": "definite", "data_grounded": False},
        {"subject": "Radiation Therapy", "relation": "has_side_effect", "object": "Lymphedema", "strength": "uncommon", "certainty": "definite", "data_grounded": False},

        # Recurrence and survival
        {"subject": "Adjuvant Therapy", "relation": "reduces_recurrence_risk", "object": "Recurrence", "strength": "strong", "certainty": "definite", "data_grounded": False},
        {"subject": "Trastuzumab", "relation": "improves_survival", "object": "HER2-Positive Breast Cancer", "strength": "strong", "certainty": "definite", "source": "HERA trial", "data_grounded": False},
        {"subject": "Endocrine Therapy", "relation": "reduces_recurrence_risk", "object": "Recurrence", "strength": "strong", "certainty": "definite", "note": "For HR+ BC", "data_grounded": False},

        # Detection
        {"subject": "Mammogram", "relation": "detects", "object": "Breast Cancer", "certainty": "definite", "data_grounded": False},
        {"subject": "Ultrasound", "relation": "detects", "object": "Breast Cancer", "certainty": "definite", "note": "Complementary to mammography", "data_grounded": False},
        {"subject": "MRI", "relation": "detects", "object": "Breast Cancer", "certainty": "definite", "note": "High sensitivity; for high-risk women", "data_grounded": False},
        {"subject": "Biopsy", "relation": "detects", "object": "Breast Cancer", "certainty": "definite", "note": "Definitive diagnostic test", "data_grounded": False},
        {"subject": "Immunohistochemistry", "relation": "detects", "object": "Estrogen Receptor", "certainty": "definite", "data_grounded": False},
        {"subject": "Immunohistochemistry", "relation": "detects", "object": "Progesterone Receptor", "certainty": "definite", "data_grounded": False},
        {"subject": "Immunohistochemistry", "relation": "detects", "object": "HER2/neu", "certainty": "definite", "data_grounded": False},
        {"subject": "FISH Test", "relation": "detects", "object": "HER2/neu", "certainty": "definite", "data_grounded": False},
        {"subject": "Genetic Testing", "relation": "detects", "object": "BRCA1", "certainty": "definite", "data_grounded": False},
        {"subject": "Genetic Testing", "relation": "detects", "object": "BRCA2", "certainty": "definite", "data_grounded": False},

        # Metastasis associations
        {"subject": "Bone Pain", "relation": "associated_with", "object": "Bone Metastasis", "certainty": "probable", "data_grounded": False},
        {"subject": "HER2-Positive Breast Cancer", "relation": "associated_with", "object": "Brain Metastasis", "certainty": "probable", "data_grounded": False},
    ]


def merge_relations(expert_relations, extracted_relations):
    """Merge expert and data-extracted relations, deduplicating by triple."""
    seen = set()
    merged = []

    # Expert relations take priority
    for r in expert_relations:
        key = (r["subject"], r["relation"], r["object"])
        seen.add(key)
        merged.append(r)

    # Add data-extracted relations not already covered by expert
    for r in extracted_relations:
        key = (r["subject"], r["relation"], r["object"])
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return merged


def build_common_misinformation():
    """Common breast cancer misinformation claims."""
    return [
        {"claim": "Underwire bras cause breast cancer", "verdict": "false", "explanation": "No scientific evidence links bra type to breast cancer risk", "source": "American Cancer Society"},
        {"claim": "Deodorant/antiperspirant causes breast cancer", "verdict": "false", "explanation": "No conclusive evidence links deodorant use to breast cancer", "source": "National Cancer Institute"},
        {"claim": "If you have no family history, you won't get breast cancer", "verdict": "misleading", "explanation": "Most breast cancers (~85%) occur in women with no family history", "source": "American Cancer Society"},
        {"claim": "A breast cancer diagnosis is a death sentence", "verdict": "misleading", "explanation": "5-year survival for localized BC is ~99%. Even Stage IV has improving outcomes.", "source": "SEER"},
        {"claim": "Chemotherapy won't work for HER2-positive breast cancer", "verdict": "misleading", "explanation": "Chemo IS used for HER2+ BC, combined with HER2-targeted therapy", "source": "NCCN Guidelines"},
        {"claim": "Only women get breast cancer", "verdict": "misleading", "explanation": "Men can get breast cancer (~1% of cases)", "source": "American Cancer Society"},
        {"claim": "Mammograms cause breast cancer due to radiation", "verdict": "misleading", "explanation": "Radiation dose is very low. Benefits of early detection far outweigh risk.", "source": "American College of Radiology"},
        {"claim": "Sugar feeds cancer and you should eliminate it", "verdict": "misleading", "explanation": "All cells use glucose. Eliminating sugar doesn't starve cancer. Excess sugar contributes to obesity (a risk factor).", "source": "Mayo Clinic"},
        {"claim": "Natural remedies can cure breast cancer without conventional treatment", "verdict": "false", "explanation": "No natural remedy has been proven to cure breast cancer", "source": "National Cancer Institute"},
        {"claim": "If your mammogram is normal, you don't have breast cancer", "verdict": "misleading", "explanation": "Mammograms miss about 1 in 8 cancers. Dense tissue can obscure tumors.", "source": "American Cancer Society"},
    ]


def build_bcmo():
    """Construct the complete BCMO v2 ontology."""
    extracted_concepts, extracted_relations = load_extracted_data()
    concept_classes = build_concept_classes(extracted_concepts)
    expert_relations = build_expert_relations()
    all_relations = merge_relations(expert_relations, extracted_relations)

    # Count sources
    expert_count = sum(1 for r in all_relations if not r.get("data_grounded", False))
    data_count = sum(1 for r in all_relations if r.get("data_grounded", False))

    total_instances = sum(len(v.get("instances", [])) for v in concept_classes.values())

    bcmo = {
        "ontology_metadata": {
            "name": "Breast Cancer Misinformation Ontology",
            "acronym": "BCMO",
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "description": (
                "A purpose-built ontology for detecting misinformation in breast cancer "
                "patient conversations. Combines MedQuAD-extracted knowledge with expert-curated "
                "biomedical terminology from NCI Thesaurus, UMLS, and NCCN guidelines."
            ),
            "concept_class_count": len(concept_classes),
            "total_concept_instances": total_instances,
            "relation_type_count": len(build_relation_types()),
            "total_factual_relations": len(all_relations),
            "expert_curated_relations": expert_count,
            "data_grounded_relations": data_count,
            "sources": [
                "MedQuAD (Ben Abacha & Demner-Fushman, 2019) -- 6 subsets, 43 breast cancer documents",
                "NCI Thesaurus",
                "UMLS Metathesaurus",
                "NCCN Clinical Practice Guidelines in Oncology: Breast Cancer",
                "American Cancer Society",
            ],
            "license": "CC BY 4.0 (ontology structure); data sources retain original licenses",
        },
        "concept_classes": concept_classes,
        "relation_types": build_relation_types(),
        "modifiers": build_modifiers(),
        "factual_relations": all_relations,
        "common_misinformation": build_common_misinformation(),
    }

    return bcmo


def main():
    bcmo = build_bcmo()

    output_path = DATA_DIR / "bcmo_v2.json"
    with open(output_path, "w") as f:
        json.dump(bcmo, f, indent=2)

    meta = bcmo["ontology_metadata"]
    print(f"BCMO v{meta['version']} constructed successfully")
    print(f"  Concept classes: {meta['concept_class_count']}")
    print(f"  Total concept instances: {meta['total_concept_instances']}")
    print(f"  Relation types: {meta['relation_type_count']}")
    print(f"  Total factual relations: {meta['total_factual_relations']}")
    print(f"    Expert-curated: {meta['expert_curated_relations']}")
    print(f"    MedQuAD-grounded: {meta['data_grounded_relations']}")
    print(f"  Modifiers: {len(bcmo['modifiers'])}")
    print(f"  Common misinformation: {len(bcmo['common_misinformation'])}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
