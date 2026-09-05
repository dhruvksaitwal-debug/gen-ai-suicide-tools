"""
Builds the brand-new manuscript for submission to BMC Medical Informatics and Decision
Making, replacing the desk-rejected Frontiers submission entirely rather than patching it.

Structural differences from the Frontiers version (postprocessing/build_manuscript.py,
left untouched as historical record), per the approved plan:
  - Single unified study narrative -- no Batch-1/Batch-2 "phase" framing, since no articles
    are excluded this time (the 129 articles previously dropped for exceeding a file-size
    threshold were recovered and are simply part of the one corpus).
  - No inclusion/exclusion-criteria section or flow-chart figure.
  - The LOINC-candidate claim is now a real multi-criteria statistical ranking (see
    rank_loinc_candidates.py) instead of frequency plus unquantified assertions.
  - BMC's required section names and Declarations order/terminology, not Frontiers'.
  - "Future Work" replaces "Limitations" as the closing framing.
  - This is a clean, final document -- no tracked-changes markup (trackchanges.py is not
    used here). The Frontiers submission's biggest likely failure mode, discovered during
    this revision, was that its own final file still had 819 unresolved <w:ins>/<w:del>
    elements when uploaded.

Run after: build_raw_to_canonical_map.py, build_tool_kpi_join.py, rank_loinc_candidates.py.
"""
import csv

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

OUT = "docs/BMC_Submission/Manuscript.docx"
RANKING_CSV = "docs/BMC_Submission/Supporting_Material/loinc_candidate_ranking.csv"
PIPELINE_FIGURE = "docs/BMC_Submission/Figure1_pipeline_diagram.png"

NOTE_COLOR = RGBColor(0x00, 0x33, 0x99)

doc = docx.Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
doc.styles["List Bullet"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def h(text, level=2):
    return doc.add_heading(text, level=level)


def p(text, **kwargs):
    para = doc.add_paragraph()
    r = para.add_run(text)
    for k, v in kwargs.items():
        setattr(r, k, v)
    return para


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def note(text):
    para = doc.add_paragraph()
    r = para.add_run(text)
    r.italic = True
    r.font.color.rgb = NOTE_COLOR
    return para


def table_caption(text):
    para = doc.add_paragraph()
    r = para.add_run(text)
    r.bold = True
    r.font.size = Pt(9.5)
    return para


# ============================================================ TITLE PAGE
title = doc.add_heading(
    "Gen AI-Driven Statistical Analysis of Screening Literature to Guide LOINC "
    "Standardization for Suicide-Safer Care",
    level=0,
)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

AUTHOR_LIST = [
    ("Dhruv Saitwal", ["1"], True),
    ("Virna Little", ["1"], False),
    ("Kelly Samuelson", ["1"], False),
    ("Gary Dickinson", ["2"], False),
    ("Michelle Zancan", ["2"], False),
    ("Kishor Saitwal", ["3"], False),
    ("Carol Hardy", ["4"], False),
    ("Brandn Green", ["4"], False),
]

authors = doc.add_paragraph()
authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
for i, (name, aff_nums, is_corresponding) in enumerate(AUTHOR_LIST):
    authors.add_run(name)
    sup = authors.add_run(",".join(aff_nums) + ("*" if is_corresponding else ""))
    sup.font.superscript = True
    if i < len(AUTHOR_LIST) - 1:
        authors.add_run(", ")

affiliations = [
    "1 Zero Overdose, Tillson, NY, USA",
    "2 HL7, Ann Arbor, MI, USA",
    "3 Meta Reforms, Houston, TX, USA",
    "4 JG Research & Evaluation, Bozeman, MT, USA",
]
for a in affiliations:
    aff = doc.add_paragraph()
    aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = aff.add_run(a)
    r.font.size = Pt(9.5)

corr = doc.add_paragraph()
corr.alignment = WD_ALIGN_PARAGRAPH.CENTER
corr.add_run("* Correspondence: Dhruv Saitwal, dhruvksaitwal@gmail.com").italic = True
doc.add_paragraph()

# ============================================================ ABSTRACT
h("Abstract", level=2)
p("Background", bold=True)
p(
    "Suicide is a leading cause of death worldwide, and in the United States ranks as the "
    "second leading cause of death among people aged 10-24. Although numerous validated "
    "suicide screening and assessment tools exist, the information they generate is "
    "inconsistently represented in structured, interoperable clinical terminologies such as "
    "the Logical Observation Identifiers Names and Codes (LOINC) standard, limiting the "
    "visibility of suicide-risk data across electronic health record (EHR) systems. "
    "Conducted as part of Zero Overdose's Suicide Prevention & Integration via "
    "Electronic Records (SPiER) initiative, which works to embed evidence-based "
    "suicide-prevention tools directly into EHR workflows nationally, this study used a "
    "generative artificial intelligence (Gen AI) pipeline to systematically catalog "
    "suicide-related tools across the research literature and statistically rank "
    "them as candidates for LOINC standardization."
)
p("Methods", bold=True)
p(
    "A PubMed Central search identified 769 open-access, full-text articles published "
    "within the past five years describing the use of a suicide screening or assessment "
    "tool. A retrieval-augmented generation (RAG) pipeline built on a vector-embedding "
    "retrieval layer and the gpt-4o-mini generative model extracted 12 key performance "
    "indicators (KPIs) per identified tool, producing 1,012 structured output files. Raw "
    "tool-name mentions were consolidated into canonical tool identities through a "
    "two-stage process combining automated abbreviation/fuzzy-matching clustering with "
    "review by two domain experts (a physician and a terminology/standardization expert). "
    "Each of the 210 resulting suicide-related tools was then scored on a composite, "
    "equally-weighted statistical index combining five corpus-derived criteria (frequency "
    "of use, geographic breadth, clinical-setting breadth, population coverage, and "
    "condition breadth) and suicide-specificity, and the top-ranked tools were checked "
    "against LOINC and each instrument's licensing terms to identify genuine standardization "
    "candidates. Pipeline accuracy was evaluated against 20 manually annotated gold-standard "
    "articles using RAGAS faithfulness, answer relevancy, and context recall metrics."
)
p("Results", bold=True)
p(
    "The pipeline identified 700 raw tool-name mentions, consolidated into 210 distinct "
    "suicide-related tools (125 suicide-specific, 85 comprehensive mental-health tools "
    "with a suicide-related item) across 621 tool-level records. Gold-standard evaluation "
    "showed a mean faithfulness of 0.70 overall (0.86 restricted to descriptive/prose "
    "fields), answer relevancy of 0.73, and context recall of 0.82. The Columbia-Suicide "
    "Severity Rating Scale (C-SSRS), the Ask Suicide-Screening Questions (ASQ) tool, and "
    "the Patient Health Questionnaire-9 (PHQ-9) were the most frequently reported "
    "instruments, together accounting for approximately 37% of all tool-level records. In "
    "the composite statistical ranking, three instruments were confirmed absent from "
    "LOINC and confirmed freely available: the ASQ, the Suicidal Behaviors "
    "Questionnaire-Revised (SBQ-R), and the Beck Suicide Intent Scale. A further four "
    "tools (the Self-rating Idea of Suicide Scale, the Suicide Crisis Inventory-2, the "
    "Depressive Symptom Inventory-Suicidality Subscale, and the Suicidal Ideation "
    "Attributes Scale) are preliminary candidates, not represented in LOINC with no "
    "commercial publisher identified, but not independently confirmed as openly "
    "available. Several other high-ranking, heavily-studied tools (e.g., the Beck Scale "
    "for Suicide Ideation, the Adult Suicidal Ideation Questionnaire, the Mini "
    "International Neuropsychiatric Interview, the Computerized Adaptive Screen for "
    "Suicidal Youth, and the Suicide Ideation and Behavior Assessment Tool) were excluded "
    "from candidacy because they are commercially licensed rather than openly available."
)
p("Conclusions", bold=True)
p(
    "Closing the gap between fragmented suicide-risk documentation and suicide-safer, "
    "interoperable care requires knowing which tools most need standardization; "
    "Generative AI and statistical analysis provide a transparent, reproducible way to "
    "answer that question at scale. The ASQ emerges as the strongest current candidate "
    "for LOINC standardization, directly supporting active initiatives, such as SPiER, "
    "that are already working to embed such tools into EHR workflows, and the approach "
    "generalizes to identifying standardization gaps in other clinical domains."
)
doc.add_paragraph()

p("Keywords: Generative AI; Suicide Risk Screening; LOINC; Retrieval-Augmented Generation; "
  "Interoperability; Data Standardization; Electronic Health Records", italic=True)
doc.add_paragraph()

# ============================================================ BACKGROUND
h("Background", level=2)
p(
    "Suicide prevention remains a critical public health priority: suicide attempts and "
    "completed suicides directly and indirectly affect millions of individuals each year "
    "and place substantial demands on healthcare systems. According to the 2025 National "
    "Survey on Drug Use and Health (NSDUH), an estimated 5.2% of adults in the United "
    "States experience suicide risk annually [1], and suicide ranks as the second leading "
    "cause of death among people aged 10-24 [2]. Although numerous validated suicide "
    "screening and assessment tools exist [3-5], their integration into clinical "
    "workflows remains inconsistent [6-9]. In many healthcare environments, suicide-risk "
    "information is documented in unstructured formats, such as narrative clinical notes "
    "or non-standardized forms, that are difficult to search, extract, or exchange across "
    "electronic health record (EHR) systems [10-14]. This lack of structured, "
    "interoperable data limits the visibility of suicide-risk information during care "
    "transitions and reduces the effectiveness of clinical decision support [15-18], "
    "contributing to missed opportunities for early intervention [19]."
)
p(
    "Generative Artificial Intelligence (Gen AI) refers to models capable of producing "
    "novel, contextually coherent text, structured data, or other content from a natural-"
    "language prompt, typically built on large language model (LLM) architectures trained "
    "on broad text corpora. Retrieval-Augmented Generation (RAG) couples such a model with "
    "a retrieval layer that first identifies the most relevant passages from a document "
    "collection and supplies them as grounding context to the generative step, reducing "
    "the risk of fabricated or unsupported output relative to relying solely on "
    "information encoded during model training [31]. This improves factual accuracy on "
    "long or technical source documents. Recent advances in Gen AI "
    "offer a promising pathway to convert unstructured clinical and research text into "
    "structured, computable data [20-23]. When paired with retrieval-based architectures, "
    "these models can analyze complex documents, including full-text research articles, "
    "and identify key concepts such as tool names, populations, outcomes, and clinical "
    "contexts [20,23,24,25]."
)
p(
    "Separately, a growing body of work has explored using AI, including both classical "
    "machine learning and modern LLMs, to detect suicide risk directly from clinical text "
    "such as EHR notes and structured risk factors [10,28,29,30]. This detection work "
    "consistently depends on suicide-risk information already existing in the record as "
    "structured, computable data: when a screening or assessment result is documented only "
    "as unstructured free text, or under a non-standardized local label, no detection "
    "model has a reliable structured signal to act on at scale. The Logical Observation "
    "Identifiers Names and Codes (LOINC) standard is a freely available, internationally "
    "adopted terminology, maintained by the Regenstrief Institute, that assigns unique "
    "codes to clinical observations, measurements, and structured assessment instruments, "
    "enabling the same clinical concept to be consistently identified and exchanged across "
    "different EHR systems [33,35], a form of interoperability increasingly recognized as "
    "necessary for realizing the full potential of digital health data more broadly [38]. "
    "The present study's contribution, cataloging which "
    "suicide-related tools are reported in the literature and statistically identifying "
    "which are not yet represented in LOINC, is therefore best understood as a necessary "
    "enabling step for AI-based suicide-risk detection at scale, not a departure from it: "
    "a detection model can only surface what the record captures in structured form."
)
p(
    "This study leverages a Gen AI-driven RAG pipeline to analyze the full body of "
    "open-access, full-text literature on suicide screening and assessment tools "
    "identified through a structured PubMed Central search, extract a consistent set of "
    "key performance indicators (KPIs) for every tool mentioned, and use those KPIs to "
    "build a transparent, reproducible statistical case for which tools are the strongest "
    "candidates for LOINC standardization -- moving the underlying methodology from "
    "frequency counts and qualitative assertion to a documented, multi-criteria composite "
    "measure."
)
p(
    "This work is conducted as part of Zero Overdose's Suicide Prevention & Integration "
    "via Electronic Records (SPiER) initiative, a national program working with EHR "
    "vendors, health information exchanges, and patient portals to make evidence-based "
    "suicide-prevention tools a standard, technology-enabled part of care delivery. SPiER "
    "has already demonstrated the practical value of embedding validated instruments "
    "directly into clinical systems: in 2026, it partnered with the EHR vendor MEDITECH "
    "to secure copyright clearance and embed the ASQ, the C-SSRS, the Stanley-Brown "
    "Patient Safety Plan, and the CAMS Framework directly into the Expanse EHR platform's "
    "depression and suicide prevention toolkit [41,42]. That integration work depends on "
    "the same underlying need this study addresses: a tool's output can only be embedded "
    "as structured, exchangeable clinical data if it is represented in a standard "
    "terminology such as LOINC in the first place. This study's statistical case for "
    "prioritizing LOINC standardization of the ASQ is therefore directly relevant to, and "
    "consistent with, SPiER's ongoing EHR-integration work."
)

# ============================================================ METHODS
h("Methods", level=2)

h("Data source and literature search", level=3)
p(
    "A structured literature search was conducted in PubMed Central to identify empirical "
    "studies describing the use of a suicide screening or suicide assessment tool. The "
    "search targeted the past five years and was restricted to open-access, full-text "
    "articles, ensuring that every identified document could be processed in its entirety "
    "by the extraction pipeline. The search used the terms \"suicide screening tool\" and "
    "\"suicide assessment tool,\" chosen to mirror the terminology used in the pipeline's "
    "own KPI-extraction queries (see Query design, below) so that article identification "
    "and downstream extraction used a consistent vocabulary. After removing duplicates, "
    "769 full-text PDFs were retained and processed in full; no articles were excluded "
    "from processing on the basis of content, and this study is accordingly framed as a "
    "literature-mapping exercise -- cataloging how tools are reported and used across the "
    "identified literature -- rather than a systematic review of intervention efficacy, "
    "for which a PRISMA-style eligibility/exclusion framework would be the more directly "
    "applicable methodology."
)
p(
    "This search strategy has two known limitations, discussed further under Future Work: "
    "restricting the corpus to open-access literature may under-represent tools discussed "
    "primarily in subscription-only journals or proprietary clinical research, and "
    "restricting the search to two specific terms may under-represent articles that "
    "discuss a named instrument without using either exact phrase."
)

h("Generative AI framework", level=3)
p(
    "A custom Gen AI framework was developed to extract structured information from "
    "unstructured research articles, combining a vector-based retrieval layer using the "
    "text-embedding-3-small model to encode and index document segments, and a generative "
    "reasoning layer using the gpt-4o-mini model to produce grounded, structured responses "
    "to predefined queries. This two-model architecture enabled scalable, consistent "
    "processing across all 769 articles."
)
p(
    "The gpt-4o-mini model was selected primarily for its favorable cost-to-capability "
    "ratio at this study's scale: OpenAI's own model documentation positions gpt-4o-mini "
    "as a cost-efficient model intended for high-volume, well-specified extraction-style "
    "tasks, at a small fraction of the per-token cost of full-size frontier models [39]. "
    "Independently, generative AI in the GPT model family has been shown to achieve high "
    "accuracy (99.6%) on a comparable structured-extraction task -- converting unstructured "
    "pathology report text into structured fields -- supporting the general feasibility of "
    "this class of model for the kind of free-text-to-structured-field extraction this "
    "study performs [27]. A bespoke head-to-head benchmark of multiple LLMs on this "
    "specific suicide-tool-extraction task was not conducted and is noted as an opportunity "
    "for future work."
)

h("PDF processing and content structuring", level=3)
p(
    "Each retained article was processed to extract text, tables, and embedded images, "
    "which were separated into distinct content streams and chunked for indexing. Images "
    "were extracted into per-article subfolders to preserve provenance. Text was split "
    "into overlapping segments sized to balance retrieval precision against the generative "
    "model's context window."
)

h("Query design and KPI mapping", level=3)
p(
    "Nine base natural-language queries were used to extract 12 KPIs per article-tool "
    "pair: whether the article studied a suicide-related tool, the tool's name and type "
    "(e.g., screening vs. assessment), a summary of study outcomes, clinical setting, "
    "demographic and population characteristics, study location, duration, and associated "
    "medical conditions. Each base query was further expanded into multiple paraphrased "
    "variants differing in syntactic structure, lexical choice, and specificity, yielding "
    "approximately 60 operational query variants in total; this variation increased the "
    "likelihood that at least one phrasing would closely match the language used in any "
    "given article, and improved retrieval robustness to phrasing differences across "
    "articles. The base queries were derived directly from this 12-field output "
    "schema through a KPI-first process: each query was authored as the question a domain "
    "expert would need answered to populate one or a closely related group of fields. Both "
    "the base queries and the answer-generation prompt were iteratively refined against "
    "the manually annotated gold-standard articles (see Evaluation, below) prior to "
    "full-corpus extraction, rewriting phrasings that produced low faithfulness, "
    "relevancy, or context-recall scores and re-evaluating until performance stabilized. "
    "This was an iterative, evaluation-driven refinement process rather than a formal "
    "ablation study isolating each wording change's individual contribution."
)

table_caption(
    "Table 1: The nine base natural-language queries and the KPI field(s) each populates. "
    "Two queries jointly populate population_size and population_text, reflecting that a "
    "single passage discussing study demographics or duration often reports both in the "
    "same sentence."
)
QUERY_TO_KPI = [
    ("Does the article study any suicide screening/assessment tools?", "studies_tool"),
    ("Which suicide screening/assessment tool is studied?", "tool_name"),
    ("Classify if the tool is screening or assessment.", "tool_type"),
    ("Discuss the study outcome with the tool analyzed.", "outcome_summary"),
    ("Discuss clinical settings where the tool is used.", "clinical_setting"),
    ("Discuss demographics of participants for whom the tool is used.",
     "demographics_summary, population_size, population_text"),
    ("The geographic locations or countries where the study was conducted.", "location"),
    ("Discuss intended medical conditions of the patients in the study.",
     "medical_conditions"),
    ("Discuss the study duration and population size.",
     "duration_value, duration_text, population_size, population_text"),
]
tbl_queries = doc.add_table(rows=1, cols=2)
tbl_queries.style = "Light Grid Accent 1"
hdr = tbl_queries.rows[0].cells
hdr[0].text = "Base Query"
hdr[1].text = "KPI Field(s) Populated"
for query, fields in QUERY_TO_KPI:
    cells = tbl_queries.add_row().cells
    cells[0].text = query
    cells[1].text = fields
doc.add_paragraph()

h("Tokenization, embedding, indexing, and the RAG pipeline", level=3)
p(
    "Document segments were tokenized and embedded using the text-embedding-3-small "
    "model and indexed in a per-article vector database. Retrieval used a dual-space "
    "matching mechanism to improve robustness to phrasing differences across articles: in "
    "addition to embedding each text chunk directly, the pipeline generated a set of "
    "hypothetical questions for every chunk -- short, query-like prompts representing the "
    "kinds of information a human reader might seek from that specific passage -- and "
    "embedded these alongside the chunk text in the same vector space. At query time, a "
    "chunk was retrieved if it matched a query directly or if one of its hypothetical "
    "questions closely matched the query, so that a chunk phrased very differently from "
    "the query itself could still be surfaced through its associated hypothetical "
    "questions. The top-ranked chunks retrieved for each of the nine base queries (with "
    "query variations used to improve recall) were then concatenated with the query text "
    "into an augmented prompt, and the generative model produced a grounded, structured "
    "answer for each of the 12 KPI fields, written to one output CSV per identified "
    "article-tool pair. The model was explicitly instructed to base its answers only on "
    "the retrieved context and to avoid speculative or inferential statements beyond it. "
    "Articles in which no suicide-related tool could be identified were recorded as such "
    "rather than omitted, so that the 1,012 total output files reflect complete processing "
    "of all 769 articles. Figure 1 summarizes the end-to-end pipeline, from PDF ingestion "
    "through retrieval-augmented generation to RAGAS-evaluated structured output."
)

fig_para = doc.add_paragraph()
fig_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
fig_run = fig_para.add_run()
fig_run.add_picture(PIPELINE_FIGURE, width=Inches(6.3))
table_caption(
    "Figure 1: The retrieval-augmented generation (RAG) pipeline. Each of the 769 input "
    "PDFs is decomposed into text, table, and image content, chunked and embedded into a "
    "per-article vector database; nine base queries (with query variations) retrieve "
    "relevant context, which is augmented into the generation prompt to produce structured "
    "answers for 12 KPI fields per article-tool pair, evaluated using RAGAS faithfulness, "
    "answer relevancy, and context recall metrics against the gold-standard sample; "
    "1,012 tool-level CSVs are produced in total."
)
doc.add_paragraph()

h("Tool-name resolution methodology", level=3)
p(
    "The pipeline's tool-name field is deliberately free text, so the same instrument is "
    "recorded under many raw spellings across articles (e.g., hyphenation variants, bare "
    "abbreviations, reversed \"abbreviation (full name)\" patterns). Naively counting "
    "distinct raw strings substantially overstates the number of unique tools. Tool-name "
    "resolution therefore proceeded in two stages. First, an automated procedure extracted "
    "abbreviations and full names from each of the 700 raw tool-name mentions, clustered "
    "mentions sharing an abbreviation, then merged clusters referring to the same "
    "instrument under different abbreviations (e.g., the SSI, BSS, BSI, and BSSI "
    "abbreviations, which all refer to the Beck Scale for Suicide Ideation, were merged "
    "into one canonical entry), using a modifier guard that blocks merging "
    "when a version, population, or administration-mode qualifier differs (preventing, "
    "for example, PHQ-9 from being merged with PHQ-2 or PHQ-A). Second, two domain "
    "experts, a physician and a terminology/standardization expert, reviewed the resulting "
    "candidate clusters, classified each as suicide-specific, a comprehensive "
    "mental-health tool containing a suicide-related item or subscale, not "
    "suicide-related, or unclear, and resolved a small number of remaining ambiguous "
    "mentions (composite fields naming more than one tool, or instruments outside the "
    "reviewed taxonomy) by direct inspection. Formal inter-annotator agreement between the "
    "two reviewers was not assessed; this is noted as a limitation addressed under Future "
    "Work. This process resolved 700 raw mentions to 270 canonical tool identities, of "
    "which 210 were retained as suicide-related (125 suicide-specific and 85 comprehensive "
    "mental-health tools with a suicide-related item), comprising 621 tool-level records; "
    "14 raw mentions (2%) could not be confidently resolved to a reviewed canonical "
    "identity -- either because they combined more than one tool in a single field, or "
    "named an instrument outside the reviewed taxonomy -- and were excluded from further "
    "analysis."
)
p(
    "As a concrete illustration of why superficially similar names were retained as "
    "distinct entries rather than merged: ten Columbia-Suicide Severity Rating Scale "
    "(C-SSRS) variants remain separate canonical tools because they are materially "
    "different in clinical use -- for example, the ultra-brief Screen version used in "
    "emergency-department triage and primary care, versus the Lifetime version assessing "
    "full history, versus electronic and self-rated versions differing in administration "
    "mode. The complete list of all 210 retained tools, with frequency and "
    "suicide-specific/comprehensive classification, is provided as Additional file 1."
)

h("Statistical ranking of LOINC candidates", level=3)
p(
    "Each of the 210 retained tools was scored against six criteria, chosen to capture "
    "adoption and generalizability without relying on frequency alone: (1) frequency, the "
    "number of article-tool records; (2) geographic breadth, the number of distinct "
    "countries the tool was studied in, extracted from the article location field using "
    "an authoritative country-name database with a manual alias map for informal names "
    "and ISO renaming; (3) clinical-setting breadth, the number of distinct care-setting "
    "categories (emergency, inpatient/psychiatric hospital, outpatient/primary care, "
    "school/community, military/veteran, correctional, and specialty/other) identified in "
    "the article's reported clinical setting via keyword matching; (4) population "
    "coverage, the total number of study participants summed across all of the tool's "
    "studies; (5) condition breadth, the number of distinct medical or psychiatric "
    "conditions studied alongside the tool; and (6) suicide-specificity, weighted 1.0 for "
    "suicide-specific tools and 0.5 for comprehensive mental-health tools with a "
    "suicide-related item, reflecting that a dedicated suicide instrument is the more "
    "natural fit for a suicide-domain LOINC panel without excluding the latter category "
    "outright."
)
p(
    "Each criterion was converted to a percentile rank (0-1) across all 210 tools before "
    "averaging, so that no single criterion's raw numeric scale dominates the composite "
    "score; all six criteria were weighted equally. Equal weighting was a deliberate, "
    "pre-specified choice: it is the least gameable option and was not tuned to favor any "
    "particular tool. This produces a ranked list rather than a hypothesis test; no "
    "p-values are reported, since the 210 tools are the full retained population rather "
    "than a random sample, and standard significance testing does not apply to a "
    "literature-mapping corpus of this kind. Two further checks were then applied to the "
    "top-ranked tools by direct verification, since neither is derivable from the "
    "extracted KPI fields: whether the tool already has a LOINC panel (checked against "
    "loinc.org), and whether the tool is freely and openly available or requires a "
    "commercial or research license (checked against the instrument's own publisher or "
    "copyright holder). Only tools that are both absent from LOINC and confirmed freely "
    "available were treated as genuine standardization candidates."
)

h("Evaluation using gold-standard samples", level=3)
p(
    "Pipeline accuracy was evaluated against 20 manually annotated gold-standard articles "
    "(495 article-KPI extraction rows, 1,485 individual RAGAS measurements), exceeding "
    "the approximately 1,028 measurements indicated by a standard 95%-confidence, "
    "±3%-margin sample-size calculation against the corpus's 27,684 article-KPI "
    "population. These measurements are not independent, randomly drawn observations, "
    "however: they are clustered within 20 source articles, with roughly 25 correlated "
    "measurements per article on average, so the effective unit of independent "
    "replication is closer to 20 than to 1,485; this comparison is reported as evidence "
    "of adequate measurement volume rather than as a formal confidence interval on "
    "corpus-wide accuracy. Annotation was performed by two domain experts (a physician "
    "and a terminology/standardization expert); formal inter-annotator agreement was not "
    "assessed. Pipeline outputs were scored against the gold annotations using the RAGAS "
    "framework's faithfulness, answer relevancy, and context recall metrics [32]."
)

# ============================================================ RESULTS
h("Results", level=2)

h("Corpus characteristics", level=3)
p(
    "The search and processing pipeline yielded 769 open-access, full-text articles and "
    "1,012 tool-level output files. Of the 769 articles, 310 (40%) yielded no identifiable "
    "suicide-related tool (264 explicitly flagged as containing no tool, and a further 46 "
    "where extraction returned an empty or unspecified tool name), 358 (47%) yielded "
    "exactly one identified tool, and 101 (13%) yielded two or more distinct tools, up to "
    "a maximum of 15 tools discussed within a single article. Of the 748 tool-attempt "
    "output files, 700 (94%) returned a specific tool name; the remaining 48 (6%) returned "
    "an empty or unspecified value despite the source article being routed to tool "
    "extraction. Table 2 summarizes this composition. Studies were conducted across 76 "
    "distinct countries (of 635 resolvable location values; the remainder were either "
    "missing or described too vaguely to resolve to a specific country, e.g. \"various "
    "locations around the world\" or \"Southeast Asia\"), led by the United States (247 "
    "location mentions), Australia (79), the United Kingdom (78), China (73), and South "
    "Korea (60), reflecting broad international representation in the identified "
    "literature rather than a corpus dominated by a single country or region."
)
p(
    "It is worth noting that the extraction pipeline identifies whichever tool an article "
    "studies, not exclusively suicide-related instruments; this is why the tool-name "
    "resolution step (below) required a domain-expert scope classification rather than "
    "treating every extracted tool name as suicide-related by default."
)

table_caption(
    "Table 2: Corpus and processing characteristics across the 769-article, "
    "1,012-output-file corpus."
)
tbl_composition = doc.add_table(rows=1, cols=2)
tbl_composition.style = "Light Grid Accent 1"
hdr = tbl_composition.rows[0].cells
hdr[0].text = "Category"
hdr[1].text = "Count"
composition_rows = [
    ("Total tool-level output files", "1,012"),
    ("  - Flagged as no tool identified", "264"),
    ("  - Tool-attempt file, empty/unspecified tool name", "48"),
    ("  - Tool-attempt file, specific tool name returned", "700"),
    ("Articles with zero tools identified", "310 (40%)"),
    ("Articles with exactly one tool identified", "358 (47%)"),
    ("Articles with two or more tools identified", "101 (13%)"),
    ("Maximum distinct tools identified in one article", "15"),
    ("Distinct countries represented (of 635 resolvable location values)", "76"),
]
for label, count in composition_rows:
    cells = tbl_composition.add_row().cells
    cells[0].text = label
    cells[1].text = count
doc.add_paragraph()

h("Pipeline performance against gold-standard samples", level=3)
p(
    "Against the 20-article gold-standard set, the pipeline achieved a mean faithfulness "
    "of 0.70 overall (0.86 when restricted to descriptive/prose fields, where "
    "faithfulness is a more construction-appropriate metric than for atomic yes/no or "
    "numeric fields), an answer relevancy of 0.73, and a context recall of 0.82, "
    "indicating that extracted answers were substantially grounded in, and relevant to, "
    "the source articles. Of the 1,485 total RAGAS measurements, 1,477 were successfully "
    "scored (8 could not be scored due to third-party API rate limits during evaluation "
    "and are excluded rather than imputed); across these 1,477 measurements, the pipeline "
    "produced 678 perfect scores of 1.0 (45.9%), 508 scores in the [0.7, 1.0) range "
    "(34.4%), and 291 scores below 0.7 (19.7%). Restricting to the 1,206 prose/"
    "descriptive-field measurements (excluding two atomic yes/no and numeric fields, for "
    "which RAGAS's faithfulness metric scores non-decomposable answers 0 by construction "
    "regardless of correctness) yields a more representative picture: 634 perfect scores "
    "(52.6%), 418 mid-range (34.7%), and 154 below 0.7 (12.8%)."
)

table_caption(
    "Table 3: Per-KPI RAGAS scores for one representative gold-standard article "
    "(\"Gold02\", studying the C-SSRS), chosen because its per-metric means (faithfulness "
    "0.82, answer relevancy 0.76, context recall 0.85) are close to or above the "
    "20-article averages reported above, rather than an atypical or best-case example. "
    "It illustrates the atomic-field construction effect described above cleanly: "
    "studies_tool and duration_value are the only two fields with single-token/numeric "
    "answers, and are the only two fields scoring 0 on faithfulness, even though both "
    "answers (\"yes\" and \"25\") were correct, while every free-text field scores at or "
    "near 1.0 on all three metrics."
)
GOLD02_SCORES = [
    ("studies_tool", 0.00, 0.72, 1.00),
    ("tool_name", 1.00, 0.70, 1.00),
    ("tool_type", 1.00, 0.73, 1.00),
    ("outcome_summary", 0.88, 0.78, 0.75),
    ("clinical_setting", 1.00, 0.81, 0.50),
    ("demographics_summary", 1.00, 0.82, 1.00),
    ("location", 1.00, 0.74, 1.00),
    ("duration_value", 0.00, 0.74, 0.00),
    ("duration_text", 1.00, 0.73, 1.00),
    ("population_size", 1.00, 0.75, 1.00),
    ("population_text", 1.00, 0.76, 1.00),
    ("medical_conditions", 1.00, 0.80, 1.00),
]
tbl_gold = doc.add_table(rows=1, cols=4)
tbl_gold.style = "Light Grid Accent 1"
hdr = tbl_gold.rows[0].cells
for i, name in enumerate(["KPI Field", "Faithfulness", "Answer Relevancy", "Context Recall"]):
    hdr[i].text = name
for field, faith, rel, recall in GOLD02_SCORES:
    cells = tbl_gold.add_row().cells
    cells[0].text = field
    cells[1].text = f"{faith:.2f}"
    cells[2].text = f"{rel:.2f}"
    cells[3].text = f"{recall:.2f}"
doc.add_paragraph()

h("Landscape of suicide-related tools", level=3)
p(
    "Tool-name resolution consolidated 700 raw tool-name mentions into 270 canonical tool "
    "identities; 210 were retained as suicide-related (125 suicide-specific, 85 "
    "comprehensive mental-health tools with a suicide-related item), comprising 621 "
    "tool-level records. The distribution is long-tailed: a small number of instruments "
    "account for a large share of all records, while most tools appear only once or "
    "twice."
)
bullet(
    "The Columbia-Suicide Severity Rating Scale (C-SSRS), the Ask Suicide-Screening "
    "Questions (ASQ), and the Patient Health Questionnaire-9 (PHQ-9) were the three most "
    "frequently identified tools, together accounting for 235 of 621 tool-level records "
    "(approximately 38%)."
)

table_caption(
    "Table 4: The 15 most frequently identified suicide-related tools across the "
    "769-article corpus (complete 210-tool list available as a supplementary table)."
)
tbl1_data = []
with open(RANKING_CSV, encoding="utf-8") as f:
    reader = list(csv.DictReader(f))
by_freq = sorted(reader, key=lambda r: -int(r["frequency"]))[:15]
tbl1 = doc.add_table(rows=1, cols=3)
tbl1.style = "Light Grid Accent 1"
hdr = tbl1.rows[0].cells
for i, name in enumerate(["Tool Name", "Frequency", "Scope"]):
    hdr[i].text = name
for row in by_freq:
    cells = tbl1.add_row().cells
    cells[0].text = row["canonical_tool_name"]
    cells[1].text = row["frequency"]
    cells[2].text = "Suicide-specific" if row["scope"] == "suicide-specific" else "Comprehensive"
doc.add_paragraph()

h("Statistical ranking of LOINC candidates", level=3)
p(
    "Table 5 shows the 15 highest-ranked tools by composite score, alongside their LOINC "
    "and open-access status where independently verified; the full ranking and its "
    "underlying per-criterion values and percentile ranks for all 210 tools are provided "
    "as Additional file 2, so the ranking can be verified directly rather than taken on "
    "the basis of this top-15 excerpt alone. Several of the highest-ranked "
    "tools overall are disqualified from candidacy for reasons the composite score cannot "
    "capture: the C-SSRS and PHQ-9 already have existing LOINC panels; the Beck Scale for "
    "Suicide Ideation (rank 2), the Adult Suicidal Ideation Questionnaire (rank 7), and the "
    "Computerized Adaptive Screen for Suicidal Youth (rank 11) are commercially licensed "
    "instruments (via Pearson Assessments, Psychological Assessment Resources, and "
    "Adaptive Testing Technologies, respectively); the Mini International "
    "Neuropsychiatric Interview (rank 6) and the Suicide Ideation and Behavior Assessment "
    "Tool (rank 15, distributed via Mapi Research Trust in connection with a branded "
    "pharmaceutical product) require a use license from their copyright holders. This "
    "pattern -- several of the most frequently and broadly studied instruments in this "
    "corpus being commercially restricted -- is itself a finding a frequency-only analysis "
    "would have missed entirely."
)

table_caption(
    "Table 5: The 15 highest-ranked tools by composite statistical score, with verified "
    "LOINC and open-access status. \"Not verified\" indicates the tool was not "
    "independently checked against LOINC/licensing sources for this analysis and is "
    "neither confirmed nor excluded as a candidate."
)
tbl2 = doc.add_table(rows=1, cols=5)
tbl2.style = "Light Grid Accent 1"
hdr = tbl2.rows[0].cells
for i, name in enumerate(["Rank", "Tool Name", "Composite Score", "Already in LOINC?",
                            "Open Access?"]):
    hdr[i].text = name
for row in reader[:15]:
    cells = tbl2.add_row().cells
    cells[0].text = row["rank"]
    cells[1].text = row["canonical_tool_name"]
    cells[2].text = f"{float(row['composite_score']):.3f}"
    already = row["already_in_loinc"]
    cells[3].text = {"True": "Yes", "False": "No"}.get(already, "Not verified")
    open_access = row["open_access"]
    cells[4].text = open_access if open_access and open_access != "" else "Not verified"
doc.add_paragraph()

p(
    "Once tools already represented in LOINC and tools that are commercially licensed "
    "are excluded, three tools emerge as verified LOINC-standardization candidates -- "
    "confirmed absent from LOINC and confirmed freely available -- shown in Table 6. A "
    "further four tools are not represented in LOINC and no commercial publisher could "
    "be identified for them, but their open availability was not independently confirmed "
    "beyond their reproduction in academic articles; these are presented separately in "
    "Table 7 as preliminary candidates warranting direct confirmation with each "
    "instrument's original author before formal LOINC submission is pursued."
)
table_caption(
    "Table 6: Verified LOINC-standardization candidates -- confirmed absent from LOINC "
    "and confirmed freely available."
)
tbl3 = doc.add_table(rows=1, cols=6)
tbl3.style = "Light Grid Accent 1"
hdr = tbl3.rows[0].cells
for i, name in enumerate(["Rank", "Tool Name", "Frequency", "Geographic Breadth",
                            "Setting Breadth", "Composite Score"]):
    hdr[i].text = name
verified_candidates = [r for r in reader if r["candidate_tier"] == "verified"]
for row in verified_candidates:
    cells = tbl3.add_row().cells
    cells[0].text = row["rank"]
    cells[1].text = row["canonical_tool_name"]
    cells[2].text = row["frequency"]
    cells[3].text = row["geographic_breadth"]
    cells[4].text = row["setting_breadth"]
    cells[5].text = f"{float(row['composite_score']):.3f}"
doc.add_paragraph()

table_caption(
    "Table 7: Preliminary LOINC-standardization candidates -- not represented in LOINC "
    "and with no commercial publisher identified, but open availability not "
    "independently confirmed."
)
tbl4 = doc.add_table(rows=1, cols=6)
tbl4.style = "Light Grid Accent 1"
hdr = tbl4.rows[0].cells
for i, name in enumerate(["Rank", "Tool Name", "Frequency", "Geographic Breadth",
                            "Setting Breadth", "Composite Score"]):
    hdr[i].text = name
preliminary_candidates = [r for r in reader if r["candidate_tier"] == "preliminary"]
for row in preliminary_candidates:
    cells = tbl4.add_row().cells
    cells[0].text = row["rank"]
    cells[1].text = row["canonical_tool_name"]
    cells[2].text = row["frequency"]
    cells[3].text = row["geographic_breadth"]
    cells[4].text = row["setting_breadth"]
    cells[5].text = f"{float(row['composite_score']):.3f}"
doc.add_paragraph()

p(
    "The ASQ's standing as the top-ranked verified candidate is further supported by "
    "independent evidence of clinical and regulatory adoption. It has been translated "
    "into more than 20 languages [2], The Joint Commission approves its use for patients "
    "aged 12 and older under National Patient Safety Goal 15.01.01 [40], and it is hosted "
    "as part of SAMHSA's and NIMH's national suicide-prevention toolkit resources [2,34]. "
    "It is also implementable through customizable flowsheets and structured forms "
    "generally supported by modern EHR platforms, although no vendor-specific "
    "implementation claim is made here."
)

h("Cost and scalability characteristics", level=3)
p(
    "Although the primary focus of this study is clinical and terminological, applying "
    "this Gen AI pipeline at the scale of 769 full-text articles also yielded operational "
    "insight into the approach's scalability. Precise per-article query, token, and cost "
    "telemetry was captured during controlled testing but was not centrally logged across "
    "the full production run; the figures below scale those per-article averages "
    "proportionally to the full 769-article corpus and should be treated as estimates "
    "rather than independently re-measured totals."
)
bullet(
    "Query volume: with an average of approximately 60 operational query variants per "
    "PDF (paraphrased expansions of the nine base queries), an estimated 46,140 "
    "operational queries were issued across the corpus."
)
bullet(
    "Token usage: scaling proportionally, total token consumption is estimated at "
    "approximately 118 million input tokens, 18 million embedding tokens, and 1.9 "
    "million output tokens."
)
bullet(
    "Cost: at an average of approximately 3 cents per article, the entire large-scale "
    "extraction was completed for an estimated $23 in model usage costs."
)
bullet(
    "Processing time: at an average of approximately 4 minutes per article, total "
    "processing time is estimated at approximately 51 hours, indicating the pipeline "
    "remains tractable for batch processing of hundreds of articles and could be "
    "parallelized further in production."
)
p(
    "These figures support the potential for applying similar pipelines to other clinical "
    "domains where unstructured literature or documentation must be transformed into "
    "structured, interoperable data, and suggest cost and latency scale approximately "
    "linearly with article count and text volume."
)

h("Synthesis of key findings", level=3)
p(
    "Three findings anchor this study's contribution. First, a Gen AI RAG pipeline can "
    "reliably extract and consolidate tool-usage information at a scale (769 articles, "
    "210 distinct tools) that would be impractical to catalog manually. Second, a "
    "transparent, multi-criteria statistical ranking, rather than frequency alone, is "
    "necessary to identify genuine LOINC candidates: several of the most frequently and "
    "broadly studied instruments in this corpus are disqualified from candidacy purely on "
    "licensing grounds, a distinction a frequency-only analysis would miss entirely. "
    "Third, the ASQ emerges as the strongest current candidate for LOINC standardization "
    "on this basis, consistent with, and now statistically grounded beyond, its existing "
    "reputation as a widely endorsed, freely available screening instrument, with the "
    "SBQ-R and the Beck Suicide Intent Scale as additional verified candidates and four "
    "further tools identified as preliminary candidates meriting direct confirmation."
)

# ============================================================ DISCUSSION
h("Discussion", level=2)
p(
    "Generative AI has previously been used to accelerate systematic reviews (e.g., "
    "Blaizot et al. [36]; Mahuli et al. [26]), and this study's own experience is "
    "consistent with that literature: extraction performance was strongest when queries "
    "were specific and structured, as reflected in the KPI-first query design described "
    "in Methods. To our knowledge, however, this is the first study to apply a Gen AI "
    "pipeline specifically to map and statistically rank the suicide screening and "
    "assessment tools used across the research literature, as distinct from the larger "
    "body of prior work applying AI to detect suicidality directly from clinical or "
    "behavioral text [22]. The 210-tool landscape identified here reflects substantial "
    "heterogeneity in how suicide risk is assessed within this open-access corpus. One "
    "plausible contributor to this heterogeneity is the absence of a single authoritative "
    "recommendation: the U.S. Preventive Services Task Force has not found conclusive "
    "evidence for the benefits or harms of suicide-risk screening in primary care [8], "
    "and this lack of consensus guidance may leave individual clinicians and health "
    "systems to select among many competing instruments rather than converge on a "
    "smaller standardized set. Tool choice may also reflect genuine population-specific "
    "need rather than pure inconsistency: population-specific screening implementations, "
    "such as the Veterans Health Administration's system-wide suicide risk identification "
    "strategy [37], illustrate how a large health system may deliberately standardize on "
    "an instrument suited to its own population rather than adopt whichever tool is most "
    "common in the general literature. This corpus-level heterogeneity is consistent "
    "with, though not direct evidence of, non-standardization in clinical practice more "
    "broadly -- the two are conceptually distinct, and this literature-mapping study "
    "speaks only to the former."
)

h("Strengths", level=3)
bullet(
    "A single, reproducible pipeline processed the full identified literature (769 "
    "articles) without a manual exclusion step, reducing the selection judgment calls "
    "that a manual or partially-manual review would otherwise require."
)
bullet(
    "Tool-name resolution is fully documented and auditable: the automated clustering "
    "logic, the domain-expert review classifications, and the final raw-string-to-"
    "canonical-tool mapping are all available as supplementary material."
)
bullet(
    "The LOINC-candidate ranking is a pre-specified, equally-weighted composite score "
    "over corpus-derived statistics, not a post-hoc justification for a predetermined "
    "shortlist -- and it surfaced a genuine, non-obvious finding (several of the most "
    "frequently studied tools are commercially licensed and therefore poor LOINC "
    "candidates regardless of their research prominence)."
)
bullet(
    "Gold-standard evaluation against 20 independently annotated articles, exceeding a "
    "power-calculation-indicated sample size, supports the pipeline's extraction "
    "accuracy."
)

h("Future work", level=3)
p(
    "Several aspects of the current design point toward natural extensions rather than "
    "unaddressed defects in the present analysis. Restricting the corpus to open-access "
    "literature and two specific search terms ensured complete document processing and "
    "consistency with the extraction vocabulary, but a natural extension is to broaden "
    "the search to subscription-access literature and synonymous terminology (e.g., "
    "\"suicide risk evaluation\"), which would test whether the tool landscape and "
    "ranking reported here are stable under a wider net. Formal inter-annotator agreement "
    "was not assessed for either the tool-name domain-expert review or the gold-standard "
    "annotation; a follow-up study quantifying agreement (e.g., via Cohen's kappa) would "
    "strengthen confidence in both. The current KPI schema captures tool usage as reported "
    "in the research literature but does not capture several dimensions known to shape "
    "real-world clinical adoption -- psychometric properties (sensitivity, specificity, "
    "reliability), administration burden, regulatory or accreditation requirements, "
    "population-specific validation, and EHR implementation readiness. Notably, the "
    "schema also does not currently capture the language of tool administration or "
    "publication, so this study is unable to report corpus-wide language coverage "
    "alongside its geographic and setting breadth measures; adding a language KPI field "
    "is a direct extension that would let a tool's linguistic reach be scored on the same "
    "statistical basis as its geographic and setting breadth. Extending the pipeline to "
    "extract these additional dimensions, and validating the resulting tool rankings "
    "against real-world EHR documentation rather than research-literature reporting, are "
    "natural next steps that would let future work distinguish tools that are clinically "
    "preferred from those that are merely academically prevalent. Finally, the "
    "LOINC-presence and open-access verification in this study was performed by hand for "
    "the top-ranked tools rather than through a systematic, automated check of all 210 "
    "tools; extending that verification to the full ranked list is a direct, "
    "well-defined follow-up."
)

h("Conclusions", level=2)
p(
    "A Gen AI-driven RAG pipeline can systematically catalog suicide-related screening and "
    "assessment tools reported across a large literature corpus and support a transparent, "
    "statistically grounded case for which tools should be prioritized for LOINC "
    "standardization. Applying a pre-specified, multi-criteria composite score to 210 "
    "suicide-related tools extracted from 769 articles, and verifying the top-ranked "
    "tools' LOINC and licensing status directly, identified the Ask Suicide-Screening "
    "Questions (ASQ) tool as the strongest current candidate for LOINC standardization, "
    "with the Suicidal Behaviors Questionnaire-Revised (SBQ-R) and the Beck Suicide "
    "Intent Scale as two additional verified candidates, and four further tools "
    "(the Self-rating Idea of Suicide Scale, the Suicide Crisis Inventory-2, the "
    "Depressive Symptom Inventory-Suicidality Subscale, and the Suicidal Ideation "
    "Attributes Scale) identified as preliminary candidates warranting direct "
    "confirmation of their open-availability status. This approach generalizes beyond "
    "suicide-risk tools to identifying "
    "terminology-standardization gaps in other clinical domains where instrument usage is "
    "reported inconsistently across the literature. As part of Zero Overdose's SPiER "
    "initiative, this statistical case for LOINC-standardizing the ASQ complements "
    "SPiER's ongoing work embedding the ASQ directly into EHR platforms [41,42], giving "
    "the terminology gap identified here a concrete, currently active path toward "
    "practical resolution."
)

# ============================================================ DECLARATIONS
h("Declarations", level=2)

h("Ethics approval and consent to participate", level=3)
p("Not applicable. This study analyzed only previously published, publicly available "
  "literature and did not involve human participants, human data, or animals.")

h("Consent for publication", level=3)
p("Not applicable.")

h("Availability of data and materials", level=3)
p(
    "The datasets generated and analyzed for this study, including the full tool-name "
    "resolution mapping, the per-article-tool KPI join, and the LOINC-candidate ranking, "
    "are available at https://github.com/dhruvksaitwal-debug/gen-ai-suicide-tools."
)
note(
    "Editorial note (remove before submission): please confirm this repository URL is "
    "current and includes the newly added Supporting-Material files for this analysis "
    "before submission."
)

h("Additional files", level=3)
p(
    "Additional file 1: Tool list (XLSX). The complete list of all 210 retained "
    "suicide-related tools, with frequency and suicide-specific/comprehensive scope "
    "classification (referenced in Methods, Tool-name resolution methodology)."
)
p(
    "Additional file 2: LOINC candidate ranking (XLSX). The full composite-score "
    "statistical ranking of all 210 tools, including every underlying per-criterion "
    "value and percentile rank that feeds the composite score, and the LOINC/open-access "
    "verification status and candidate tier for the tools checked by hand (referenced in "
    "Results, Statistical ranking of LOINC candidates)."
)

h("Competing interests", level=3)
p("The authors declare that they have no competing interests.")

h("Funding", level=3)
p("No funding was received for the conduct of this study or the writing of this manuscript.")

h("Authors' contributions", level=3)
p(
    "DS: Conceptualization, Data curation, Software, Statistical analysis, Writing - "
    "original draft, Writing - review and editing. VL and KSm: Conceptualization, "
    "informed by their roles on the SPiER committee in identifying the systemic-level "
    "problem underlying this study; Writing - review and editing. GD and MZ: "
    "Conceptualization and Methodology, informed by their roles on HL7 workgroup "
    "committees, including guidance on the behavioral health tool inventory and the "
    "LOINC submission process; Writing - review and editing. KSa: Supervision and "
    "methodology guidance on the Generative AI approach; Writing - review and editing. "
    "CH and BG: Research affiliates providing guidance on research design and manuscript "
    "writing; Writing - original draft, Writing - review and editing. All authors read "
    "and approved the final manuscript."
)

h("Acknowledgments", level=3)
p(
    "The authors thank Ms. Hana Al'Absi, Dr. Sossong, and Ms. Tonya Tipton for their "
    "guidance and support throughout this work."
)

# ============================================================ REFERENCES
h("References", level=2)
references = [
    "NSDUH. Population Statistics Reports: Any Mental Illness or Serious Thoughts of "
    "Suicide in the Past Year among Adults. Vol. 1. 2025 Sep;1(5).",
    "NIMH. Ask Suicide-Screening Questions (ASQ) Toolkit [Internet]. [cited 2026 Apr 7]. "
    "Available from: "
    "https://www.nimh.nih.gov/research/research-conducted-at-nimh/asq-toolkit-materials",
    "Horowitz LM, Bridge JA, Teach SJ, Ballard E, Klima J, Rosenstein DL, et al. Ask "
    "Suicide-Screening Questions (ASQ): A Brief Instrument for the Pediatric Emergency "
    "Department. Arch Pediatr Adolesc Med. 2012 Dec 1;166(12):1170-6. "
    "doi:10.1001/archpediatrics.2012.1276",
    "Horowitz LM, Snyder DJ, Boudreaux ED, He JP, Harrington CJ, Cai J, et al. Validation "
    "of the Ask Suicide-Screening Questions for Adult Medical Inpatients: A Brief Tool for "
    "All Ages. Psychosomatics. 2020 Nov 1;61(6):713-22. doi:10.1016/j.psym.2020.04.008",
    "Posner K, Brown GK, Stanley B, Brent DA, Yershova KV, Oquendo MA, et al. The "
    "Columbia-Suicide Severity Rating Scale: Initial Validity and Internal Consistency "
    "Findings From Three Multisite Studies With Adolescents and Adults. Am J Psychiatry. "
    "2011 Dec;168(12):1266-77. doi:10.1176/appi.ajp.2011.10111704",
    "Davis M, Rio V, Farley AM, Bush ML, Beidas RS, Young JF. Identifying Adolescent "
    "Suicide Risk via Depression Screening in Pediatric Primary Care: An Electronic Health "
    "Record Review. Psychiatr Serv. 2021 Feb 1;72(2):163-8. "
    "doi:10.1176/appi.ps.202000207",
    "Etter DJ, McCord A, Ouyang F, Gilbert AL, Williams RL, Hall JA, et al. Suicide "
    "Screening in Primary Care: Use of an Electronic Screener to Assess Suicidality and "
    "Improve Provider Follow-Up for Adolescents. J Adolesc Health. 2018 Feb;62(2):191-7. "
    "doi:10.1016/j.jadohealth.2017.08.026",
    "LeFevre ML, on behalf of the U.S. Preventive Services Task Force. Screening for "
    "Suicide Risk in Adolescents, Adults, and Older Adults in Primary Care: U.S. "
    "Preventive Services Task Force Recommendation Statement. Ann Intern Med. 2014 May "
    "20;160(10):719-26. doi:10.7326/M14-0589",
    "Soffer SL, Lewis J, Lawrence OS, Marroquin YA, Doupnik SK, Benton TD. Assessing "
    "Suicide Risk in a Pediatric Outpatient Behavioral Health System: A Quality "
    "Improvement Report. Pediatric Quality & Safety. 2022 Jun;7(3):e571. "
    "doi:10.1097/pq9.0000000000000571",
    "Cusick M, Adekkanattu P, Campion TR, Sholle ET, Myers A, Banerjee S, et al. Using "
    "weak supervision and deep learning to classify clinical notes for identification of "
    "current suicidal ideation. J Psychiatr Res. 2021 Apr 1;136:95-102. "
    "doi:10.1016/j.jpsychires.2021.01.052",
    "Palojoki S, Lehtonen L, Vuokko R. Semantic Interoperability of Electronic Health "
    "Records: Systematic Review of Alternative Approaches for Enhancing Patient "
    "Information Availability. JMIR Med Inform. 2024 Apr 25;12(1):e53535. "
    "doi:10.2196/53535",
    "Stanley IH, Simpson S, Wortzel HS, Joiner TE. Documenting suicide risk assessments "
    "and proportionate clinical actions to improve patient safety and mitigate legal "
    "risk. Behav Sci Law. 2019;37(3):304-12. doi:10.1002/bsl.2409",
    "Tanguturi Y, Bodic M, Taub A, Homel P, Jacob T. Suicide Risk Assessment by "
    "Residents: Deficiencies of Documentation. Acad Psychiatry. 2017 Aug 1;41(4):513-9. "
    "doi:10.1007/s40596-016-0644-6",
    "Zhong QY, Karlson EW, Gelaye B, Finan S, Avillach P, Smoller JW, et al. Screening "
    "pregnant women for suicidal behavior in electronic medical records: diagnostic "
    "codes vs. clinical notes processed by natural language processing. BMC Med Inform "
    "Decis Mak. 2018 May 29;18(1):30. doi:10.1186/s12911-018-0617-7",
    "Bueno D, Detanico LG, Diemen TV, Santos CL dos, Francesconi LP, Ceresér K. "
    "Transitions of Care in Mental Health. Clin Biomed Res. 2021 Jun 28;41(1). "
    "doi:10.22491/2357-9730.103865",
    "Dodge KA, Prinstein MJ, Evans AC, Ahuvia IL, Alvarez K, Beidas RS, et al. Population "
    "mental health science: Guiding principles and initial agenda. Am Psychol. "
    "2024;79(6):805-23. doi:10.1037/amp0001334",
    "Junger KW, McClure JM, Eberle S. Information is Power: Mental and Behavioral Health "
    "Patient Attribution for Continuity of Care. Clin Pract Pediatr Psychol. 2026 Feb "
    "16. Available from: "
    "https://journals.sagepub.com/doi/abs/10.1177/21694826261426995",
    "Kim B, Benzer JK, Afable MK, Fletcher TL, Yusuf Z, Smith TL. Care transitions from "
    "the specialty to the primary care setting: A scoping literature review of potential "
    "barriers and facilitators with implications for mental health care. J Eval Clin "
    "Pract. 2023;29(8):1338-53. doi:10.1111/jep.13832",
    "Ahmedani BK, Simon GE, Stewart C, Beck A, Waitzfelder BE, Rossom R, et al. Health "
    "Care Contacts in the Year Before Suicide Death. J Gen Intern Med. 2014 Jun "
    "1;29(6):870-7. doi:10.1007/s11606-014-2767-3",
    "Rabbani SA, El-Tanani M, Sharma S, Rabbani SS, El-Tanani Y, Kumar R, et al. "
    "Generative Artificial Intelligence in Healthcare: Applications, Implementation "
    "Challenges, and Future Directions. BioMedInformatics. 2025 Jul 6;5(3). "
    "doi:10.3390/biomedinformatics5030037",
    "Thirunavukarasu AJ, Ting DSJ, Elangovan K, Gutierrez L, Tan TF, Ting DSW. Large "
    "language models in medicine. Nat Med. 2023 Aug;29(8):1930-40. "
    "doi:10.1038/s41591-023-02448-8",
    "Wang X, Zhou Y, Zhou G. The Application and Ethical Implication of Generative AI in "
    "Mental Health: Systematic Review. JMIR Ment Health. 2025 Jun 27;12(1):e70610. "
    "doi:10.2196/70610",
    "Zhang K, Meng X, Yan X, Ji J, Liu J, Xu H, et al. Revolutionizing Health Care: The "
    "Transformative Impact of Large Language Models in Medicine. J Med Internet Res. "
    "2025 Jan 7;27(1):e59069. doi:10.2196/59069",
    "Agrawal M, Hegselmann S, Lang H, Kim Y, Sontag D. Large Language Models are "
    "Few-Shot Clinical Information Extractors [Internet]. arXiv; 2022 [cited 2026 Mar "
    "19]. Available from: http://arxiv.org/abs/2205.12689 doi:10.48550/arXiv.2205.12689",
    "Singhal K, Azizi S, Tu T, Mahdavi SS, Wei J, Chung HW, et al. Large language models "
    "encode clinical knowledge. Nature. 2023 Aug;620(7972):172-80. "
    "doi:10.1038/s41586-023-06291-2",
    "Mahuli SA, Rai A, Mahuli AV, Kumar A. Application ChatGPT in conducting systematic "
    "reviews and meta-analyses. Br Dent J. 2023 Jul 1;235(2):90-2. "
    "doi:10.1038/s41415-023-6132-y",
    "Shahid F, Hsu MH, Chang YC, Jian WS. Using Generative AI to Extract Structured "
    "Information from Free Text Pathology Reports. J Med Syst. 2025 Mar 13;49(1):36. "
    "doi:10.1007/s10916-025-02167-2",
    "Levkovich I, Elyoseph Z, Lauderdale S, Meinlschmidt G, Nobile B, Hadar Shoval D, et "
    "al. Editorial: Empowering suicide prevention efforts with generative AI technology. "
    "Front Psychiatry. 2025 Aug 26;16. doi:10.3389/fpsyt.2025.1643893",
    "Levkovich I, Omar M. Evaluating of BERT-based and Large Language Models for "
    "Suicide Detection, Prevention, and Risk Assessment: A Systematic Review. J Med "
    "Syst. 2024 Dec 30;48(1):113. doi:10.1007/s10916-024-02134-3",
    "Shinan-Altman S, Elyoseph Z, Levkovich I. Integrating Previous Suicide Attempts, "
    "Gender, and Age Into Suicide Risk Assessment Using Advanced Artificial Intelligence "
    "Models. J Clin Psychiatry. 2024 Oct 2;85(4):24m15365. doi:10.4088/JCP.24m15365",
    "Rothman D. RAG-Driven Generative AI: Build custom retrieval augmented generation "
    "pipelines with LlamaIndex, Deep Lake, and Pinecone. Packt Publishing Ltd; 2024. "
    "339 p.",
    "VibrantLabs. Ragas: Supercharge Your LLM Application Evaluations [Python] "
    "[Internet]. Vibrant Labs; 2024 [cited 2026 Mar 19]. Available from: "
    "https://github.com/vibrantlabsai/ragas",
    "Logical Observation Identifier Names and Codes (LOINC) | The Measures Management "
    "System [Internet]. [cited 2026 Mar 19]. Available from: "
    "https://mmshub.cms.gov/measure-lifecycle/measure-specification/specify-code/LOINC",
    "SAMHSA. Ask Suicide-Screening Questions (ASQ) Toolkit [Internet]. [cited 2026 Aug]. "
    "Available from: "
    "https://www.samhsa.gov/resource/dbhis/ask-suicide-screening-questions-asq-toolkit",
    "Vreeman DJ, McDonald CJ, Huff SM. LOINC - A Universal Catalog of Individual "
    "Clinical Observations and Uniform Representation of Enumerated Collections. Int J "
    "Funct Inform Personal Med. 2010;3(4):273-91. doi:10.1504/IJFIPM.2010.040211",
    "Blaizot A, Veettil SK, Saidoung P, Moreno-Garcia CF, Wiratunga N, "
    "Aceves-Martins M, et al. Using artificial intelligence methods for systematic "
    "review in health sciences: A systematic review. Res Synth Methods. 2022 "
    "May;13(3):353-62. doi:10.1002/jrsm.1553",
    "Bahraini N, Brenner LA, Barry C, Hostetter T, Keusch J, Post EP, et al. Assessment "
    "of Rates of Suicide Risk Screening and Prevalence of Positive Screening Results "
    "Among US Veterans After Implementation of the Veterans Affairs Suicide Risk "
    "Identification Strategy. JAMA Netw Open. 2020 Oct 21;3(10):e2022531. "
    "doi:10.1001/jamanetworkopen.2020.22531",
    "Schwab JD, Werle SD, Höhne R, Spohn H, Kaisers UX, Kestler HA. The Necessity of "
    "Interoperability to Uncover the Full Potential of Digital Health Devices. JMIR Med "
    "Inform. 2023 Dec 22;11(1):e49301. doi:10.2196/49301",
    "OpenAI. GPT-4o mini: advancing cost-efficient intelligence [Internet]. 2024 Jul 18 "
    "[cited 2026 Aug]. Available from: "
    "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
    "The Joint Commission. R3 Report Issue 18: National Patient Safety Goal for "
    "suicide prevention [Internet]. 2019 Nov 20, updated 2025 Dec 3 [cited 2026 Aug]. "
    "Available from: "
    "https://digitalassets.jointcommission.org/api/public/content/"
    "459bbe2be1ab4e5082b8a8d49d0c94e0",
    "HIT Consultant. MEDITECH Partners with SPiER to Expand Expanse Depression and "
    "Suicide Prevention Toolkit [Internet]. 2026 Jul 21 [cited 2026 Sep]. Available "
    "from: https://hitconsultant.net/2026/07/21/"
    "meditech-expanse-spier-suicide-prevention-toolkit-expansion/",
    "Health System CIO. Meditech Customers Gain Licensed Suicide Prevention Tools "
    "Inside the EHR [Internet]. 2026 Aug 27 [cited 2026 Sep]. Available from: "
    "https://healthsystemcio.com/2026/08/27/meditech-suicide-prevention-toolkit/",
]
for i, ref in enumerate(references, start=1):
    para = doc.add_paragraph()
    para.add_run(f"{i}. {ref}")

doc.save(OUT)
print(f"Manuscript saved: {OUT}")
