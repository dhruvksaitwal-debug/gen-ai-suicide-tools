"""
Rebuilds the full manuscript as a Word document with genuine tracked changes (w:ins/w:del),
starting from the text of the submitted version and applying every revision agreed on with
the corresponding author across this review cycle.
"""
import sys
sys.path.insert(0, "postprocessing")
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from trackchanges import add_normal, add_ins, add_del, replace_paragraph, mixed_paragraph

OUT = "docs/manuscript_revision/Manuscript_TrackedChanges.docx"

doc = docx.Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)


def h(text, level=1):
    return doc.add_heading(text, level=level)


def p(text=""):
    return doc.add_paragraph(text)


def note(text):
    para = doc.add_paragraph()
    r = para.add_run("[EDITOR NOTE: " + text + "]")
    r.font.color.rgb = RGBColor(0x00, 0x55, 0xAA)
    r.italic = True
    return para


# ============================================================= TITLE =====
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_del(title, "Improving Suicide-Safe Care through Gen AI-Driven Recommendations for Tool "
               "Standardization in LOINC", bold=True)
add_ins(title, "Toward Suicide-Safe Care: Gen AI-Driven Mapping of Suicide Screening and "
               "Assessment Tools for LOINC Standardization", bold=True)

p("Dhruv Saitwal¹*, Carol Hardy², Virna Little¹, Gary Dickinson³, Brandn Green²")
p("¹Zero Overdose, Tillson, NY, USA")
p("²JG Research & Evaluation, Bozeman, MT, USA")
p("³HL7, Ann Arbor, MI, USA")
p("* Correspondence: Dhruv Saitwal, dhruvksaitwal@gmail.com")
p("Keywords: Generative AI, Suicide Risk Screening, LOINC, Retrieval-Augmented Generation "
  "(RAG), Interoperability, Data Standardization, Electronic Health Record (EHR), Clinical "
  "Workflow")

# ============================================================= ABSTRACT ===
h("Abstract", level=2)

para = doc.add_paragraph()
add_normal(para, "Introduction: ", bold=True)
add_normal(para,
    "Suicide is a global health concern. In the United States, suicide ranks as the second "
    "leading cause of death among young people ages 10-24. Although validated suicide "
    "screening and assessment tools exist, their integration into clinical workflows remains "
    "inconsistent, and suicide-risk information is frequently captured in formats that may "
    "limit interoperability across electronic health record systems. The authors investigated "
    "whether generative artificial intelligence"
)
add_ins(para, " (specifically, large language models)")
add_normal(para, " can be used as a scalable mechanism for extracting and analyzing "
                  "information from research literature.")

para = doc.add_paragraph()
add_normal(para, "Methods: ", bold=True)
add_normal(para, "A PubMed Central search identified ")
add_del(para, "639")
add_ins(para, "769")
add_normal(para, " open-access, full-text articles"); add_normal(para, " published within the "
    "past five years that evaluated suicide screening or assessment tools")
add_ins(para, " (an initial 639-article core corpus plus 129 articles originally excluded for "
    "exceeding a file-size processing threshold, subsequently recovered via PDF compression or "
    "replaced with equivalent substitute articles - see §2.1)")
add_normal(para, ". A retrieval-augmented generation pipeline extracted 12 key performance "
    "indicators per identified tool, including tool type, clinical setting, population "
    "characteristics, and associated medical conditions. Tools appearing in three or more "
    "studies were cross-referenced against the Logical Observation Identifiers Names and "
    "Codes (LOINC) terminology to identify gaps in current coverage.")

para = doc.add_paragraph()
add_normal(para, "Results: ", bold=True)
add_normal(para, "The pipeline generated ")
add_del(para, "841 tool-level files, corresponding to 239 unique suicide-related tools")
add_ins(para, "1,012 tool-level files. Following a documented two-stage tool-name resolution "
    "process - automated normalization followed by review by two domain experts (a physician "
    "and a terminology/standardization expert) - these corresponded to 216 unique "
    "suicide-related tools (130 suicide-specific instruments and 86 comprehensive "
    "mental-health tools containing a suicide-related item or subscale)")
add_normal(para, ". Pipeline accuracy was evaluated against ")
add_del(para, "four")
add_ins(para, "20")
add_normal(para, " manually annotated gold-sample articles, with faithfulness, answer "
    "relevancy, and context recall scores indicating ")
add_del(para, "strong")
add_ins(para, "generally strong, though not uniform,")
add_normal(para, " performance by the pipeline")
add_ins(para, " (mean faithfulness 0.70 overall, 0.86 restricted to descriptive/prose fields; "
    "answer relevancy 0.73; context recall 0.82)")
add_normal(para, ". Five tools were identified as ")
add_del(para, "strong candidates")
add_ins(para, "candidates warranting further evaluation")
add_normal(para, " for LOINC encoding based on frequency of use, open accessibility, format, "
    "and absence from current terminology coverage.")

para = doc.add_paragraph()
add_normal(para, "Discussion: ", bold=True)
add_del(para, "The 239 tools identified reflect a substantial lack of standardization in "
    "suicide risk assessment.")
add_ins(para, "The heterogeneity of the 216 tools identified in this open-access literature "
    "sample reflects substantial variation in suicide risk assessment instrumentation within "
    "the corpus studied; whether this heterogeneity extends to clinical practice more broadly "
    "is a separate question this literature-mapping study does not directly address.")
add_normal(para, " Several widely used instruments are not currently represented in standard "
    "clinical terminology, limiting the ability to capture suicide-risk data in structured, "
    "interoperable formats. Codifying these tools could improve the visibility of "
    "suicide-risk information across care settings and strengthen the foundation for timely, "
    "data-driven intervention.")

doc.add_page_break()

# ============================================================= 1 INTRO ====
h("1 Introduction", level=2)

para = doc.add_paragraph(
    "The prevention of suicide is a critical public health concern, as suicide attempts and "
    "completed suicides directly and indirectly affect millions of individuals each year "
    "while also placing substantial demands on healthcare systems. According to 2025 "
    "National Survey on Drug Use and Health (NSDUH) report, in the United States alone, an "
    "estimated 5.2% of adults experience suicide risk annually, underscoring the need for "
    "reliable and timely identification of individuals at risk (1). In the United States, "
    "suicide ranks as the second leading cause of death among young people ages 10-24 (2). "
    "Even more common than death by suicide are suicide attempts and suicidal thoughts. "
    "Although numerous validated suicide screening and assessment tools exist (3–5), their "
    "integration into clinical workflows remains inconsistent (6–9). In many healthcare "
    "environments, suicide-risk information is documented in unstructured formats such as "
    "narrative clinical notes, scanned Portable Document Format (PDFs), or non-standardized "
    "forms that are difficult to search, extract, or interpret across electronic health "
    "record (EHR) systems (10–14). This lack of structured and interoperable data limits the "
    "visibility of suicide-risk information during care transitions and reduces the "
    "effectiveness of clinical decision support (15–18), ultimately contributing to missed "
    "opportunities for early intervention (19)."
)

para = doc.add_paragraph()
add_normal(para,
    "Recent advances in Generative Artificial Intelligence (Gen AI) offer a promising "
    "pathway to address these long-standing challenges (20–23). "
)
add_ins(para,
    "In this manuscript, “Gen AI” refers specifically to large language models "
    "(LLMs) - the two models used in this study, gpt-4o-mini for text generation and image "
    "interpretation and text-embedding-3-small for embeddings, are named explicitly in §2.2. "
)
add_normal(para,
    "Modern Gen AI systems can process large volumes of unstructured text, extract clinically "
    "relevant information, and convert it into structured formats suitable for analysis and "
    "interoperability (24,25). When paired with retrieval-based architectures, these models "
    "can analyze complex documents, including full-text research articles, and identify key "
    "concepts such as tool names, populations, outcomes, and clinical contexts (20,23,26,27). "
)
add_ins(para,
    "One such retrieval-based architecture, Retrieval-Augmented Generation (RAG), grounds a "
    "language model's response in text retrieved from an external document collection at "
    "query time, rather than relying solely on information encoded during model training - "
    "reducing the risk of fabricated or unsupported output when the model is asked about "
    "specific source documents (31); the RAG pipeline used in this study is described in "
    "full in §2.6. "
)
add_normal(para,
    "Applying Gen AI to the domain of suicide-risk research can enable scalable, systematic "
    "identification of the tools most frequently used in the literature and provides a "
    "mechanism for evaluating their suitability for integration into standardized clinical "
    "terminologies (28–30). "
)
add_del(para,
    "Prior work has shown that untrained, general-purpose LLMs can identify suicide risk in "
    "clinical case vignettes (28–30), demonstrating the applicability of Gen AI to "
    "suicide-related text analysis. "
)
add_ins(para,
    "General-purpose, untrained LLMs have shown promise in suicide-risk-relevant "
    "classification tasks using constructed clinical vignettes (28–30), suggesting the "
    "broader feasibility of applying LLMs to suicide-related text; however, this prior work "
    "evaluated risk detection in synthetic case descriptions rather than structured "
    "information extraction from full-text literature, and should not be read as direct "
    "precedent for the specific extraction task undertaken here. "
)
add_normal(para,
    "This approach has the potential to bridge the gap between research evidence and "
    "clinical implementation by transforming fragmented information into computable data "
    "that can support real-time decision-making. Additionally, this could help support the "
    "advancement of foundational requirements for electronic records, such as inclusion in "
    "Logical Observation Identifiers Names and Codes (LOINC)"
)
add_ins(para,
    " - a freely available, internationally adopted standard terminology, maintained by the "
    "Regenstrief Institute, that assigns unique codes to clinical observations, measurements, "
    "and structured assessment instruments, enabling the same clinical concept to be "
    "consistently identified and exchanged across different electronic health record systems "
    "(33,35)"
)
add_normal(para,
    ". The variation in which screening tools are utilized is present both within the "
    "evidence-base for suicide care and within clinical practice, creating an opportunity "
    "for the application of Gen AI to efficiently review the nature of the evidence-base, to "
    "inform furthering national standards of care."
)

para = doc.add_paragraph()
add_ins(para,
    "A substantial and growing body of machine learning (ML) and LLM research has "
    "investigated the direct detection of suicide risk from clinical and behavioral text: "
    "systematic reviews have catalogued dozens of studies applying both traditional ML "
    "classifiers and, more recently, BERT-based and generative LLM architectures to predict "
    "suicidal ideation, attempts, and behavior from clinical notes, social media text, and "
    "structured EHR variables (22,29,30). This literature demonstrates that AI models can "
    "achieve clinically meaningful discrimination between at-risk and not-at-risk "
    "individuals when given text or structured data as input, and that incorporating "
    "additional clinical context (e.g., prior attempt history, demographic factors) "
    "improves model performance (30). "
)
add_ins(para,
    "This body of work, however, is built on an important precondition that is rarely made "
    "explicit: it requires the outcome of a suicide screening or assessment tool to already "
    "exist in the record as structured, computable data. When a clinician's completion of "
    "the Columbia-Suicide Severity Rating Scale or a similar instrument is documented only "
    "as free text in a narrative note, or under a locally idiosyncratic label with no "
    "standard terminology mapping, a detection or prediction model, whether ML-based, "
    "LLM-based, or rule-based, has no reliable structured signal to act on at scale across "
    "institutions. The present study addresses this precondition directly: rather than "
    "using Gen AI to detect suicide risk in an individual patient's text, as in the prior "
    "work just discussed, we use it to systematically catalog which instruments the field "
    "already relies on and to what extent their outputs are represented in a standard "
    "terminology (LOINC) capable of making that structured signal available. In this sense, "
    "the tool-standardization work presented here is a necessary enabling step for, rather "
    "than a departure from, the broader trajectory of AI-assisted suicide risk detection "
    "described above."
)

para = doc.add_paragraph()
add_normal(para,
    "The present study leverages a Gen AI-driven pipeline to analyze "
)
add_del(para, "639")
add_ins(para, "769")
add_normal(para,
    " open-access, full-text publications related to suicide screening and assessment. Using "
    "a structured set of queries, the system generated "
)
add_del(para, "841")
add_ins(para, "1,012")
add_normal(para,
    " structured Comma-Separated Values (CSV) outputs, with one file produced for each "
    "suicide-related tool identified within an article. Across the dataset, the analysis "
    "identified "
)
add_del(para, "239 unique suicide-screening or assessment tools used in studies spanning 71 "
               "countries")
add_ins(para, "216 unique suicide-screening or assessment tools - 130 suicide-specific "
    "instruments and 86 comprehensive mental-health tools containing a suicide-related item "
    "or subscale, distinguished through a documented two-stage automated-then-expert-reviewed "
    "resolution process (§2.10) - used in studies spanning 76 countries")
add_normal(para,
    ". Each CSV captured 12 key performance indicators (KPIs), including tool type, clinical "
    "setting, demographic characteristics, study location, population size, duration, and "
    "associated medical conditions. These structured outputs enabled downstream analyses of "
    "tool frequency, geographic distribution, and alignment with existing standards. The "
    "study then compared the identified tools against the LOINC terminology to determine "
    "which widely used tools are not currently represented in the standard. Based on "
    "frequency of use, openness, structure, and suitability for coding, the study proposes a "
    "set of "
)
add_del(para, "high-value")
add_ins(para, "candidate")
add_normal(para,
    " suicide screening and assessment tools for potential inclusion in LOINC. By supporting "
    "the development of structured, interoperable representations of suicide-risk "
    "information, this work aims to "
)
add_del(para, "enhance the visibility and clinical utility of suicide-risk data within EHR "
    "systems and strengthen the foundation for timely, data-driven intervention.")
add_ins(para, "contribute toward enhanced visibility of suicide-risk data within EHR systems, "
    "as a proof-of-concept for AI-supported literature mapping and terminology-gap "
    "identification - future work on clinical implementation and validation would be needed "
    "to establish a direct effect on timely, data-driven intervention.")

doc.add_page_break()

# ============================================================ 2 METHODS ===
h("2 Materials and Methods", level=2)

h("2.1 Data Source and Article Selection", level=3)
para = doc.add_paragraph()
add_normal(para,
    "A structured literature search was conducted in PubMed Central to identify empirical "
    "studies that used suicide screening or suicide assessment tools. "
)
add_ins(para,
    "PubMed Central was selected specifically because it is the primary open-access "
    "aggregator guaranteeing full-text availability, which the automated extraction pipeline "
    "requires - a paywalled abstract-only record cannot be processed. "
)
add_normal(para,
    "The search strategy targeted the past five years"
)
add_ins(para,
    " (chosen to capture the contemporary tool-usage landscape rather than instruments that "
    "may have fallen out of active use)"
)
add_normal(para,
    " and was restricted to open‑access, full‑text articles, ensuring that each document "
    "could be processed in its entirety by the extraction pipeline. The search used the "
    "terms “suicide screening tool” and “suicide assessment tool”"
)
add_ins(para,
    " - chosen to mirror the terminology used in the pipeline's own KPI-extraction queries "
    "(Table 1), ensuring consistency between article identification and downstream "
    "extraction"
)
add_normal(para,
    ", and after removing duplicates and applying inclusion criteria, "
)
add_del(para, "639")
add_ins(para, "768 unique")
add_normal(para, " full‑text PDFs "); add_del(para, "were retained")
add_ins(para, "remained, of which 639 were immediately processable and 129 exceeded a 1.5 MB "
    "file-size threshold that prevented automated processing at the time (Figure 1)")
add_normal(para,
    ". This approach ensured broad coverage of contemporary approaches to inclusion criteria "
    "for scoping reviews, while preserving a consistent and accessible corpus for automated "
    "analysis. Figure 1 shows the screening process, detailing the inclusion and exclusion "
    "steps that resulted in the final set of articles."
)

fig1 = doc.add_picture("docs/manuscript_revision/Figure1_flow_chart.png", width=Inches(5.6))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = cap.add_run("Figure 1: Inclusion and Exclusion Flow Chart (revised)")
r.italic = True
r.font.size = Pt(9.5)

para = doc.add_paragraph()
add_ins(para,
    "The 129 articles originally excluded for exceeding the file-size threshold were "
    "subsequently revisited: file size was reduced below the processing threshold using PDF "
    "compression, and articles that could not be recovered through compression were replaced "
    "with equivalent, newly-identified open-access articles meeting the identical inclusion "
    "criteria. All 129 were successfully processed as a second batch (Batch-2). Combined with "
    "one additional article identified and added to the first batch after initial processing "
    "(Batch-1, final n=640), the corpus totals 769 articles and 1,012 tool-level CSVs. This "
    "directly addresses a concern, raised during peer review, that the original file-size "
    "exclusion represented an unaddressed technical limitation rather than a scientific one."
)

para = doc.add_paragraph()
add_ins(para,
    "This study did not follow the PRISMA (Preferred Reporting Items for Systematic "
    "Reviews and Meta-Analyses) checklist, and we clarify this choice explicitly. PRISMA is "
    "designed for systematic reviews that synthesize quantitative outcomes or evaluate "
    "intervention efficacy across included studies, typically incorporating formal "
    "risk-of-bias appraisal of each study's findings. The present study is not a synthesis "
    "of intervention outcomes; it is a literature-mapping exercise that catalogs which "
    "suicide screening/assessment instruments are used across a corpus of articles and "
    "extracts descriptive metadata about each usage, without appraising or synthesizing the "
    "clinical findings of the underlying studies themselves. PRISMA-ScR, the PRISMA "
    "extension for scoping reviews, is the more directly applicable reporting framework for "
    "this kind of mapping exercise, and its core principles - explicit eligibility "
    "criteria, a transparent and reproducible search strategy, and a structured charting "
    "process - are reflected in the search and extraction methodology described in this "
    "section and in §2.4, even though a formal PRISMA-ScR checklist was not completed for "
    "this submission."
)

para = doc.add_paragraph()
add_ins(para,
    "This search strategy has an acknowledged limitation: restricting identification to two "
    "specific search terms may under-represent articles that study a named instrument "
    "without using the words “screening” or “assessment” in the discoverable "
    "text, or that use synonymous terminology (e.g., “suicide risk evaluation”). This "
    "is discussed further in §4.2."
)

h("2.2 Generative AI Framework", level=3)
para = doc.add_paragraph()
add_normal(para, "A custom Generative AI framework")
add_ins(para, " - specifically, a large language model (LLM) framework")
add_normal(para, " was developed to extract structured information from unstructured research "
    "articles. The system combined:")
doc.add_paragraph("a vector‑based retrieval layer using the text‑embedding-3-small model to "
                   "encode and index document segments, and", style="List Bullet")
doc.add_paragraph("a generative reasoning layer using the gpt‑4o‑mini model to produce "
                   "grounded, structured responses to predefined queries.", style="List Bullet")
p("This two‑model architecture enabled scalable processing across hundreds of documents "
  "while maintaining consistency in how information was extracted and mapped to key "
  "performance indicators (KPIs).")

para = doc.add_paragraph()
add_ins(para,
    "Model selection rationale: gpt-4o-mini and text-embedding-3-small were selected "
    "primarily for their favorable cost-performance tradeoff on high-volume structured "
    "extraction tasks, rather than for accessibility or name recognition alone. On general "
    "capability benchmarks, gpt-4o-mini scores 82.0% on MMLU, ahead of comparably-priced "
    "small models such as Gemini 1.5 Flash (77.9%) and Claude 3 Haiku (73.8%) (40). At the "
    "same time, its list pricing ($0.15 per million input tokens, $0.60 per million output "
    "tokens) is more than an order of magnitude below larger frontier models (40). "
    "Independent benchmarking of structured information extraction specifically has found "
    "that larger frontier models (e.g., full GPT-4o) outperform mini-tier models by a "
    "modest margin on extraction accuracy (96.1% vs. 87.9% in one pathology-report "
    "extraction benchmark) at roughly 44 times the per-document cost (41) - a tradeoff "
    "that does not favor the larger model for a study processing hundreds of full-text "
    "articles on a limited budget (§3.8). We did not conduct a bespoke head-to-head "
    "comparison of multiple LLMs on this specific suicide-tool-extraction task; the "
    "model selection instead relies on these published, task-adjacent benchmarks, and a "
    "systematic comparison across models on this exact extraction task is noted as valuable "
    "future work."
)

h("2.3 PDF Processing and Content Structuring", level=3)
p("Each PDF underwent a standardized preprocessing workflow designed to convert "
  "heterogeneous research articles into machine‑readable text suitable for "
  "retrieval‑augmented generation. The workflow included:")
for item in [
    "Text parsing to extract continuous text from PDFs, including multi‑column layouts.",
    "Table flattening, where tabular data were converted into descriptive text to preserve "
    "semantic meaning.",
    "Image interpretation using the gpt‑4o‑mini model to extract textual information "
    "embedded in figures or diagrams.",
    "Segmentation into overlapping text chunks, ensuring that contextual information was "
    "preserved during retrieval.",
    "Embedding and indexing of all chunks in a vector database to support "
    "similarity‑based retrieval.",
]:
    doc.add_paragraph(item, style="List Bullet")
p("This process ensured that the pipeline could handle diverse document formats and "
  "maintain high recall during information extraction.")

h("2.4 Query Design and KPI Mapping", level=3)
p("To ensure consistent extraction of KPIs across heterogeneous research articles, the "
  "system employed a structured query design process. A set of nine base natural‑language "
  "queries was developed to represent the core semantic intents required for KPI extraction, "
  "including tool type, population characteristics, clinical setting, study duration, and "
  "associated medical conditions. Each base query was then expanded into multiple "
  "paraphrased variants, resulting in approximately 60 operational queries. These variations "
  "differed in syntactic structure, lexical choice, and specificity, enabling the system to "
  "accommodate the diverse ways in which relevant information is expressed across scientific "
  "literature.")
p("This query‑variation strategy (see Figure 1) served two purposes. First, it increased the "
  "likelihood that at least one query variant would closely match the linguistic patterns "
  "present in each article. Second, it improved retrieval robustness by reducing sensitivity "
  "to phrasing differences between articles. Each query variant was mapped to a specific "
  "KPI, as summarized in Table 1, ensuring that all extracted responses aligned with the "
  "structured output schema.")

para = doc.add_paragraph()
add_ins(para,
    "Query design strategy: the nine base queries were derived directly from the "
    "12-field structured output schema (Table 1) using a KPI-first design process: for "
    "each target field, we authored the natural-language question a domain expert would "
    "need answered to populate that field from a source article, then grouped closely "
    "related fields (e.g., population size and its narrative description) under a single "
    "query where a human reader would naturally extract both from the same passage. This "
    "produced the nine-query set shown in Table 1, rather than one query per output field, "
    "reducing redundant retrieval passes while preserving one-to-one traceability between "
    "each query and its output KPI(s). Prompt wording for both the base queries and the "
    "answer-generation prompt (§2.6) was iteratively refined against the manually "
    "annotated gold-standard articles (§2.7) prior to large-scale extraction: early query "
    "phrasings that produced low faithfulness, answer-relevancy, or context-recall scores, "
    "or that were ambiguous about which KPI they targeted, were rewritten and re-evaluated "
    "against the same gold articles until performance stabilized. This iterative, "
    "evaluation-driven refinement is the basis for the query set used in the full-scale "
    "run, though we did not conduct a separate, formal ablation study isolating the "
    "contribution of each individual refinement."
)

cap = doc.add_paragraph()
r = cap.add_run("Table 1: Natural language queries used to generate answers through "
                 "Generative AI before mapping those answers to the output KPIs")
r.bold = True
r.font.size = Pt(9.5)

table1_data = [
    ("1", "Does the article study any suicide screening/assessment tools?", '["studies_tool"]'),
    ("2", "Which suicide screening/assessment tool is studied?", '["tool_name"]'),
    ("3", "Classify if the tool is screening or assessment.", '["tool_type"]'),
    ("4", "Discuss the study outcome with the tool analyzed.", '["outcome_summary"]'),
    ("5", "Discuss clinical settings where the tool is used.", '["clinical_setting"]'),
    ("6", "Discuss demographics of participants for whom the tool is used.",
     '["demographics_summary", "population_size", "population_text"]'),
    ("7", "The geographic locations or countries where the study was conducted.",
     '["location"]'),
    ("8", "Discuss intended medical conditions of the patients in the study.",
     '["medical_conditions"]'),
    ("9", "Discuss the study duration and population size.",
     '["duration_value", "duration_text", "population_size", "population_text"]'),
]
tbl1 = doc.add_table(rows=1, cols=3)
tbl1.style = "Light Grid Accent 1"
hdr = tbl1.rows[0].cells
hdr[0].text, hdr[1].text, hdr[2].text = "Query ID", "Natural language query", "Mapped output KPIs"
for qid, q, kpi in table1_data:
    row = tbl1.add_row().cells
    row[0].text, row[1].text, row[2].text = qid, q, kpi
doc.add_paragraph()

h("2.5 Tokenization, Embedding, and Indexing", level=3)
p("The transformation of raw text into machine‑interpretable representations relies on "
  "three foundational processes: tokenization, embedding, and indexing. Tokenization refers "
  "to breaking the text into smaller units, such as words, subwords, or characters, that can "
  "be processed by language models. These tokens serve as the basic analytical units for "
  "downstream computation. Once tokenized, each segment is converted into a "
  "high‑dimensional numerical vector through embedding, a technique that captures semantic "
  "meaning by placing conceptually similar text segments closer together in vector space. "
  "The embedding technique allows the system to see how phrases are related to each other, "
  "even when they are said in different ways. Finally, all embeddings are stored in a vector "
  "index, which functions as a searchable database optimized for similarity‑based "
  "retrieval. During query processing, the system compares the embedding of a query against "
  "this index to efficiently identify the most semantically relevant text chunks. Together, "
  "these steps form the computational backbone that allows the pipeline to navigate "
  "heterogeneous full‑text articles and retrieve context with high fidelity.")

h("2.6 Retrieval-Augmented Generation Pipeline", level=3)
p("The extraction workflow followed a four‑stage Retrieval‑Augmented Generation (RAG) "
  "pipeline designed to maximize factual grounding and minimize hallucination when "
  "generating structured outputs from full‑text articles (31). In addition to standard "
  "retrieval over document chunks, the system incorporated a hypothetical‑question "
  "mechanism that enriched the semantic representation of each chunk and improved retrieval "
  "accuracy.")
p("1. Query encoding: All operational natural‑language queries (approximately 60 variants "
  "derived from 9 base queries) were converted into vector representations using the "
  "text‑embedding-3-small model. These embeddings captured the semantic intent of each "
  "query in the same vector space used to represent document content.")
p("2. Context retrieval with dual‑space matching: During preprocessing, each article was "
  "segmented into overlapping text chunks to preserve local context. For every chunk, the "
  "system generated a set of hypothetical questions, short, query‑like prompts that "
  "reflected the types of information a human reader might seek from that specific passage. "
  "Both the chunk text and its associated hypothetical questions were embedded in the same "
  "vector space. At retrieval time, each query embedding was compared against the "
  "embeddings of all text chunks and the embeddings of all hypothetical questions "
  "associated with those chunks. A chunk was selected as relevant if it showed high "
  "semantic similarity to the query directly or if one of its hypothetical questions "
  "closely matched the query in the embedding space.")
p("3. Query augmentation and response generation: For each query, the top‑ranked unique "
  "chunks retrieved through this dual‑space matching process were concatenated with the "
  "original query text to form an augmented prompt. This prompt was then passed to the "
  "gpt‑4o‑mini model, which generated a structured response conforming to the predefined "
  "KPI schema. The model was explicitly instructed to base its answers only on the context "
  "provided and to avoid any speculative or inferential statements beyond the retrieved "
  "text.")
p("4. KPI mapping and output structuring: The generated responses were parsed and mapped "
  "to the corresponding KPI fields defined in Table 1. For each article, the pipeline "
  "produced one CSV file per identified suicide‑related tool, with each file containing a "
  "complete set of KPIs for that tool. When multiple tools were reported in a single "
  "article, multiple CSV files were generated, each representing a distinct tool‑level "
  "record.")

doc.add_picture("docs/manuscript_revision/Figure2_pipeline_diagram.png", width=Inches(6.3))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = cap.add_run("Figure 2: Retrieval Augmented Generation (RAG) pipeline (with an optional "
                 "evaluation step) that helps extract answers for input natural language "
                 "queries asked of each PDF article, providing the corresponding results in "
                 "CSV format (revised).")
r.italic = True
r.font.size = Pt(9.5)

para = doc.add_paragraph()
add_normal(para, "Optional evaluation step: ")
add_normal(para, "As illustrated in Figure 1, an optional evaluation step could be inserted "
    "after response generation to assess output quality using RAGAS metrics (faithfulness, "
    "answer relevancy, and context recall) (32). This evaluation was employed during system "
    "validation to characterize pipeline performance but was not executed during the "
    "large‑scale extraction phase applied to all ")
add_del(para, "639")
add_ins(para, "769")
add_normal(para, " articles.")

h("2.7 Evaluation Using Gold Samples", level=3)
para = doc.add_paragraph()
add_normal(para, "To assess extraction accuracy, the system was evaluated using ")
add_del(para, "four")
add_ins(para, "20")
add_normal(para, " manually annotated gold‑sample articles")
add_ins(para, " (more than a five-fold increase from an earlier 4-article validation "
    "assessed during initial peer review, expanding coverage from 0.5% to 2.60% of the "
    "corpus)")
add_normal(para, ". Human‑generated answers ")
add_del(para, "served")
add_ins(para, "- produced by a physician and a terminology/standardization expert, "
    "independently of the pipeline's own output - served")
add_normal(para, " as the reference standard and the following three RAGAS metrics were "
    "used:")
for item in ["Faithfulness – alignment of generated answers with source text",
             "Answer relevancy – appropriateness of the generated answer to the query",
             "Context recall – extent to which retrieved context captured relevant "
             "information"]:
    doc.add_paragraph(item, style="List Bullet")

para = doc.add_paragraph()
add_del(para, "Most scores exceeded 0.70 (% of those at 0.7 or more), indicating strong "
    "reliability of the pipeline for large‑scale extraction.")
add_ins(para,
    "This 20-article validation set yielded 495 article–KPI extraction rows (accounting for "
    "articles studying more than one tool, which contribute one row per tool per field) and "
    "1,485 individual RAGAS measurements - exceeding the approximately 1,028 measurements "
    "indicated by a standard 95%-confidence, ±3%-margin sample-size calculation against the "
    "corpus's 27,684 article–KPI population. Mean faithfulness was 0.70 (0.86 restricted to "
    "descriptive/prose fields - two atomic yes/no and numeric fields are excluded from this "
    "restricted figure because RAGAS's faithfulness metric scores non-decomposable answers 0 "
    "by construction, independent of correctness), answer relevancy 0.73, and context recall "
    "0.82 (0.89 prose-only). See §3.3 for the complete results and §4.2 for discussion of the "
    "clustered (non-independent) nature of these measurements."
)

para = doc.add_paragraph()
add_normal(para, "All CSV outputs were aggregated into a master dataset. This dataset "
    "enabled:")
for item in [
    "identification of 216 unique suicide-related tools,",
    "assessment of tool frequency, and",
    "detection of articles using multiple or unspecified tools.",
]:
    doc.add_paragraph(item, style="List Bullet")
p("This aggregated dataset served as the basis for comparison with standardized clinical "
  "terminologies.")

h("2.8 Comparison With LOINC", level=3)
para = doc.add_paragraph()
add_normal(para, "Suicide tools identified in three or more studies were cross‑referenced "
    "with existing entries in the Logical Observation Identifiers Names and Codes (LOINC) "
    "terminology (33). Tools were evaluated based on ")
add_ins(para, "the following explicit, operationalized criteria: ")
for item in [
    "frequency of appearance in literature (minimum of three occurrences),",
    "availability as an open or publicly accessible instrument,",
    "presence of a structured question‑and‑answer format,",
    "absence from current LOINC coverage, and",
    "structural suitability for codification - illustrated concretely through the SAFE-T "
    "case discussion in §3.7, where a tool meeting the frequency and accessibility "
    "criteria was nonetheless excluded because its clinical decision-support structure does "
    "not align with LOINC's data-collection-instrument model.",
]:
    doc.add_paragraph(item, style="List Bullet")
p("This process produced a prioritized list of most frequently used tools, which were then "
  "evaluated to determine whether they were proprietary or open‑source, since open‑source "
  "tools are easier to incorporate into the LOINC terminology because they do not require "
  "payment or approval for utilization. The process produced a shortlist of five tools to "
  "support standardized representation of suicide‑risk information in clinical systems.")

h("2.9 Tool-Name Resolution Methodology", level=3)
para = doc.add_paragraph()
add_ins(para,
    "Raw tool-name extractions were resolved to canonical instruments through a documented "
    "two-stage process. First, an automated normalization pass grouped raw extractions by "
    "shared parenthetical abbreviations - merging spelling and hyphenation variants, and "
    "merging different abbreviations known to denote the same instrument (e.g., SSI, BSS, "
    "BSI, and BSSI, which all refer to the Beck Scale for Suicide/Suicidal Ideation) - while "
    "explicitly keeping distinct any pair differing by a version, edition, population, or "
    "administration-mode qualifier (e.g., PHQ-9 vs. PHQ-2 vs. PHQ-A; BDI vs. BDI-II; MMPI-2 "
    "vs. MMPI-3 vs. MMPI-A). This automated pass reduced 700 raw tool-name mentions to 319 "
    "candidate canonical tools. Second, this candidate list was reviewed in full by two "
    "domain experts (a physician and a terminology/standardization expert), who classified "
    "each candidate as suicide-specific, a comprehensive mental-health tool containing a "
    "suicide-related item or subscale, not actually suicide-related, or unclear/unverifiable. "
    "This yields the final, human-verified total of 216 unique suicide-related tools (130 "
    "suicide-specific, 86 comprehensive) used throughout this manuscript. As a concrete "
    "illustration of why superficially similar names were retained as distinct entries "
    "rather than merged: ten Columbia-Suicide Severity Rating Scale (C-SSRS) variants remain "
    "separate because they are materially different in clinical use - for example, the "
    "ultra-brief 5–6 item Screen version used in emergency-department triage and primary "
    "care, versus the Lifetime version assessing full history, versus electronic and "
    "self-rated versions differing in administration mode. The complete list of all 216 "
    "tools, with frequency and suicide-specific/comprehensive classification, is provided as "
    "Supplementary Table S1."
)

doc.add_page_break()

# ============================================================= 3 RESULTS ==
h("3 Results", level=2)

h("3.1 Corpus Characteristics and Overall Output", level=3)
para = doc.add_paragraph()
add_normal(para, "The structured literature search and filtering process yielded ")
add_del(para, "639")
add_ins(para, "769")
add_normal(para, " open‑access, full‑text research articles published within the past "
    "five years")
add_ins(para, " (640 in Batch-1, including one article added after initial processing, and "
    "129 in Batch-2, comprising articles originally excluded for file size and subsequently "
    "recovered or replaced - see §2.1)")
add_normal(para, ". These articles constituted the full input corpus for the generative AI "
    "extraction pipeline. The pipeline processed each article independently and produced "
    "structured outputs at the level of individual suicide‑related tools.")

para = doc.add_paragraph()
add_normal(para, "Across all articles, the system generated ")
add_del(para, "841")
add_ins(para, "1,012")
add_normal(para, " tool‑level CSV files, reflecting the fact that many studies evaluated "
    "more than one instrument. As mentioned earlier, for each article, the system generated "
    "one CSV file per identified tool resulting in the final aggregated dataset that "
    "included:")
for item_old, item_new in [
    (f"{'639'} articles processed", f"{'769'} articles processed"),
    (f"{'841'} tool‑level structured outputs", f"{'1,012'} tool‑level structured outputs"),
    (f"{'239'} unique suicide‑related tools", f"{'216'} unique suicide‑related tools "
     "(130 suicide-specific, 86 comprehensive)"),
]:
    para = doc.add_paragraph(style="List Bullet")
    add_del(para, item_old)
    add_ins(para, item_new)
para = doc.add_paragraph(style="List Bullet")
add_del(para, "71 countries represented across studies")
add_ins(para, "76 countries represented across studies")
p("This breadth demonstrates the global diversity of suicide‑risk research and the "
  "heterogeneity of tools used across clinical and research settings.")

h("3.2 Article-Level Patterns of Tool Reporting", level=3)
p("The corpus exhibited substantial variation in how suicide‑related tools were "
  "described and whether they were explicitly named. The structured extraction revealed "
  "four primary categories of articles:")

para = doc.add_paragraph(style="List Bullet")
add_normal(para, "Articles with no identifiable suicide‑related tool: ", bold=True)
add_normal(para, "A sizable subset of articles (total of ")
add_del(para, "208")
add_ins(para, "262")
add_normal(para, ") discussed suicide risk conceptually or descriptively but did not "
    "evaluate or apply a specific instrument. These articles were retained for completeness "
    "but did not contribute tool‑level records.")

para = doc.add_paragraph(style="List Bullet")
add_normal(para, "Articles referencing suicide‑related tools without naming them: ", bold=True)
add_normal(para, "Some studies (total of ")
add_del(para, "43")
add_ins(para, "48")
add_normal(para, ") described the use of “a suicide screening tool” or “a "
    "suicide assessment tool” without specifying the tool name. These were categorized "
    "as unspecified tool cases.")

para = doc.add_paragraph(style="List Bullet")
add_normal(para, "Articles studying exactly one identifiable tool: ", bold=True)
add_normal(para, "These articles (total of ")
add_del(para, "304")
add_ins(para, "358")
add_normal(para, ") generated a single CSV output and contributed directly to tool‑level "
    "frequency counts.")

para = doc.add_paragraph(style="List Bullet")
add_normal(para, "Articles studying multiple tools: ", bold=True)
add_normal(para, "When an article evaluated more than one instrument, the pipeline produced "
    "multiple CSV files (one per tool) ensuring that each instrument was represented as a "
    "distinct analytic unit. There were ")
add_del(para, "84")
add_ins(para, "101")
add_normal(para, " such articles.")

p("This classification reflects the heterogeneity of reporting practices.")

h("3.3 Performance of the Retrieval-Augmented Generation Pipeline", level=3)
para = doc.add_paragraph()
add_normal(para, "Accuracy Against Gold‑Sample Articles: ", bold=True)
add_normal(para, "Gen AI extraction accuracy was evaluated using ")
add_del(para, "four")
add_ins(para, "20")
add_normal(para, " manually annotated gold‑sample articles. Human‑generated KPI values "
    "served as the reference standard. As stated earlier, to assess extraction accuracy, "
    "the system was evaluated using three RAGAS metrics, viz., Faithfulness, Answer "
    "relevancy, and Context recall. Table 2 shows one reference gold sample used to "
    "evaluate the performance of the Gen AI solution. It clearly shows how well Gen AI "
    "represents the manual output without going overboard with potential hallucination.")

para = doc.add_paragraph()
add_del(para, "Across all four gold samples, the pipeline demonstrated:")
add_ins(para, "Across all 20 gold samples, the pipeline demonstrated:")
for item in [
    "Moderate-to-high faithfulness, with a large share of KPIs scoring 1.0, indicating that "
    "generated answers were fully grounded in retrieved text for a majority of fields.",
    "Reasonable answer relevancy, with mean scores of 0.73, showing that responses "
    "generally addressed the intended semantic query.",
    "Generally strong context recall (mean 0.82, 0.89 for descriptive/prose fields), "
    "indicating that retrieval frequently surfaced correct supporting passages.",
]:
    doc.add_paragraph(item, style="List Bullet")

para = doc.add_paragraph()
add_del(para,
    "Note that, across all KPIs evaluated in the gold‑sample articles, the pipeline "
    "produced 107 perfect scores of 1.0, 64 scores in the [0.7, 1.0) range, and only 9 "
    "scores below 0.7, reflecting consistently strong grounding and retrieval performance "
    "across the majority of extracted fields."
)
add_ins(para,
    "Across all 1,477 successfully scored measurements from the 20 gold‑sample articles "
    "(1,485 total measurements; 8 could not be scored due to third-party API rate limits "
    "during evaluation and are excluded rather than imputed), the pipeline produced 678 "
    "perfect scores of 1.0 (45.9%), 508 scores in the [0.7, 1.0) range (34.4%), and 291 "
    "scores below 0.7 (19.7%). Restricting to the 1,206 prose/descriptive-field measurements "
    "(excluding two atomic yes/no and numeric fields for which RAGAS's faithfulness metric "
    "scores non-decomposable answers 0 by construction, independent of correctness) yields a "
    "more representative picture: 634 perfect scores (52.6%), 418 mid-range (34.7%), and 154 "
    "below 0.7 (12.8%). We note this expanded, more representative sample yields a somewhat "
    "less uniformly high performance picture than an earlier 4-article validation assessed "
    "during initial peer review (45.9% vs. 59.4% perfect scores across all fields), which we "
    "consider a more credible estimate of true pipeline performance precisely because it is "
    "larger and less susceptible to sampling variability."
)

cap = doc.add_paragraph()
r = cap.add_run(
    "Table 2: Sample performance of the retrieval-augmented generation pipeline, shown for "
    "one representative article (Gold1) from the 20-article gold-standard validation set "
    "(previously evaluated against a 4-article set), showing faithfulness, answer "
    "relevancy, and context-recall scores for each KPI."
)
r.bold = True
r.font.size = Pt(9.5)
table2_data = [
    ("studies_tool", "yes", "yes", "1.00", "0.72", "1.00"),
    ("tool_name", "Ask Suicide-Screening Questions (ASQ)",
     "Ask Suicide-Screening Questions (ASQ)", "1.00", "0.70", "1.00"),
    ("tool_type", "screening", "screening", "1.00", "0.71", "1.00"),
    ("clinical_setting", "Urban pediatric ED in the United States",
     "pediatric emergency departments", "1.00", "0.73", "1.00"),
    ("location", "United States", "USA", "1.00", "0.74", "1.00"),
    ("duration_value", "69 Months", "2115", "1.00", "0.78", "1.00"),
    ("population_size", "15003", "15003", "1.00", "0.76", "1.00"),
    ("medical_conditions", "Behavioral and psychiatric presenting problems",
     "suicidal ideation", "0.70", "0.80", "1.00"),
]
tbl2 = doc.add_table(rows=1, cols=6)
tbl2.style = "Light Grid Accent 1"
hdr = tbl2.rows[0].cells
for i, name in enumerate(["User input", "Gold answer", "GenerativeAI answer",
                            "Faithfulness", "Answer relevancy", "Context recall"]):
    hdr[i].text = name
for row_data in table2_data:
    row = tbl2.add_row().cells
    for i, val in enumerate(row_data):
        row[i].text = val
doc.add_paragraph()

h("3.4 Qualitative Characteristics of Extracted KPIs", level=3)
p("The pipeline performed especially well on structured KPIs such as:")
for item in ["tool name", "tool type", "study location", "population size", "duration text"]:
    doc.add_paragraph(item, style="List Bullet")
p("For more complex free‑text KPIs, such as demographics summaries, outcome summaries, "
  "and medical conditions, the model produced detailed, contextually grounded descriptions "
  "that aligned with human annotations. These findings support the feasibility of applying "
  "retrieval‑augmented generative models to large‑scale extraction of clinically "
  "relevant metadata from heterogeneous scientific literature. As one can see from Table 3, "
  "the Gen AI solution provides a cohesive output across multiple heterogenous studies.")

cap = doc.add_paragraph()
r = cap.add_run("Table 3: Summary of qualitative performance across structured and "
                 "free-text KPIs (illustrative sample rows, unchanged from the submitted "
                 "version).")
r.bold = True
r.font.size = Pt(9.5)
table3_cols = ["Document name", "Study tool", "Tool name", "Tool type", "Clinical setting",
               "Location", "Duration days", "Population size", "Medical conditions"]
table3_data = [
    ("52-Week Open-Label Safety and Tolerability Study of Centanafadine...", "yes",
     "Columbia-Suicide Severity Rating Scale (C-SSRS)", "assessment",
     "clinical trials and research studies", "USA", "365", "662",
     "Attention-Deficit/Hyperactivity Disorder (ADHD)"),
    ("Ability and utility of the Physician Well-Being Index...", "yes",
     "Physician Well-Being Index (PWBI)", "screening",
     "various healthcare environments", "Southern Mainland China", "92", "3071",
     "Chinese physicians"),
]
tbl3 = doc.add_table(rows=1, cols=len(table3_cols))
tbl3.style = "Light Grid Accent 1"
for i, name in enumerate(table3_cols):
    tbl3.rows[0].cells[i].text = name
for row_data in table3_data:
    row = tbl3.add_row().cells
    for i, val in enumerate(row_data):
        row[i].text = val
doc.add_paragraph()

h("3.5 Landscape of Suicide-Related Tools in Research Literature", level=3)
para = doc.add_paragraph()
add_normal(para, "Diversity and Global Distribution: ", bold=True)
add_normal(para, "The aggregated dataset revealed ")
add_del(para, "239 unique suicide‑screening or suicide‑assessment tools used across 71 "
    "countries")
add_ins(para, "216 unique suicide‑screening or suicide‑assessment tools (130 "
    "suicide-specific, 86 comprehensive mental-health tools with a suicide-related item or "
    "subscale) used across 76 countries")
add_normal(para, ". This diversity reflects:")
for item in [
    "wide variation in clinical and research traditions across regions",
    "differences in population‑specific or culturally adapted instruments",
    "the coexistence of both widely used and highly specialized tools",
]:
    doc.add_paragraph(item, style="List Bullet")
p("The structured KPIs enabled systematic comparison across tools, including:")
for item in ["tool type (screening vs. assessment)",
             "clinical settings (e.g., emergency departments, inpatient units, outpatient "
             "clinics)",
             "population characteristics", "study duration and sample size",
             "associated medical or psychiatric conditions"]:
    doc.add_paragraph(item, style="List Bullet")

h("3.6 Frequency of Tool Usage", level=3)
para = doc.add_paragraph()
add_del(para, "While the full distribution spans 239 tools, a subset appeared repeatedly "
    "across the literature.")
add_ins(para, "While the full distribution spans 216 tools, a small number appear "
    "repeatedly across the literature: the Columbia-Suicide Severity Rating Scale alone "
    "accounts for approximately 25% of all tool-level records (156 of 626 records among the "
    "216 tools), and together with the next four most frequent instruments (Ask "
    "Suicide-Screening Questions, PHQ-9, the Beck Scale for Suicide Ideation, and the "
    "Suicidal Behaviors Questionnaire-Revised) accounts for nearly half of all records, "
    "while the majority of tools appear only once or twice - a long-tailed distribution.")
para = doc.add_paragraph(style="List Bullet")
add_ins(para,
    "The Columbia-Suicide Severity Rating Scale (C-SSRS), the Ask Suicide-Screening "
    "Questions (ASQ), and the Patient Health Questionnaire-9 (PHQ-9) were the three most "
    "frequently identified tools in the corpus, together accounting for 237 of 626 "
    "tool-level records (approximately 38%) among the 216 catalogued tools."
)
p("These frequently used tools represent high‑value candidates for standardization "
  "because they:")
for item in ["are widely adopted", "have structured question‑and‑answer formats",
             "are often open or publicly accessible",
             "are suitable for encoding in clinical terminologies"]:
    doc.add_paragraph(item, style="List Bullet")
p("The structured dataset (see Table 4) allowed identification of these high‑frequency "
  "instruments, which later informed the LOINC comparison.")

cap = doc.add_paragraph()
r = cap.add_run(
    "Table 4: Frequency of suicide-screening or suicide-assessment tool appearance across "
    "the 769-article corpus (tools appearing 3 or more times; complete 216-tool list in "
    "Supplementary Table S1). “Previously verified LOINC status” carries forward "
    "determinations made during the original review for matching tool names; entries "
    "marked “Pending verification” are tools newly surfaced or re-ranked by the "
    "corrected normalization methodology (§2.9) and require a fresh LOINC lookup before "
    "final submission."
)
r.bold = True
r.font.size = Pt(9.5)

import csv as _csv
with open("docs/manuscript_revision/Supplementary_Table_S1_tool_frequencies.csv",
          encoding="utf-8") as f:
    _all_tools = list(_csv.DictReader(f))
_freq3plus = [t for t in _all_tools if int(t["frequency"]) >= 3]

_known_loinc = {
    "Columbia-Suicide Severity Rating Scale (C-SSRS)": "Present (multiple C-SSRS codes, e.g. 93373-9)",
    "Ask Suicide-Screening Questions (ASQ)": "Not present - original candidate",
    "Patient Health Questionnaire-9 (PHQ-9)": "Present (44249-1, item-level)",
    "Suicidal Behaviors Questionnaire-Revised (SBQ-R)": "Not present - original candidate",
    "Home, Education, Activities/peers, Drugs/alcohol, Suicidality, Emotions/behaviour, "
    "Discharge resources tool (HEADS-ED)": "Not present - original candidate",
    "Hamilton Depression Rating Scale (HDRS)": "Not present - original candidate",
    "Beck Depression Inventory (BDI-II)": "Present (89210-9)",
    "Edinburgh Postnatal Depression Scale (EPDS)": "Present (71354-5)",
}

tbl4 = doc.add_table(rows=1, cols=4)
tbl4.style = "Light Grid Accent 1"
hdr = tbl4.rows[0].cells
for i, name in enumerate(["Tool Name", "Frequency", "Scope",
                            "Previously verified LOINC status"]):
    hdr[i].text = name
for t in _freq3plus:
    row = tbl4.add_row().cells
    row[0].text = t["canonical_tool_name"]
    row[1].text = t["frequency"]
    row[2].text = "Suicide-specific" if t["scope"] == "suicide-specific" else "Comprehensive"
    row[3].text = _known_loinc.get(t["canonical_tool_name"], "Pending verification")
doc.add_paragraph()

h("3.7 Comparison with LOINC and Identification of Gaps", level=3)
p("Tools with frequency count of three or more were cross‑referenced with the Logical "
  "Observation Identifiers Names and Codes (LOINC) terminology, yielding following three "
  "categories:")
doc.add_paragraph("Tools already represented in LOINC: These tools have existing codes and "
                   "require no further action.", style="List Bullet")
doc.add_paragraph("Tools not represented in LOINC and not suitable for encoding: Reasons "
                   "included proprietary licensing, lack of structured format, or "
                   "insufficient documentation.", style="List Bullet")
para = doc.add_paragraph(style="List Bullet")
add_del(para, "High‑value tools not represented in LOINC but suitable for encoding: ")
add_ins(para, "Candidate tools not represented in LOINC and warranting further evaluation "
    "for suitability: ")
add_normal(para, "These tools were (see Table 5): frequently used; open or publicly "
    "accessible; structured in a Q&A format; straightforward to encode.")
p("These findings demonstrate that generative AI can systematically identify terminology "
  "gaps and support evidence‑based recommendations for expanding clinical standards.")
p("It is important to note that Tools 5. Suicidal Behavior Questionnaire eRevised "
  "(SBQ-R), 9. Hamilton Depression Rating Scale (HDRS), and 11. Depressive Symptom Index "
  "Suicidality Subscale (DSI-SS) in Table 5 are not exclusively focused on suicide "
  "assessment; however, each contains items relevant to evaluating suicide risk and was "
  "therefore retained for completeness. Tool 31. Substance Abuse and Mental Health "
  "Services Administration (SAMHSA) Suicide Assessment Five-step Evaluation and Triage "
  "(SAFE-T) (34), although referenced only twice in the literature, was identified by "
  "psychology subject-matter experts as a widely used instrument in the United States and "
  "was initially considered a candidate for inclusion in LOINC. Nevertheless, subsequent "
  "evaluation determined that the SAFE-T does not meet LOINC’s requirements for "
  "codification, primarily because it serves as an evaluation and triage tool that does "
  "not align with the structural requirements typically associated with LOINC "
  "submissions. LOINC primarily encompasses data‑collection instruments that capture "
  "clinical observations through discrete, codifiable questions and answers. Its model is "
  "specifically designed to represent variables, answer lists, and the structured "
  "collections that contain them (35). In contrast, SAFE‑T operates as a clinical "
  "decision‑support framework rather than a standardized assessment instrument with "
  "measurable, codifiable data elements. Its purpose is to guide clinical judgment, not to "
  "generate discrete observational data suitable for encoding within the LOINC framework. "
  "For this reason, SAFE‑T was ultimately excluded from consideration for LOINC "
  "representation.")

para = doc.add_paragraph()
add_ins(para,
    "We clarify two related points raised during peer review. First, the Columbia-Suicide "
    "Severity Rating Scale and the Patient Health Questionnaire-9 were both identified and "
    "extracted by the pipeline; far from being absent from the corpus, they are, "
    "respectively, the first and third most frequently occurring tools overall (§3.6), "
    "together with the ASQ accounting for approximately 38% of all tool-level records. "
    "Second, the inclusion of the Hamilton Depression Rating Scale and the SAFE-T reflects "
    "two analytically distinct properties that should not be conflated: suicide-specificity "
    "and structural suitability for LOINC coding. The Hamilton Depression Rating Scale is "
    "not suicide-specific - it is a general depression-severity instrument - but its "
    "item 3 assesses suicidal ideation directly, which is why it is classified in this "
    "study as a comprehensive mental-health tool containing a suicide-related item rather "
    "than a suicide-specific instrument (§2.10), and why it was retained rather than "
    "excluded. SAFE-T, by contrast, is suicide-specific by design and purpose (it is "
    "explicitly a suicide assessment and triage protocol) and is classified as such in this "
    "study; its exclusion from the five LOINC candidates was not a judgment that it is "
    "insufficiently suicide-related, but the narrower, structural finding that its "
    "clinical-judgment-guiding format does not align with LOINC's model of discrete, "
    "codifiable observations, as detailed above."
)

cap = doc.add_paragraph()
r = cap.add_run(
    "Table 5: Candidate tools not currently represented in LOINC and warranting further "
    "evaluation for LOINC submission (frequencies updated to the corrected 769-article "
    "corpus and 216-tool human-verified list; LOINC-absence status carried forward from "
    "original verification, tool identity confirmed by name-matching against the "
    "corrected list). SAFE-T (frequency 2, below the table's frequency-3 threshold) is "
    "discussed separately in §3.7 as a case excluded from LOINC candidacy despite meeting "
    "the frequency and accessibility criteria."
)
r.bold = True
r.font.size = Pt(9.5)
table5_data = [
    ("Ask Suicide-Screening Questions (ASQ)", "45", "Open source", "Screening"),
    ("Suicidal Behaviors Questionnaire-Revised (SBQ-R)", "18", "Open source", "Assessment"),
    ("Hamilton Depression Rating Scale (HDRS)", "6", "Open source", "Assessment"),
    ("Depressive Symptom Inventory-Suicidality Subscale (DSI-SS)", "5", "Open source",
     "Screening"),
    ("Home, Education, Activities/peers, Drugs/alcohol, Suicidality, Emotions/behaviour, "
     "Discharge resources tool (HEADS-ED)", "9", "Open source", "Screening"),
]
tbl5 = doc.add_table(rows=1, cols=4)
tbl5.style = "Light Grid Accent 1"
hdr = tbl5.rows[0].cells
for i, name in enumerate(["Tool Name", "Frequency", "Proprietary/Open Source",
                            "Screening/Assessment"]):
    hdr[i].text = name
for row_data in table5_data:
    row = tbl5.add_row().cells
    for i, val in enumerate(row_data):
        row[i].text = val
doc.add_paragraph()

h("3.8 Computational Workload, Cost, and Scalability Characteristics", level=3)
para = doc.add_paragraph()
add_normal(para, "Although the primary focus of this study is clinical and terminological, "
    "the large-scale application of Gen AI to ")
add_del(para, "639")
add_ins(para, "769")
add_normal(para, " full-text articles also yielded important operational insights.")
add_ins(para, " Precise per-article query, token, and cost telemetry was captured during "
    "controlled testing (§2.6–2.7) but was not centrally logged across the full production "
    "run; the figures below scale the originally reported per-article averages "
    "proportionally to the corrected article count and should be treated as estimates "
    "pending confirmation against production logs, rather than as independently "
    "re-measured totals.")
add_normal(para, " The research quantifies the workload and cost profile of the pipeline:")

for item_old, item_new in [
    ("Query volume: With average of 60 queries per PDF, total 38,280 operational queries "
     "were used across the corpus, reflecting the multiple paraphrased variants used for "
     "each KPI.",
     "Query volume: With an average of 60 queries per PDF, an estimated total of "
     "approximately 46,140 operational queries were used across the corrected corpus, "
     "reflecting the multiple paraphrased variants used for each KPI."),
    ("Token usage: Total tokens spanning both embedding and generation models included "
     "~98M for input text, ~15M for embedding vectors, while ~1.6M for the output answers.",
     "Token usage: Scaling proportionally, total tokens spanning both embedding and "
     "generation models are estimated at ~118M for input text, ~18M for embedding vectors, "
     "and ~1.9M for output answers."),
    ("Cost: With average of 3 cents per PDF (639 * 0.03 = $19.17), the entire large-scale "
     "extraction was achieved for under $20 in model usage costs.",
     "Cost: With an average of approximately 3 cents per PDF (769 * 0.03 ≈ $23.07), the "
     "entire large-scale extraction was achieved for approximately $23 in model usage "
     "costs."),
    ("Processing time: Finally, with average ~4min processing time per PDF, it took total "
     "of (639 * 4min) ~43 hours, indicating that the pipeline is tractable for batch "
     "processing of hundreds of articles and could be parallelized further in production "
     "environments.",
     "Processing time: With an average of ~4min processing time per PDF, total processing "
     "time is estimated at approximately (769 * 4min) ~51 hours, indicating that the "
     "pipeline remains tractable for batch processing of hundreds of articles and could be "
     "parallelized further in production environments."),
]:
    para = doc.add_paragraph(style="List Bullet")
    add_del(para, item_old)
    add_ins(para, item_new)

p("These characteristics support the potential for applying similar pipelines to other "
  "clinical domains where unstructured literature or documentation must be transformed "
  "into structured, interoperable data. To summarize, the designed solution provides "
  "following high-value characteristics.")
para = doc.add_paragraph(style="List Bullet")
add_del(para, "High throughput, with all 639 articles processed using a consistent, "
    "automated pipeline")
add_ins(para, "High throughput, with all 769 articles processed using a consistent, "
    "automated pipeline")
for item in ["Efficient retrieval, supported by dual‑space matching using both text "
             "chunks and hypothetical questions",
             "Low cost and modest compute requirements, demonstrating feasibility for "
             "broader application",
             "Robust handling of heterogeneous PDFs, including multi‑column text, "
             "tables, and images"]:
    doc.add_paragraph(item, style="List Bullet")

h("3.9 Synthesis of Key Findings", level=3)
p("Across all analyses, the results demonstrate that:")
para = doc.add_paragraph(style="List Bullet")
add_normal(para, "A retrieval‑augmented generative AI pipeline can reliably extract "
    "structured, clinically relevant metadata from hundreds of full‑text research "
    "articles.")
para = doc.add_paragraph(style="List Bullet")
add_del(para, "The contemporary suicide‑risk literature is highly heterogeneous, "
    "encompassing 239 tools across 71 countries.")
add_ins(para, "The open-access suicide‑risk literature sampled in this study is highly "
    "heterogeneous, encompassing 216 tools across 76 countries.")
doc.add_paragraph("Structured extraction enables systematic identification of frequently "
    "used tools and reveals clear gaps in existing clinical terminologies.", style="List Bullet")
para = doc.add_paragraph(style="List Bullet")
add_del(para, "Several high‑value, widely used, open, and structured suicide‑risk "
    "instruments are not currently represented in LOINC, presenting opportunities for "
    "standardization.")
add_ins(para, "Several widely used, open, and structured suicide‑risk instruments are not "
    "currently represented in LOINC, presenting candidate opportunities for further "
    "standardization evaluation.")
doc.add_paragraph("The pipeline is computationally efficient and scalable, supporting its "
    "use in broader digital health applications.", style="List Bullet")

para = doc.add_paragraph()
add_del(para, "These findings directly support the overarching goal of this research. By "
    "supporting the development of structured, interoperable representations of "
    "suicide‑risk information, this work aims to enhance the visibility and clinical "
    "utility of suicide‑risk data within EHR systems and strengthen the foundation for "
    "timely, data‑driven intervention.")
add_ins(para, "These findings support the overarching goal of this proof-of-concept "
    "research: by supporting the development of structured, interoperable representations "
    "of suicide‑risk information, this work aims to contribute toward the future "
    "visibility of suicide‑risk data within EHR systems. Establishing a direct effect on "
    "clinical utility or timely, data‑driven intervention would require subsequent "
    "clinical-implementation studies beyond the scope of this literature-mapping work.")

doc.add_page_break()

# =========================================================== 4 DISCUSSION =
h("4 Discussion", level=2)

para = doc.add_paragraph()
add_del(para, "This paper aimed to investigate the application of a generative AI pipeline "
    "to quantify and analyze the current state of suicide prevention tools within EHR "
    "systems.")
add_ins(para, "This paper aimed to investigate the application of a generative AI pipeline "
    "to quantify and analyze the current state of suicide prevention tools as represented "
    "in the open-access research literature - a proof-of-concept for AI-supported "
    "literature mapping and terminology-gap identification, rather than a study of clinical "
    "implementation, EHR use, or patient outcomes.")
add_normal(para, " A literature search identified ")
add_del(para, "639")
add_ins(para, "769")
add_normal(para, " documents to be evaluated for suicide assessment tool key performance "
    "indicators. Of the ")
add_del(para, "239 unique suicide-related tools that were identified, 5 of them would be "
    "good candidates")
add_ins(para, "216 unique suicide-related tools that were identified, 5 of them were "
    "identified as candidates warranting further evaluation")
add_normal(para, " for LOINC encoding. These tools are frequently used, publicly "
    "accessible, in a structured format, and not currently encoded.")

p("Generative AI has been successfully implemented in systematic reviews in the "
  "literature (Blaizot et al., 2022; Mahuli et al., 2023). In alignment with both Blaizlot "
  "and Mahuli we found the generative AI pipeline to streamline and increase the "
  "efficiency of extracting relevant metadata. We found that the AI pipeline performed "
  "best when the query was specific and structured. This query specificity may be why the "
  "authors found increased alignment between human and AI summaries. Query specificity is "
  "worth noting for future research.")

p("To our knowledge, this is the only paper using a generative AI pipeline to evaluate "
  "suicide tools in the literature. In a systematic review of GenAI in mental health, most "
  "papers used AI to detect the presence of suicidality or depression from text data (22). "
  "Unlike the bulk of previous research, this paper used AI to map and analyze the tools "
  "clinicians use to measure suicidality. This demonstrates that generative AI can be used "
  "to not only detect suicidality in patients but may be used to standardize the field to "
  "encourage continuity of care.")

para = doc.add_paragraph()
add_del(para, "Researchers found 239 suicide tools, a finding that reflects a lack of "
    "standardization in suicide screening. This heterogeneity aligns with the current "
    "state of non-standardization in suicide screening practice.")
add_ins(para, "Researchers found 216 suicide tools in the sampled open-access literature, "
    "a finding that reflects substantial heterogeneity in how suicide screening is studied "
    "and reported within this corpus. This heterogeneity in the research literature is "
    "consistent with, though not direct evidence of, non-standardization in suicide "
    "screening clinical practice more broadly; distinguishing the two is an important "
    "caveat we did not sufficiently make in the original submission (see §4.2).")
add_normal(para, " The US Preventative Task Force did not find conclusive evidence for the "
    "benefits or harms of screening for suicide risk in a primary care setting (8). This "
    "lack of evidence and guidance is probably a contributor to the large variety of "
    "suicide tools evaluated by the generative AI pipeline. Additionally, the "
    "proliferation of tools may also reflect clinician expertise as different "
    "demographics may benefit from different screening tools; such as pediatrics compared "
    "to veterans (7,38). There were ")
add_del(para, "208")
add_ins(para, "262")
add_normal(para, " articles that lacked an identifiable suicide-related tool. While the "
    "reasons for this cannot be definitive given the data, it may reflect gaps in "
    "standardized screening tools (9), which may place the burden of suicidality detection "
    "on individual clinicians rather than structured screening tools.")

para = doc.add_paragraph()
add_normal(para, "Five tools were identified by the GenAI pipeline as ")
add_del(para, "strong candidates")
add_ins(para, "candidates warranting further evaluation")
add_normal(para, " for LOINC codification (ASQ, SBQ-R, HDRS, DSI-SS, and HEADS-ED). Prior "
    "research has suggested that linking questionnaires to LOINC improves interoperability "
    "(39). The five candidate tools identified in this study represent a criteria-driven "
    "foundation for expanding LOINC coverage of suicide screener tools, with the potential "
    "to strengthen interoperability, support clinical decision-making, and improve the "
    "visibility of suicide-risk information across care settings.")

para = doc.add_paragraph()
add_del(para, "In conclusion, to our knowledge this is the first paper to use Generative AI "
    "to complete a large-scale literature review about suicide prevention tools.")
add_ins(para, "In conclusion, to our knowledge this is the first paper to use Generative AI "
    "to complete a large-scale literature mapping of suicide prevention tools within the "
    "open-access literature.")
add_normal(para, " The GenAI pipeline completed a large-scale literature review with "
    "minimal cost while using a reproducible workflow, indicating that the pipeline is "
    "applicable to many domains. This paper also identified potential gaps in the current "
    "LOINC suicide risk tool code system. The results of the paper imply a lack of "
    "standardization of suicide screening tools ")
add_del(para, "in the field.")
add_ins(para, "within the sampled literature, which may or may not extend to clinical "
    "practice.")
add_normal(para, " Interoperable data limits the visibility of suicidality of patients in "
    "care transitions, reducing clinician effectiveness (15–18). Continuity of care "
    "across healthcare systems could be aided by codifying the commonly used suicide "
    "screener tools identified by the GenAI pipeline.")

h("4.1 Strengths", level=3)
para = doc.add_paragraph()
add_normal(para, "A major strength of this study is the breadth and depth of the corpus. "
    "The pipeline processed ")
add_del(para, "639")
add_ins(para, "769")
add_normal(para, " full‑text, open‑access research articles, enabling a panoramic view "
    "of contemporary suicide‑risk research. Because the system operated on full text "
    "rather than abstracts, it captured nuanced details about tools, populations, settings, "
    "and outcomes that are often absent from meta analysis. This allowed the study to "
    "identify ")
add_del(para, "239 unique suicide‑screening or assessment tools used across 71 countries")
add_ins(para, "216 unique suicide‑screening or assessment tools used across 76 "
    "countries")
add_normal(para, ", providing an unprecedented empirical map of the global "
    "suicide‑risk instrument landscape.")

p("The study employed a consistent, predefined set of 12 key performance indicators "
  "(KPIs) for every tool identified. This uniform structure enabled direct comparison "
  "across heterogeneous studies and ensured that extracted information was aligned with "
  "clinically meaningful dimensions such as tool type, clinical setting, demographics, and "
  "associated medical conditions.")

para = doc.add_paragraph()
add_normal(para, "The use of a dual‑space retrieval architecture, combining "
    "text‑chunk embeddings with hypothetical question embeddings, substantially "
    "strengthened the accuracy of context retrieval. This design allowed the system to "
    "match queries even when articles used varied or indirect phrasing. The resulting "
    "outputs showed high fidelity to source text, with gold‑sample evaluation "
    "demonstrating ")
add_del(para, "strong")
add_ins(para, "generally strong, though not uniform,")
add_normal(para, " faithfulness, answer relevancy, and context recall")
add_ins(para, " across an expanded, 20-article validation set (§3.3)")
add_normal(para, ".")

p("By cross‑referencing extracted tools with existing LOINC entries, the study "
  "produced a data‑driven assessment of terminology coverage. The methodology "
  "prioritized tools based on frequency, openness, structure, and suitability for "
  "encoding, enabling the identification of high‑value candidates for standardization. "
  "This evidence‑based approach directly supports ongoing efforts to improve "
  "interoperability of suicide‑risk information in clinical systems.")

p("The pipeline demonstrated that large‑scale extraction from hundreds of PDFs can be "
  "performed efficiently and at low cost. The modular architecture, incorporating "
  "preprocessing, retrieval, generation, and optional evaluation, can be adapted to other "
  "clinical domains where unstructured literature or documentation must be converted into "
  "structured, computable data. This positions the approach as a generalizable framework "
  "for digital health research and terminology development.")

h("4.2 Limitations", level=3)
para = doc.add_paragraph()
add_normal(para, "The study was restricted to open‑access, full‑text publications, "
    "which, while ensuring complete document availability, may introduce selection bias. "
    "Important suicide‑risk tools discussed in subscription‑only journals or "
    "proprietary clinical research may not be represented. As a result, the set of ")
add_del(para, "239")
add_ins(para, "216")
add_normal(para, " tools reflects the open‑access literature rather than the full "
    "universe of tools used in clinical practice.")

doc.add_paragraph(
    "This search strategy also relied on the specific terms “suicide screening tool” "
    "and “suicide assessment tool.” Articles that study a named instrument without "
    "using these exact words in the discoverable text, or that use synonymous terminology "
    "(e.g., “suicide risk evaluation”), may have been under-represented, and this "
    "limitation was not previously stated explicitly."
)

p("Despite the structured extraction pipeline, inconsistencies in how tools are named or "
  "described across studies can introduce ambiguity. Some articles refer to instruments "
  "using abbreviations, partial names, or modified versions, while others describe "
  "suicide‑risk assessment without naming a specific tool. This variability can affect "
  "tool identification and frequency counts.")

p("The classification of tools into screening or assessment categories relies on how "
  "each article describes the instrument. However, some tools are used in both "
  "capacities, and authors may label them inconsistently. This can lead to classification "
  "noise in the tool-type KPI and may affect downstream analyses of tool purpose and "
  "clinical applicability.")

p("Although the RAG pipeline demonstrated strong performance on gold samples, generative "
  "models can still produce errors when articles contain ambiguous phrasing, overlapping "
  "constructs, or complex methodological descriptions. Free‑text KPIs such as "
  "demographics summaries or outcome summaries may include minor omissions or "
  "over‑generalizations. While the system was explicitly instructed to avoid "
  "speculation, the risk of subtle generative inaccuracies cannot be fully eliminated.")

para = doc.add_paragraph()
add_del(para, "The accuracy evaluation used four manually annotated gold‑sample "
    "articles, which provided detailed insight into model performance but represent a "
    "small fraction of the 639‑article corpus. While the RAGAS scores were strong, a "
    "larger validation set would provide more robust estimates of extraction accuracy "
    "across diverse study designs, writing styles, and clinical contexts.")
add_ins(para, "The accuracy evaluation was expanded from an initial four to 20 manually "
    "annotated gold‑sample articles (2.60% of the 769‑article corpus), yielding 495 "
    "article–KPI extraction rows and 1,485 individual measurements - a sample size "
    "exceeding a standard 95%-confidence, ±3%-margin power calculation against the corpus "
    "population (§2.7). These measurements are not independently and randomly drawn "
    "observations, however: they are clustered within 20 source articles, with roughly 25 "
    "correlated measurements per article on average, so the effective unit of independent "
    "replication is closer to 20 than to 1,485, and this comparison should be read as "
    "supporting evidence of adequate measurement volume rather than a formal confidence "
    "interval on corpus-wide accuracy.")

para = doc.add_paragraph()
add_del(para, "Human‑generated gold answers served as the reference standard, but the "
    "study did not assess inter‑annotator agreement or variability in human "
    "interpretation. Some KPIs, especially free‑text summaries, may have multiple valid "
    "representations, and the absence of multi‑annotator comparison limits the ability "
    "to quantify human‑level variability relative to model outputs.")
add_ins(para, "Human‑generated gold answers - produced by two domain experts, a "
    "physician and a terminology/standardization expert, working independently of the "
    "pipeline's own output - served as the reference standard, but the study did not "
    "assess formal inter‑annotator agreement between them. Some KPIs, especially "
    "free‑text summaries, may have multiple valid representations, and the absence of a "
    "formal agreement statistic limits the ability to quantify human‑level variability "
    "relative to model outputs.")

p("The study analyzed research articles rather than real‑world clinical notes or EHR "
  "data. While this provides a strong foundation for identifying tools used in the "
  "literature, it does not directly measure how frequently these tools are used in "
  "clinical practice or how they are documented in operational systems. Future work will "
  "be needed to validate the relevance of identified tools in real‑world care "
  "environments.")

p("The KPI framework, while comprehensive for literature‑level extraction, does not "
  "capture several factors that strongly influence how suicide‑risk tools are selected "
  "and applied in real‑world clinical environments. In clinical decision-making, "
  "physicians, hospitals, and researchers frequently depend on additional dimensions not "
  "included in the current KPI set. These include (1) psychometric properties such as "
  "sensitivity, specificity, predictive validity, and reliability; (2) administration "
  "burden, including time to complete, required training, and workflow fit; (3) "
  "regulatory or accreditation requirements, such as Joint Commission mandates or "
  "state‑level screening policies; (4) population‑specific validation, including "
  "evidence for use in pediatric, geriatric, culturally diverse, or neurodivergent "
  "populations; (5) risk‑stratification capabilities, such as whether the tool "
  "supports tiered risk levels or actionable thresholds; and (6) implementation "
  "characteristics, including licensing constraints, digital availability, and "
  "integration readiness for EHR systems.")

p("Because these KPIs were not extracted in the current study, the resulting landscape "
  "of tools reflects frequency of use in research literature rather than the full set of "
  "factors that shape clinical adoption. In practice, clinicians may favor tools "
  "exhibiting robust predictive validity, despite their infrequent appearance in "
  "published studies, whereas hospitals may prioritize instruments that comply with "
  "regulatory standards or reduce workflow disruption. Similarly, researchers may select "
  "tools based on psychometric rigor or suitability for specific populations, which may "
  "not correlate with overall frequency in literature. The absence of these additional "
  "KPIs therefore limits the ability to fully interpret why certain tools are used in "
  "particular contexts and may obscure important distinctions between tools that are "
  "clinically preferred versus those that are academically prevalent. Future work "
  "incorporating these broader dimensions would enable a more holistic understanding of "
  "tool selection and support more nuanced recommendations for standardization and "
  "terminology development.")

doc.add_page_break()

# ==================================================== BACK MATTER =========
h("5 Conflict of Interest", level=2)
para = doc.add_paragraph()
r = para.add_run("The authors declare that the research was conducted in the absence of "
                  "any commercial or financial relationships that could be construed as a "
                  "potential conflict of interest.")
r.italic = True

h("6 Author Contributions", level=2)
p("DS: Conceptualization, Data curation, Methodology, Formal analysis, Writing – original "
  "draft, Writing – review & editing. CH: Writing – original draft, Writing – review & "
  "editing. VL: Writing – review & editing. GD: Writing – review & editing. BG: Writing – "
  "original draft, Writing – review & editing.")

h("7 Funding", level=2)
p("No funding was received for the writing or publishing of this manuscript.")

h("8 Acknowledgments", level=2)
p("Dhruv Saitwal would like to express sincere gratitude to Ms. Kelly Samuelson, Ms. "
  "Michelle Zancan, Ms. Hana Al’Absi, Dr. Sossong, and Ms. Tonya Tipton for their "
  "invaluable support and guidance throughout this work.")

h("9 Data Availability Statement", level=2)
para = doc.add_paragraph()
add_normal(para, "The datasets generated and analyzed for this study can be found at "
    "https://github.com/dhruvksaitwal-debug/gen-ai-suicide-tools. Please see the "
    "“Availability of data” section of Materials and data policies in the Author "
    "guidelines for more details.")
add_ins(para, " The complete list of 216 canonical tools with frequency and "
    "suicide-specific/comprehensive classification is additionally provided as "
    "Supplementary Table S1.")

doc.add_page_break()
h("References", level=2)
p("[Reference list carried forward unchanged from the submitted version - see original "
  "manuscript refs. 1-39. New in-text citations added during revision (e.g., the RAG and "
  "LOINC definitions in §1) reuse existing reference numbers 31, 33, and 35 and require no "
  "new bibliography entries.]")

note("Two new references are cited in the revised §2.2 (LLM selection rationale) and "
     "should be appended as refs. 40-41. Suggested entries, to be verified and formatted "
     "to house style by the corresponding author:")
p("40. OpenAI. GPT-4o mini: advancing cost-efficient intelligence [Internet]. 2024 Jul 18 "
  "[cited 2026 Aug]. Available from: "
  "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/")
note("Benchmark figures (MMLU 82.0% for gpt-4o-mini vs. 77.9% Gemini 1.5 Flash and 73.8% "
     "Claude 3 Haiku; per-token pricing) are drawn from this source and third-party "
     "aggregator reporting current as of August 2026 - please verify against the primary "
     "source before final submission, as model benchmark reporting is periodically revised.")
p("41. [Structured information extraction cost/accuracy benchmark comparing GPT-4o to "
  "GPT-4o mini on pathology report extraction (96.1% vs. 87.9% accuracy; $0.44 vs. $0.01 "
  "per report) - full citation to be confirmed by the corresponding author from the "
  "arXiv preprint identified during this revision (arXiv:2502.12183) before final "
  "submission, as preprint-to-publication status may have changed.]")

doc.save(OUT)
print(f"Final manuscript saved: {OUT}")
