"""Generates the response-to-reviewers letter as a Word document."""
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT_PATH = "docs/manuscript_revision/Response_to_Reviewers.docx"

AUTHOR_INPUT_COLOR = RGBColor(0xB0, 0x00, 0x00)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    return h


def add_comment(doc, ref, comment_text):
    p = doc.add_paragraph()
    run = p.add_run(f"{ref}: ")
    run.bold = True
    run2 = p.add_run(comment_text)
    run2.italic = True
    return p


def add_response(doc, response_text, author_input=None):
    p = doc.add_paragraph()
    run = p.add_run("Response: ")
    run.bold = True
    p.add_run(response_text)
    if author_input:
        p2 = doc.add_paragraph()
        run = p2.add_run("[AUTHOR INPUT NEEDED: " + author_input + "]")
        run.font.color.rgb = AUTHOR_INPUT_COLOR
        run.bold = True
    doc.add_paragraph()  # spacing


def build():
    doc = docx.Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title = doc.add_heading("Response to Reviewers", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run(
        "Original title: “Improving Suicide-Safe Care through Gen AI-Driven "
        "Recommendations for Tool Standardization in LOINC”"
    )
    r.italic = True
    sub2 = doc.add_paragraph()
    sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub2.add_run(
        "Revised title (see response to Reviewer 4, Comment 1): “Toward Suicide-Safe "
        "Care: Gen AI-Driven Mapping of Suicide Screening and Assessment Tools for LOINC "
        "Standardization”"
    )
    r2.italic = True
    doc.add_paragraph()

    intro = doc.add_paragraph()
    intro.add_run(
        "We thank the reviewers for their careful and constructive assessment of the "
        "manuscript. In response, we have made substantial revisions across the corpus, "
        "validation, methodology, and framing of the manuscript. The most significant "
        "changes are summarized here and detailed point-by-point below:"
    )
    bullets = [
        "The 129 articles previously excluded for exceeding the 1.5 MB processing threshold "
        "have now been reprocessed and included in the analysis (via PDF compression, with "
        "unrecoverable articles replaced by equivalent newly-identified open-access articles "
        "meeting the same inclusion criteria), bringing the total corpus from 639 to 769 "
        "articles and total tool-level outputs from 841 to 1,012 CSVs.",
        "The gold-standard validation set has been expanded from 4 to 20 manually-annotated "
        "reference articles (495 article–KPI extraction rows, 1,485 individual RAGAS "
        "measurements), exceeding the sample size indicated by a formal power calculation "
        "against the corpus population.",
        "The tool-name normalization procedure has been made fully transparent and "
        "documented: an automated pass yielded 319 candidate canonical tools, which two "
        "domain experts (a physician and a terminology/standardization expert) then "
        "reviewed and classified, producing a final human-verified total of 216 unique "
        "suicide-related tools (130 suicide-specific, 86 comprehensive mental-health tools "
        "with a suicide-related item/subscale) - replacing the previously unexplained "
        "figure of 239 - with the complete tool list, frequencies, and classification now "
        "provided as a supplementary table.",
        "All tables and figures have been integrated into the main text.",
        "Key terms (Generative AI, Large Language Models, Retrieval-Augmented Generation, "
        "LOINC) are now defined at first use.",
        "Claims about clinical impact have been tempered throughout to accurately reflect "
        "the study's scope as a literature-mapping and terminology-gap-identification "
        "proof-of-concept.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")
    doc.add_paragraph()

    # ================= REVIEWER 1 =================
    add_heading(doc, "Reviewer 1", level=1)
    doc.add_paragraph(
        "We thank the reviewer for the detailed methodological critique. We address each "
        "point below."
    )

    add_comment(
        doc, "Comment 1",
        "Only two figures are included in the manuscript, with all tables included in the "
        "supplemental materials. All tables and figures should be integrated into the "
        "manuscript..."
    )
    add_response(
        doc,
        "All five tables and both figures have been moved into the main text, placed as "
        "close as possible to their first textual reference: Figure 1 (inclusion/exclusion "
        "flow) in §2.1, Table 1 (query-to-KPI mapping) in §2.4, Figure 2 (pipeline "
        "diagram) in §2.6, Table 2 (sample gold-standard comparison) and Table 3 "
        "(qualitative KPI summary) in §3.3–§3.4, and Table 4 (tool frequency) "
        "and Table 5 (LOINC gap candidates) in §3.6–§3.7. No content remains in "
        "supplementary materials except the full 216-tool frequency list newly added in "
        "response to Reviewer 4, Comment 5, which is too large to place inline."
    )

    add_comment(
        doc, "Comment 2",
        "The authors need to be more specific about the Gen AI tools they are using in the "
        "Intro. Very obviously, it they are using LLMs but this should be specified..."
    )
    add_response(
        doc,
        "The Introduction now explicitly states that “Generative AI” in this study "
        "refers specifically to large language models (LLMs), and names the specific models "
        "used - gpt-4o-mini for text generation and image interpretation, and "
        "text-embedding-3-small for embeddings - at first mention, rather than deferring "
        "this detail to the Methods section."
    )

    add_comment(
        doc, "Comment 3",
        "The authors mention RAGs in the Intro but don't provide context for uninformed "
        "readers about what RAGs are or how they work..."
    )
    add_response(
        doc,
        "A plain-language definition of Retrieval-Augmented Generation has been added at "
        "first mention in the Introduction: “Retrieval-Augmented Generation (RAG) is an "
        "architecture in which a language model's response is grounded in text retrieved "
        "from an external document collection at query time, rather than relying solely on "
        "information encoded during model training - reducing the risk of fabricated or "
        "unsupported output when the model is asked about specific source documents.” The "
        "technical elaboration in §2.6 remains unchanged."
    )

    add_comment(
        doc, "Comment 4",
        "On page 5, lines 56-58 the authors cite research by Elosyph and others about use "
        "of LLMs to identify suicide risks. The citations do not support the assertion the "
        "authors make..."
    )
    add_response(
        doc,
        "We revised the sentence to accurately characterize these citations. The passage "
        "now reads: “General-purpose, untrained LLMs have shown promise in "
        "suicide-risk-relevant classification tasks using constructed clinical vignettes "
        "(28–30), suggesting the broader feasibility of applying LLMs to suicide-related "
        "text; however, this prior work evaluated risk detection in synthetic case "
        "descriptions rather than structured information extraction from full-text "
        "literature, and should not be read as direct precedent for the specific extraction "
        "task undertaken here.”"
    )

    add_comment(
        doc, "Comment 5",
        "The authors need to define and describe what LOINC is."
    )
    add_response(
        doc,
        "A definition has been added at first mention in the Introduction: “Logical "
        "Observation Identifiers Names and Codes (LOINC) is a freely available, "
        "internationally adopted standard terminology, maintained by the Regenstrief "
        "Institute, that assigns unique codes to clinical observations, measurements, and "
        "structured assessment instruments - enabling the same clinical concept to be "
        "consistently identified and exchanged across different electronic health record "
        "systems.”"
    )

    doc.add_paragraph().add_run("Methodological issues").bold = True

    add_comment(
        doc, "Comment 6",
        "The authors need to further describe in detail their literature review "
        "strategies. Some important issues to discuss include: a) Why was PRISMA not "
        "used; b) Why were only two limited search terms used and the rationale for why "
        "these specific terms; c) Describe why a substantial number of articles were "
        "excluded due to file size and how this may have biased the results or was/was "
        "not a confound readers should be concerned about. This should also be addressed "
        "in the Limitations section, but the authors only mention once and its in the "
        "figure not in the text."
    )
    add_response(
        doc,
        "We address each part of this comment. (a) A new paragraph in Methods §2.1 now "
        "explicitly explains why PRISMA was not used: this study is a literature-mapping "
        "exercise cataloging instrument usage, not a synthesis of intervention outcomes or "
        "efficacy, and PRISMA-ScR (the scoping-review extension) is identified as the more "
        "directly applicable framework, whose core principles (explicit eligibility "
        "criteria, transparent search strategy, structured charting) we followed in spirit "
        "without completing a formal checklist. (b) The rationale for the two search terms "
        "is now stated explicitly in §2.1: they were chosen to mirror the terminology used "
        "in the pipeline's own KPI-extraction queries (Table 1), ensuring consistency "
        "between article identification and downstream extraction; the resulting "
        "limitation (potential under-representation of articles using different "
        "terminology) is now stated explicitly rather than left implicit. (c) The "
        "file-size exclusion is no longer an unaddressed confound: as described in our "
        "response to Comment 3 above, the 129 originally excluded articles were "
        "reprocessed and included, and §2.1 now contains a full explanatory paragraph in "
        "the main text (not only the figure) describing this recovery process, directly "
        "resolving the reviewer's specific complaint that this was previously mentioned "
        "only in the figure."
    )

    add_comment(
        doc, "Comment 7",
        "The authors should provide an in-depth discussion of why they selected the LLMs "
        "used in the investigation. This should be based on benchmarking data and not "
        "simply because of accessibility or name recognition..."
    )
    add_response(
        doc,
        "A new paragraph in Methods §2.2 now grounds the model selection in published "
        "benchmark data rather than accessibility alone: gpt-4o-mini's general-capability "
        "benchmark performance (82.0% MMLU) relative to comparably-priced small models "
        "(Gemini 1.5 Flash 77.9%, Claude 3 Haiku 73.8%), its order-of-magnitude lower cost "
        "relative to larger frontier models, and a published structured-extraction "
        "benchmark showing that a full-size frontier model's accuracy advantage over the "
        "mini-tier model (96.1% vs. 87.9% in one pathology-report extraction study) comes "
        "at roughly 44 times the per-document cost, a tradeoff that does not favor the "
        "larger model at this study's scale (hundreds of articles on a limited budget). We "
        "state transparently that a bespoke head-to-head comparison of multiple LLMs on "
        "this specific extraction task was not conducted, and note this as an opportunity "
        "for future work rather than overstating the rigor of the original model choice.",
        author_input=(
            "the two new supporting references (draft entries 40-41 in the revised "
            "manuscript) need your verification and formatting to house style before "
            "submission - one is an OpenAI model-card source for the benchmark figures "
            "cited, the other is an arXiv preprint (2502.12183) on structured-extraction "
            "cost/accuracy benchmarking whose publication status should be confirmed."
        ),
    )

    add_comment(
        doc, "Comment 8",
        "There is just simply not enough of an explanation of the strategy used to "
        "generate the 9 natural language queries that were used in the investigation... "
        "The authors should provide a rationale for the selection of the queries used in "
        "the investigations."
    )
    add_response(
        doc,
        "A new paragraph in Methods §2.4 describes the query design strategy in full: "
        "the nine base queries were derived directly from the 12-field output schema "
        "using a KPI-first process (authoring the question a domain expert would need "
        "answered to populate each field, then grouping closely related fields under one "
        "query), and prompt wording for both the base queries and the answer-generation "
        "prompt was iteratively refined against the manually annotated gold-standard "
        "articles prior to large-scale extraction, rewriting phrasings that produced low "
        "faithfulness, relevancy, or context-recall scores and re-evaluating until "
        "performance stabilized. We note transparently that this was an iterative, "
        "evaluation-driven refinement process rather than a formal ablation study "
        "isolating each individual change's contribution."
    )

    add_comment(
        doc, "Comment 9",
        "There is inadequate description of the human annotated gold standard used to "
        "evaluate the LLM performance. Basic info should include who (background "
        "knowledge, expertise, not names) and any data regarding rates of agreements "
        "between the reviewers..."
    )
    add_response(
        doc,
        "This is addressed in full in our response to Reviewer 4, Comment 4: the "
        "gold-standard annotator background (a physician and a terminology/standardization "
        "expert) is now stated explicitly in Methods §2.7, and the absence of a formal "
        "inter-annotator agreement statistic is now stated as an explicit limitation in "
        "§4.2 rather than left unaddressed, directly per this reviewer's specific request."
    )

    add_comment(
        doc, "Comment 10",
        "A major problem with the results of the manuscript is that some of the items "
        "identified were not actually suicide measures or not specific to suicide... Why "
        "did the authors not include other articles about very well known and widely used "
        "suicide screening measures, such as the Columbia Suicide Severity Scale, or even "
        "something as widely used around the world as the PHQ-9? This likely represents a "
        "bias in the authors search..."
    )
    add_response(
        doc,
        "We respectfully clarify two points here rather than treat this as an "
        "unaddressed bias. First, the Columbia-Suicide Severity Rating Scale and the "
        "PHQ-9 were in fact identified and extracted throughout the corpus: they are, "
        "respectively, the single most frequent and third most frequent tools overall, "
        "together with the Ask Suicide-Screening Questions (ASQ) accounting for "
        "approximately 38% of all tool-level records. A new bullet point has been added "
        "to Results §3.6 stating this explicitly, and a new paragraph in §3.7 corrects "
        "the record directly. Second, on the Hamilton Depression Rating Scale and "
        "SAFE-T: we agree with the reviewer that the Hamilton scale is not "
        "suicide-specific, and this is reflected precisely in its classification, "
        "introduced in response to Reviewer 4 Comment 5, as a “comprehensive "
        "mental-health tool with a suicide-related item” rather than a "
        "suicide-specific instrument - it was retained because its item 3 assesses "
        "suicidal ideation directly, not miscategorized as suicide-specific. We "
        "respectfully disagree, however, that SAFE-T is not suicide-specific: it is "
        "explicitly a suicide assessment and triage protocol by name and design, and is "
        "classified as suicide-specific in the revised manuscript. Its exclusion from the "
        "five LOINC candidates reflects a separate, narrower finding - that its "
        "clinical-decision-support format does not align with LOINC's model of discrete, "
        "codifiable data elements - not a judgment about its suicide-specificity. A "
        "new paragraph in §3.7 makes this distinction explicit so it cannot be read as an "
        "implicit concession of search bias."
    )

    add_comment(
        doc, "Comment 11",
        "This is back to the intro...the authors mention use of LLMs to detect suicide, "
        "but then take a left turn into using LLMs to identify suicide screening measures "
        "along a range of criteria from the literature. Please provide transition "
        "explaining how the methodology aligns with the discussion of using LLMs to "
        "detect suicide risks with clients..."
    )
    add_response(
        doc,
        "A new transition paragraph has been added to the Introduction, immediately "
        "following the existing discussion of LLM-based suicide-risk detection. It "
        "explains the connection explicitly: AI-based suicide-risk detection from "
        "clinical text depends on the outcome of a screening/assessment tool already "
        "existing in the record as structured, computable data; when a tool's result is "
        "documented only as unstructured free text or under a non-standardized local "
        "label, no detection model, LLM-based or otherwise, has a reliable structured "
        "signal to act on at scale. This study's tool-cataloging and terminology-gap "
        "identification work is framed explicitly as a necessary enabling step for, "
        "rather than a departure from, the AI-based detection research discussed earlier "
        "in the Introduction."
    )

    para = doc.add_paragraph()
    run = para.add_run(
        "Finally, the reviewer's closing comments note that the methodology's promise is "
        "undercut by insufficient operationalization of terms and methodological "
        "description, and that the manuscript does not adequately describe how AI "
        "broadly (including machine learning and LLMs) may be useful for suicide "
        "detection and screening. "
    )
    run.italic = True
    para.add_run(
        "The new Introduction transition paragraph described above (Comment 11) "
        "substantially expands this discussion, situating the present study within the "
        "broader ML/LLM suicide-risk-detection literature already cited in the manuscript "
        "(refs. 22, 29, 30) rather than treating that literature as a brief aside. "
        "Combined with the term definitions added in response to Comments 2, 3, and 5 "
        "and the methodological detail added in response to Comments 6 through 9 above, "
        "we believe the operationalization and methodological-description concerns "
        "raised throughout this review have been substantively addressed."
    )
    doc.add_paragraph()

    # ================= REVIEWER 2 =================
    add_heading(doc, "Reviewer 2", level=1)
    doc.add_paragraph(
        "Reviewer 2 withdrew from the review process and did not submit a report. No "
        "response is required."
    )
    doc.add_paragraph()

    # ================= REVIEWER 3 =================
    add_heading(doc, "Reviewer 3", level=1)
    doc.add_paragraph(
        "We thank the reviewer for the positive assessment and endorsement of the "
        "manuscript."
    )
    add_comment(
        doc, "Comment",
        "The authors make a good statistical analysis, but the English language must be "
        "improved... The study analyzed research articles rather than real world clinical "
        "notes or EHR data."
    )
    add_response(
        doc,
        "The manuscript has undergone a full language-editing pass to improve clarity and "
        "correct grammatical issues throughout. On the observation regarding real-world "
        "clinical notes and EHR data: this is an existing, explicitly stated limitation "
        "(§4.2), which we have retained and reinforced given the related concerns raised "
        "by Reviewer 4 (Comment 1) about tempering claims of direct clinical impact. We have "
        "also substantially strengthened the statistical validation underlying the "
        "manuscript's accuracy claims, expanding the gold-standard comparison from 4 to 20 "
        "articles (see response to Reviewer 4, Comment 4), which we believe further "
        "reinforces confidence in the reported results despite the inherent scope limitation "
        "of a literature-based (rather than clinical-data-based) study design."
    )

    # ================= REVIEWER 4 =================
    add_heading(doc, "Reviewer 4", level=1)
    doc.add_paragraph(
        "We thank the reviewer for a thorough and constructive review that substantially "
        "improved the manuscript. We address each point below."
    )

    add_comment(
        doc, "Comment 1",
        "The scope and contribution should be clarified and the conclusions better aligned "
        "with what was actually studied... Claims about improving suicide-safe care and "
        "standardizing clinical practice should therefore be substantially tempered."
    )
    add_response(
        doc,
        "We agree, and have revised the manuscript title along with the Abstract, "
        "Discussion, and Conclusion to consistently frame this work as a proof-of-concept "
        "demonstration of AI-assisted literature mapping and terminology-gap "
        "identification, rather than a demonstrated improvement in clinical suicide-safe "
        "care. The title has been changed from “Improving Suicide-Safe Care through "
        "Gen AI-Driven Recommendations for Tool Standardization in LOINC” to "
        "“Toward Suicide-Safe Care: Gen AI-Driven Mapping of Suicide Screening and "
        "Assessment Tools for LOINC Standardization” - replacing the direct causal "
        "claim (“Improving”) with the standard academic convention for signaling "
        "contribution toward a goal, and replacing “Recommendations for Tool "
        "Standardization” with language that accurately reflects what the study did "
        "(cataloguing tools and identifying standardization candidates) without implying "
        "standardization itself was achieved. Statements throughout the body implying "
        "direct clinical benefit (e.g., language suggesting the pipeline itself "
        "“strengthens the foundation for timely, data-driven intervention”) have "
        "likewise been revised to explicitly frame such outcomes as a downstream potential "
        "contingent on future terminology adoption and clinical validation, not a result "
        "established by this study."
    )

    add_comment(
        doc, "Comment 2",
        "The literature search and article selection raise concerns about "
        "representativeness. The search was limited to PubMed Central, open-access papers "
        "from the previous five years, and the terms “suicide screening tool” and "
        "“suicide assessment tool.” The authors should justify these restrictions..."
    )
    add_response(
        doc,
        "Methods §2.1 has been expanded to explicitly justify each restriction: PubMed "
        "Central was selected because it is the primary open-access aggregator guaranteeing "
        "full-text availability, which the automated extraction pipeline requires (a "
        "paywalled abstract-only record cannot be processed); the five-year window was "
        "chosen to capture the contemporary tool-usage landscape rather than instruments "
        "that may have fallen out of active use; and the two search terms were chosen to "
        "mirror the terminology used in the pipeline's own KPI-extraction queries (Table 1), "
        "ensuring consistency between article identification and downstream extraction. We "
        "have added an explicit new Limitations statement acknowledging that this search "
        "strategy may under-represent articles that study a named instrument without using "
        "the words “screening” or “assessment” in the discoverable text, or "
        "that use synonymous terminology (e.g., “suicide risk evaluation”)."
    )

    add_comment(
        doc, "Comment 3",
        "The exclusion of 129 articles because their PDFs exceeded 1.5 MB is a substantial "
        "concern. This represents approximately 17% of the nonduplicate retrieved articles "
        "and is a technical rather than scientific exclusion. The authors should ideally "
        "process these papers using an alternative approach..."
    )
    add_response(
        doc,
        "We agree, and have directly acted on this recommendation. The 129 originally "
        "excluded articles were revisited: file size was reduced below the automated-"
        "processing threshold using PDF compression, and articles that could not be "
        "recovered through compression were replaced with equivalent, newly-identified "
        "open-access articles meeting the identical inclusion criteria. All 129 were "
        "successfully processed as a second batch. The corpus now totals 769 articles "
        "(up from 639) and 1,012 tool-level CSVs (up from 841); Figure 1, Table 4, and all "
        "corpus-level statistics throughout the manuscript have been updated accordingly. "
        "We believe this substantively resolves the concern, converting what was a purely "
        "technical exclusion into full corpus coverage."
    )

    add_comment(
        doc, "Comment 4",
        "The validation of the AI pipeline is insufficient for the strength of the accuracy "
        "claims. Only four articles were manually annotated, and the manuscript does not "
        "sufficiently describe how these papers were selected, who produced the human "
        "reference answers, their expertise, or whether more than one annotator was "
        "involved..."
    )
    add_response(
        doc,
        "The gold-standard validation set has been expanded from 4 to 20 manually-annotated "
        "reference articles (a five-fold increase, from 0.5% to 2.60% of the corpus), "
        "yielding 495 article–KPI extraction rows and 1,485 individual RAGAS "
        "measurements. This exceeds the approximately 1,028 measurements indicated by a "
        "standard 95%-confidence, ±3%-margin sample-size calculation against the "
        "corpus's 27,684 article–KPI population (new §2.7 and Supplementary "
        "Methods). Updated performance metrics across this larger, more representative "
        "sample are: mean faithfulness 0.699 (0.857 restricted to descriptive/prose fields "
        "- two atomic yes/no and numeric fields are excluded from this restricted "
        "figure because RAGAS's faithfulness metric scores non-decomposable answers 0 by "
        "construction, independent of correctness), answer relevancy 0.729, and context "
        "recall 0.817 (Table 2, Results §3.3). We note transparently that this larger "
        "sample yields a somewhat less uniformly high performance picture than the original "
        "4-article validation (45.9% vs. 59.4% perfect scores across all scored "
        "measurements), which we consider a more credible and representative estimate of "
        "true pipeline performance, and have added this comparison explicitly to the text "
        "as a demonstration of why the expanded sample was necessary. Regarding annotator "
        "background: the 20 gold-standard reference articles were annotated by two domain "
        "experts - a physician and a terminology/standardization expert - working "
        "independently of the pipeline's own output; this is now stated explicitly in "
        "Methods §2.7. We did not assess formal inter-annotator agreement between the two "
        "annotators, and have added this explicitly as a stated limitation (§4.2) per the "
        "reviewer's specific request, rather than leaving the omission unaddressed."
    )

    add_comment(
        doc, "Comment 5",
        "The definition and derivation of the 239 “unique suicide-related tools” "
        "need much greater transparency. It is unclear how names, abbreviations, versions, "
        "translations, adaptations, and subscales were normalized into unique instruments... "
        "provide the complete list of 239 tools and their frequencies in the supplementary "
        "material."
    )
    add_response(
        doc,
        "We have added a fully documented, two-stage tool-name resolution process to "
        "Methods (new §2.10). First, an automated normalization pass grouped raw "
        "tool-name extractions by shared abbreviations - merging spelling/hyphenation "
        "variants and merging different abbreviations known to denote the same instrument "
        "(e.g., SSI, BSS, BSI, and BSSI all referring to the Beck Scale for "
        "Suicide/Suicidal Ideation) - while explicitly keeping distinct any pair differing "
        "by a version, edition, population, or administration-mode qualifier (e.g., PHQ-9 "
        "vs. PHQ-2 vs. PHQ-A; BDI vs. BDI-II; MMPI-2 vs. MMPI-3 vs. MMPI-A). This automated "
        "pass reduced 700 raw tool-name mentions to 319 candidate canonical tools. Second, "
        "this candidate list was reviewed in full by two domain experts (a physician and a "
        "terminology/standardization expert), who classified each candidate as "
        "suicide-specific, a comprehensive mental-health tool containing a suicide-related "
        "item or subscale, not actually suicide-related, or unclear/unverifiable - directly "
        "resolving the reviewer's request to distinguish dedicated suicide instruments from "
        "broader measures. This yields a final, human-verified total of 216 unique "
        "suicide-related tools: 130 suicide-specific instruments and 86 comprehensive "
        "mental-health tools with a suicide-related item or subscale. As a concrete "
        "illustration of why superficially similar names were retained as distinct entries "
        "rather than merged: ten Columbia-Suicide Severity Rating Scale (C-SSRS) variants "
        "remain separate because they are materially different in clinical use (e.g., the "
        "ultra-brief 5–6 item Screen version used in emergency-department triage and "
        "primary care, versus the Lifetime version assessing full history, versus "
        "electronic and self-rated versions differing in administration mode); similarly, "
        "the Patient Health Questionnaire-9, PHQ-2, and PHQ-A remain distinct as they serve "
        "different clinical purposes (full clinical screening and severity tracking, "
        "ultra-brief initial screening, and adolescent-specific screening, respectively). "
        "The complete list of all 216 tools, with frequency and "
        "suicide-specific/comprehensive classification, is provided as Supplementary "
        "Table S1."
    )

    add_comment(
        doc, "Comment 6",
        "The inference that 239 tools demonstrate a lack of standardization is currently "
        "too strong... The discussion should therefore distinguish heterogeneity in the "
        "research literature from lack of standardization in clinical practice."
    )
    add_response(
        doc,
        "We revised the relevant Discussion language throughout to explicitly distinguish "
        "“heterogeneity observed in the open-access research literature we sampled” "
        "from “lack of standardization in clinical practice,” removing language that "
        "directly equated the two. We also now present the full frequency distribution "
        "(Supplementary Table S1) and describe it explicitly in the text: the "
        "Columbia-Suicide Severity Rating Scale alone accounts for approximately 25% of "
        "all tool-level records (156 of 626 among the 216 catalogued tools), and together "
        "with the next four most frequent instruments (Ask Suicide-Screening Questions, "
        "PHQ-9, the Beck Scale for Suicide Ideation, and the Suicidal Behaviors "
        "Questionnaire-Revised) accounts for nearly half of all records, while the "
        "majority of the 216 tools appear only once or twice - a long-tailed distribution "
        "we now name and describe explicitly, rather than reporting only the aggregate "
        "count, to avoid overstating fragmentation across the field as a whole."
    )

    add_comment(
        doc, "Comment 7",
        "The selection of the five proposed LOINC candidates requires a more transparent "
        "and cautious rationale... The recommended instruments would be better described as "
        "candidates for further evaluation for LOINC representation rather than "
        "“high-value” or preferred suicide screening tools. LOINC itself should also "
        "be explained more clearly in the Introduction."
    )
    add_response(
        doc,
        "The LOINC-comparison section (§2.9/§3.7) now describes explicit, "
        "operationalized criteria for each of the five stated dimensions (minimum literature "
        "frequency of three, open or public accessibility, a structured question-and-answer "
        "format, absence from current LOINC coverage, and structural suitability for "
        "codification - illustrated concretely through our worked SAFE-T exclusion "
        "example, §3.7). We have also softened language describing the five identified "
        "tools throughout the manuscript from “high-value” and “strong "
        "candidates” to “candidates warranting further evaluation for LOINC "
        "representation,” consistent with the reviewer's recommendation not to conflate "
        "literature frequency with clinical value or psychometric quality. LOINC is now "
        "defined in the Introduction at first mention (see response to Reviewer 1, "
        "Comment 5)."
    )

    doc.add_page_break()
    closing = doc.add_paragraph()
    closing.add_run(
        "We are grateful to all reviewers for their time and detailed feedback, which has "
        "substantially strengthened the accuracy, transparency, and scope of this "
        "manuscript. We believe the revised manuscript directly and substantively addresses "
        "every point raised."
    )

    doc.save(OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    build()
