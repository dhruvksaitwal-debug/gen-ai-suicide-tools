import os
import shutil
import pymupdf
import pandas as pd
from constants import PDF_STEM_MAXLEN

class PDFExtractor:
    """
    Extracts text, tables, and images from a PDF and uses an LLM client
    to summarize tables and describe images.
    """

    def __init__(self, llm_client, image_output_folder="extracted_images"):
        self.llm_client = llm_client
        self.image_output_folder = image_output_folder

    def _summarize_table(self, df: pd.DataFrame) -> str:
        table_markdown = df.to_markdown(index=False)
        prompt = (
            "Analyze the following data table provided in Markdown format. "
            "Perform a detailed analysis of all the contents within the table. "
            "Make sure that your final output analysis fits in 400 words.\n\n"
            f"Table:\n{table_markdown}"
        )
        return self.llm_client.chat_completion([
            {"role": "system", "content": "You are a data analyst who analyzes tables thoroughly and accurately."},
            {"role": "user", "content": prompt}
        ], temperature=0, max_tokens=600)

    def extract(self, pdf_path: str) -> list[str]:
        """
        Extracts text, tables, and images from the given PDF file.
        Returns a list of strings containing extracted content and LLM summaries.
        """
        # Create a unique subfolder for this PDF
        pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0][:PDF_STEM_MAXLEN].lower().replace(" ", "_")
        pdf_image_folder = os.path.join(self.image_output_folder, pdf_stem)

        # Clean the folder if it already exists
        if os.path.exists(pdf_image_folder):
            shutil.rmtree(pdf_image_folder)
        os.makedirs(pdf_image_folder)

        doc = pymupdf.open(pdf_path)
        article_contents = []

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # --- Text ---
            page_text = page.get_text("text")
            article_contents.append(f"\n--- Page {page_num + 1} Text ---\n{page_text}")

            # --- Tables ---
            tables = page.find_tables()
            if tables.tables:
                for i, table in enumerate(tables):
                    df = table.to_pandas()
                    summary = self._summarize_table(df)
                    article_contents.append(f"\n--- Page {page_num + 1} Table {i+1} Summary ---\n{summary}")

            # --- Images ---
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    image_info = doc.extract_image(xref)
                    image_bytes = image_info["image"]
                    image_ext = image_info.get("ext") or "png"
                    image_filename = os.path.join(
                        pdf_image_folder,
                        f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    )
                    with open(image_filename, "wb") as f:
                        f.write(image_bytes)

                    # Now delegate description to LLMClient
                    description = self.llm_client.describe_image(image_bytes)
                    article_contents.append(
                        f"\n--- Page {page_num + 1} Image {img_index+1} Description ---\n{description}"
                    )
                except Exception as e:
                    article_contents.append(
                        f"\n--- Page {page_num + 1} Image {img_index+1} Error ---\nCould not process image: {e}"
                    )

        doc.close()
        return article_contents