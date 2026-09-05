import concurrent.futures
import logging
import os
import shutil
import threading
import time

import pymupdf
import pandas as pd
from constants import DEFAULT_MAX_WORKERS
from naming import safe_doc_id_stem

logger = logging.getLogger(__name__)


def _rmtree_with_retry(path: str, attempts: int = 5, delay: float = 0.5) -> None:
    """
    Remove a directory tree, retrying on transient PermissionError. Cloud-sync clients
    (OneDrive, Dropbox, etc.) commonly hold a brief lock on a just-created/modified
    directory while indexing it, which raises WinError 5 here if deleted too soon after.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            logger.debug("Retrying rmtree('%s') after PermissionError (attempt %d/%d)", path, attempt + 1, attempts)
            time.sleep(delay * (attempt + 1))

# PyMuPDF (MuPDF) is not safe to call concurrently from multiple threads, even across
# entirely separate Document objects opened in different threads — confirmed empirically:
# find_tables() reliably fails with "not a textpage of this page" / returns None when
# multiple PDFs are extracted at the same time (main.py processes several PDFs
# concurrently). This lock serializes only the PyMuPDF-touching portion of extraction,
# process-wide; the actual bottleneck (table/image LLM calls) still runs concurrently
# across PDFs, since those happen after the lock is released.
_pymupdf_lock = threading.Lock()


class PDFExtractor:
    """
    Extracts text, tables, and images from a PDF and uses an LLM client
    to summarize tables and describe images.
    """

    def __init__(self, llm_client, image_output_folder="extracted_images", max_workers: int = DEFAULT_MAX_WORKERS):
        self.llm_client = llm_client
        self.image_output_folder = image_output_folder
        self.max_workers = max_workers

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

    def _table_entry(self, df: pd.DataFrame, page_num: int, table_index: int) -> str:
        try:
            summary = self._summarize_table(df)
            return f"\n--- Page {page_num + 1} Table {table_index + 1} Summary ---\n{summary}"
        except Exception as e:
            logger.warning(
                "Failed to summarize table %d on page %d: %s",
                table_index + 1, page_num + 1, e, exc_info=True
            )
            return f"\n--- Page {page_num + 1} Table {table_index + 1} Error ---\nCould not process table: {e}"

    def _image_entry(self, image_bytes: bytes, page_num: int, img_index: int) -> str:
        try:
            description = self.llm_client.describe_image(image_bytes)
            return f"\n--- Page {page_num + 1} Image {img_index+1} Description ---\n{description}"
        except Exception as e:
            logger.warning(
                "Failed to describe image %d on page %d: %s",
                img_index + 1, page_num + 1, e, exc_info=True
            )
            return f"\n--- Page {page_num + 1} Image {img_index+1} Error ---\nCould not process image: {e}"

    def extract(self, pdf_path: str) -> list[str]:
        """
        Extracts text, tables, and images from the given PDF file.

        PyMuPDF access happens serially (it is not thread-safe), but table
        summarization and image description are independent LLM calls and are
        dispatched to a thread pool so they run concurrently. Returns a list of
        strings containing extracted content and LLM summaries, in page order.
        """
        # Create a unique subfolder for this PDF
        pdf_stem = safe_doc_id_stem(os.path.splitext(os.path.basename(pdf_path))[0])
        pdf_image_folder = os.path.join(self.image_output_folder, pdf_stem)

        # Clean the folder if it already exists
        if os.path.exists(pdf_image_folder):
            _rmtree_with_retry(pdf_image_folder)
        os.makedirs(pdf_image_folder)

        # Each entry is either a ready string (page text) or a Future resolving to one
        # (table/image LLM calls), kept in extraction order for the final assembly.
        entries: list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with _pymupdf_lock:
                doc = pymupdf.open(pdf_path)
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)

                    # --- Text ---
                    page_text = page.get_text("text")
                    entries.append(f"\n--- Page {page_num + 1} Text ---\n{page_text}")

                    # --- Tables ---
                    tables = page.find_tables()
                    if tables.tables:
                        for i, table in enumerate(tables):
                            df = table.to_pandas()
                            entries.append(executor.submit(self._table_entry, df, page_num, i))

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
                        except Exception as e:
                            logger.warning(
                                "Failed to extract image %d on page %d of %s: %s",
                                img_index + 1, page_num + 1, pdf_path, e, exc_info=True
                            )
                            entries.append(
                                f"\n--- Page {page_num + 1} Image {img_index+1} Error ---\nCould not process image: {e}"
                            )
                            continue

                        entries.append(executor.submit(self._image_entry, image_bytes, page_num, img_index))

                doc.close()  # all doc-dependent extraction is done; queued futures only touch bytes/dataframes

            # Futures resolve outside the lock: table/image LLM calls don't touch PyMuPDF,
            # so they run concurrently across PDFs even though extraction itself is serialized.
            article_contents = [
                entry.result() if isinstance(entry, concurrent.futures.Future) else entry
                for entry in entries
            ]

        return article_contents