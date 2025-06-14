"""docling_parallel.py
Utility to convert a list of PDF URLs/paths into Markdown using *Docling*
in a fixed-size process pool.
"""

from __future__ import annotations

from multiprocessing import get_context
from multiprocessing.pool import Pool

import os
import warnings

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

# Per-process variable (Each process has its own copy that is reused between calls)
_CONVERTER: DocumentConverter | None = None

# Parent-process variable (Only parent process has it and reuse for all batches)
_POOL: Pool | None = None


def _init_worker(num_threads: int) -> None:
    """Instantiate a heavy :class:DocumentConverter in the worker process.

    Args:
        num_threads: Maximum number of intra-op threads the converter may use.
    """

    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    # Disable redundant pin_memory warnings on macOS (MPS backend)
    warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

    opts = PdfPipelineOptions()
    opts.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.AUTO,
        num_threads=num_threads,
    )
    opts.do_ocr = False
    opts.do_table_structure = False
    global _CONVERTER
    _CONVERTER = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _convert(source: str) -> str | None:
    """Convert one PDF to Markdown.

    Args:
        source: Local path or HTTP(S) URL pointing to a PDF.

    Returns:
        The extracted Markdown, or None if conversion fails.
    """
    try:
        res = _CONVERTER.convert(source)
        return res.document.export_to_markdown()
    except Exception:
        return None


def _create_pool(workers: int, num_threads: int) -> Pool:
    """Create a multiprocessing.Pool that reuses _init_worker.

    Args:
        workers: Number of worker processes.
        num_threads: Intra-op threads per worker process.

    Returns:
        A process Pool ready to convert PDFs in parallel.
    """
    ctx = get_context("spawn")
    return ctx.Pool(
        processes=workers, initializer=_init_worker, initargs=(num_threads,)
    )


def close_pool() -> None:
    """Terminate the cached process pool, if it exists."""
    global _POOL
    if _POOL is None:
        return

    _POOL.close()
    _POOL.join()

    _POOL = None


def convert_in_parallel(
    sources: list[str], workers: int = 4, num_threads: int = 1
) -> list[str | None]:
    """Convert multiple PDFs to Markdown in parallel.

    Args:
        sources: List of local paths or URLs to PDFs. An empty list returns [].
        workers: Number of worker processes to spawn (defaults to 4).
        num_threads: Intra-op threads available inside each worker (defaults to 1).

    Returns:
        A list of Markdown strings (None for items that failed to convert),
        ordered identically to sources.
    """
    if not sources:
        return []

    global _POOL

    if _POOL is None:
        _POOL = _create_pool(workers, num_threads)

    return _POOL.map(_convert, sources)
