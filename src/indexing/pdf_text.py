"""pdf_text.py
Helpers for (1) downloading PDF files asynchronously and (2) extracting
plain text from PDF binary content using *pypdfium2*.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional

import aiohttp
import pypdfium2 as pdfium

logger = logging.getLogger(__name__)


async def fetch_pdf(
    session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore
) -> bytes | None:
    """Download *url* with *session* obeying *semaphore* concurrency limit."""
    async with semaphore:
        try:
            async with session.get(url, timeout=30) as resp:
                if resp.status != 200:
                    logger.warning("%s responded with HTTP %d", url, resp.status)
                    return None
                return await resp.read()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None


def pdf_to_text(data: bytes) -> str | None:
    """Return plain text for a PDF given its *data* bytes using pypdfium2."""

    pdf: Optional[pdfium.PdfDocument] = None
    try:
        pdf = pdfium.PdfDocument(io.BytesIO(data))
        texts: list[str] = []

        for page in pdf:
            textpage = page.get_textpage()
            texts.append(textpage.get_text_bounded())
            textpage.close()
            page.close()

        return "\n".join(texts)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("PDF parsing error: %s", exc)
        return None
    finally:
        if pdf is not None:
            try:
                pdf.close()
            except Exception:
                pass
