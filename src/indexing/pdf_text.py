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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import pypdfium2 as pdfium

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(4),  # initial try + 3 retries
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError),
    reraise=True,
)
async def _download(session: aiohttp.ClientSession, url: str) -> bytes:
    """Perform a single HTTP GET with retry handled by *tenacity*."""
    async with session.get(url, timeout=30) as resp:
        if resp.status != 200:
            # Treat 5xx as transient, raise ClientError to trigger retry.
            if resp.status >= 500:
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message="server error",
                )
            # 4xx considered permanent
            raise Exception(f"HTTP {resp.status}")
        return await resp.read()


async def fetch_pdf(
    session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore
) -> bytes | None:
    """Download *url* obeying *semaphore*, with automatic retries."""
    async with semaphore:
        try:
            return await _download(session, url)
        except Exception:
            logger.warning("Failed to fetch %s skipping", url)
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
