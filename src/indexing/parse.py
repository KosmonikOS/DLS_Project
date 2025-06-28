import fitz
import re
import logging
import asyncio
import httpx

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.indexing.settings import settings

logger = logging.getLogger(__name__)


def process_text(text):
    match = re.search(r"\\babstract\\b", text, re.IGNORECASE)
    if match:
        # Get the text *after* the word "abstract"
        text = text[match.end() :]

    # Look for common footer section headers like "References", "Bibliography", "Acknowledgements".
    footer_match = re.search(
        r"\n\s*(references|bibliography|acknowledgements)\b", text, re.IGNORECASE
    )
    if footer_match:
        text = text[: footer_match.start()]

    text = re.sub(r"(\\w+)-\\s*\\n\\s*(\\w+)", r"\\1\\2", text)

    # Remove emails and URLs (any trailing punctuation will be consumed as well)
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", " ", text)
    text = re.sub(r"https?://[^\s]+|www\.[^\s]+", " ", text)

    # Remove any token that contains a digit (e.g. '13th', 'C.3', 'section4')
    text = re.sub(r"\b[^\s]*\d[^\s]*\b", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep only latin letters and whitespace
    text = re.sub(r"[^a-zA-Z\s]+", " ", text)

    # Lower-case first
    text = text.lower()

    # Remove patterns like "smith et al" (предыдущее слово + et al)
    text = re.sub(r"\b\w+\s+et\s+al\b", " ", text)

    # Сжать пробелы
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_pdf(data: bytes | bytearray) -> str:
    """Extract and clean text from a PDF.

    The PDF can be supplied either as raw *bytes* (recommended) or as a local
    filesystem *path* (``str`` | ``Path``). The latter is kept for backwards
    compatibility.
    """
    try:
        if not data.lstrip().startswith(b"%PDF"):
            raise ValueError("Invalid PDF header – does not start with %PDF")

        with fitz.open(stream=data, filetype="pdf") as doc:
            texts: list[str] = []
            for page_number in range(doc.page_count):
                try:
                    page = doc.load_page(page_number)
                    texts.append(page.get_text("text"))
                except Exception as page_exc:
                    logger.warning(
                        "Skipping page %d due to parsing error: %s",
                        page_number,
                        page_exc,
                    )
            raw = "".join(texts)
    except Exception as e:
        logger.error("Cannot parse PDF: %s", e)
        raise

    return process_text(raw)


@retry(
    stop=stop_after_attempt(getattr(settings, "download_retries", 3)),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
async def _get_with_retry(client: httpx.AsyncClient, url: str) -> bytes:
    """Download *url* once, raising on any HTTP/client error."""
    resp = await client.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp.content


async def fetch_and_parse(
    urls: list[str],
    *,
    client: httpx.AsyncClient | None = None,
) -> list[str | None]:
    """Download multiple PDF URLs concurrently and return cleaned texts.

    Args:
        urls: HTTP(S) links pointing to PDF files.

    Returns:
        List of cleaned document texts (``None`` for failed downloads/parses),
        preserving the input order.
    """

    results: list[str | None] = [None] * len(urls)
    semaphore = asyncio.Semaphore(settings.acl_concurrency)

    async def _worker(i: int, url: str) -> None:
        async with semaphore:
            try:
                pdf_bytes = await _get_with_retry(client, url)
            except Exception as exc:
                logger.error("Download failed for %s: %s", url, exc)
                return

            try:
                text = parse_pdf(pdf_bytes)
            except Exception as exc:
                logger.error("Parsing failed for %s: %s", url, exc)
                return

            results[i] = text

    await asyncio.gather(*(_worker(i, u) for i, u in enumerate(urls)))
    return results
