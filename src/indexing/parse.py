import fitz
import time
import os
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def process_text(text):
    
    match = re.search(r'\\babstract\\b', text, re.IGNORECASE)
    if match:
        # Get the text *after* the word "abstract"
        text = text[match.end():]

    # Look for common footer section headers like "References", "Bibliography", "Acknowledgements".
    footer_match = re.search(r"\n\s*(references|bibliography|acknowledgements)\b", text, re.IGNORECASE)
    if footer_match:
        text = text[:footer_match.start()]

    text = re.sub(r'(\\w+)-\\s*\\n\\s*(\\w+)', r'\\1\\2', text)

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

def parse_pdf(path: str | Path, *, save_md: bool = False) -> tuple[str, float]:
    try:
        t0 = time.perf_counter()
        with fitz.open(path) as doc:
            raw = "".join(page.get_text() for page in doc)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        logger.error("Cannot parse %s: %s", path, e)
        raise

    cleaned = process_text(raw)
    # Optional debug output: сохранить файл .md рядом с PDF
    if save_md:
        base_name = Path(path).stem
        md_path = Path(path).with_suffix("").with_name(f"{base_name}_pymupdf.md")
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {base_name} (PyMuPDF)\n\n")
                f.write(f"**Время парсинга:** {elapsed:.3f}s\n\n")
                f.write(cleaned)
        except Exception as err:
            logger.warning("Не удалось записать MD файл %s: %s", md_path, err)
    return cleaned, elapsed

if __name__ == "__main__":
    import argparse, textwrap, sys

    parser = argparse.ArgumentParser(
        description="Extract and clean text from PDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", nargs="+", help="Путь(и) к PDF файлам")
    parser.add_argument(
        "--save-md",
        action="store_true",
    )

    args = parser.parse_args()

    for pdf_path in args.pdf:
        try:
            cleaned, elapsed = parse_pdf(pdf_path, save_md=args.save_md)
            print("-" * 80)
            print(f"{pdf_path} — {len(cleaned.split())} слов, обработано за {elapsed:.2f} c")
            snippet = cleaned[:500] + ("…" if len(cleaned) > 500 else "")
            print(snippet)
        except Exception as exc:
            print(f"[ERROR] {pdf_path}: {exc}", file=sys.stderr)
            continue