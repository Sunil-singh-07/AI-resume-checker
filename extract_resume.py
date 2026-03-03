"""
Resume PDF Text Extractor
Supports digital PDFs (via pdfplumber) and scanned PDFs (via OCR with pytesseract).

Install dependencies:
    pip install pdfplumber pytesseract pdf2image
    # Also install Tesseract OCR engine:
    # macOS:   brew install tesseract
    # Ubuntu:  sudo apt install tesseract-ocr
    # Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import sys
import pdfplumber


def extract_text_digital(pdf_path: str) -> str:
    """Extract text from a digital (non-scanned) PDF using pdfplumber."""
    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"[INFO] Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text.append(f"--- Page {i} ---\n{text.strip()}")
            else:
                print(f"[WARN] No text found on page {i} (may be scanned/image-based)")

    return "\n\n".join(full_text)


def extract_text_ocr(pdf_path: str) -> str:
    """Extract text from a scanned PDF using OCR (pytesseract + pdf2image)."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "OCR dependencies missing. Install with:\n"
            "  pip install pytesseract pdf2image\n"
            "And install Tesseract: https://github.com/tesseract-ocr/tesseract"
        )

    print("[INFO] Running OCR on scanned PDF...")
    images = convert_from_path(pdf_path, dpi=300)
    full_text = []

    for i, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)
        full_text.append(f"--- Page {i} ---\n{text.strip()}")
        print(f"[INFO] OCR complete for page {i}/{len(images)}")

    return "\n\n".join(full_text)


def extract_resume_text(pdf_path: str, force_ocr: bool = False) -> str:
    """
    Main extractor. Tries digital extraction first; falls back to OCR if needed.

    Args:
        pdf_path:  Path to the resume PDF file.
        force_ocr: Set True to skip digital extraction and always use OCR.

    Returns:
        Extracted text as a string.
    """
    if not force_ocr:
        text = extract_text_digital(pdf_path)
        if text.strip():
            return text
        print("[INFO] Digital extraction yielded no text. Falling back to OCR...")

    return extract_text_ocr(pdf_path)


# ── Structured section parsing (optional helper) ──────────────────────────────

COMMON_SECTIONS = [
    "summary", "objective", "experience", "education",
    "skills", "projects", "certifications", "awards",
    "languages", "interests", "references",
]

def parse_sections(text: str) -> dict[str, str]:
    """
    Heuristically split extracted text into resume sections.
    Returns a dict mapping section name → content.
    """
    import re
    sections: dict[str, str] = {}
    current_section = "header"
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip().lower()
        matched = next(
            (s for s in COMMON_SECTIONS if re.match(rf"^{s}[:\s]*$", stripped)),
            None,
        )
        if matched:
            sections[current_section] = "\n".join(current_lines).strip()
            current_section = matched
            current_lines = []
        else:
            current_lines.append(line)

    sections[current_section] = "\n".join(current_lines).strip()
    return {k: v for k, v in sections.items() if v}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_resume.py <path_to_resume.pdf> [--ocr] [--sections]")
        sys.exit(1)

    path = sys.argv[1]
    use_ocr = "--ocr" in sys.argv
    show_sections = "--sections" in sys.argv

    print(f"[INFO] Processing: {path}\n")
    text = extract_resume_text(path, force_ocr=use_ocr)

    if not text.strip():
        print("[ERROR] No text could be extracted from the PDF.")
        sys.exit(1)

    if show_sections:
        sections = parse_sections(text)
        for section, content in sections.items():
            print(f"\n{'='*40}")
            print(f"  SECTION: {section.upper()}")
            print(f"{'='*40}")
            print(content)
    else:
        print(text)

    # Optionally save output
    output_path = path.replace(".pdf", "_extracted.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[INFO] Text saved to: {output_path}")
