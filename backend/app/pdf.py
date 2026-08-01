import re
from pathlib import Path
import fitz

def extract_pdf(path: Path, document_id: str) -> dict:
    pages, chunks = [], []
    with fitz.open(path) as doc:
        for page_number, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            pages.append({"page": page_number, "text": text})
            for block in re.split(r"\n\s*\n|(?<=다\.)\s+", text):
                block = re.sub(r"\s+", " ", block).strip()
                if not block:
                    continue
                paragraph = re.search(r"\[(\d{4})\]", block)
                chunks.append({"document_id": document_id, "page": page_number,
                               "paragraph": paragraph.group(1) if paragraph else None,
                               "text": block})
    full = " ".join(p["text"] for p in pages)
    return {"pages": pages, "chunks": chunks, "text": full, "ocr_required": len(full) < 40}

def classify(text: str) -> str:
    normalized = text.lower()
    if re.search(r"\[(\d{4})\]", text) or "claims" in normalized or "patent" in normalized:
        return "patent"
    if any(term in normalized for term in ("abstract", "introduction", "method", "references", "doi")):
        return "paper"
    return "technical"

