import io
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from .chunking import chunk_text
from .config import settings

ENCODINGS = ["utf-8", "utf-8-sig", "cp932", "latin-1"]  # cp932 covers Japanese CSVs


def _read_pdf(data: bytes):
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            yield text, f"page {i}"


def _decode(data: bytes) -> str:
    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _read_csv(data: bytes) -> pd.DataFrame:
    for enc in ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the CSV file.")


def _table_sections(df: pd.DataFrame, label: str, rows_per: int = 20):
    df = df.head(settings.max_table_rows)
    yield f"Table '{label}' has {len(df)} rows. Columns: {', '.join(map(str, df.columns))}", "schema"
    for start in range(0, len(df), rows_per):
        block = df.iloc[start:start + rows_per]
        lines = [" | ".join(f"{c}: {v}" for c, v in row.items()) for _, row in block.iterrows()]
        yield "\n".join(lines), f"rows {start + 1}-{start + len(block)}"


def process_upload(filename: str, data: bytes):
    """Return (chunks, tables). chunks feed the vector DB; tables feed the pandas tool."""
    ext = Path(filename).suffix.lower()
    sections, tables = [], {}

    if ext == ".pdf":
        sections = list(_read_pdf(data))
    elif ext == ".txt":
        sections = [(_decode(data), "text")]
    elif ext == ".csv":
        df = _read_csv(data)
        tables[filename] = df
        sections = list(_table_sections(df, filename))
    elif ext == ".xlsx":
        for sheet, df in pd.read_excel(io.BytesIO(data), sheet_name=None).items():
            key = f"{filename}[{sheet}]"
            tables[key] = df
            sections += list(_table_sections(df, key))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks = []
    for text, location in sections:
        for piece in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append({
                "id": f"{filename}:{len(chunks)}",
                "text": piece,
                "metadata": {"source": filename, "location": location},
            })
    if not chunks:
        raise ValueError(f"No extractable text in '{filename}' (scanned PDFs need OCR, which isn't supported).")
    return chunks, tables
