import pytest
from certsplit.extraction import CERT_LABEL_RE, HEAT_LABEL_RE, normalize_token
from certsplit.extraction import extract_page_signature


def test_cert_patterns_variants():
    text = "Inspection certificate No: 80209171-01-280\nZeugnis-Nr. 0000369904/1\n"
    vals = [normalize_token(m.group(1)) for m in CERT_LABEL_RE.finditer(text)]
    assert "80209171-01-280" in vals
    assert "0000369904/1" in vals


def test_heat_patterns_variants():
    text = "Heat No. H12345\nSchmelze: ABC987\nColada: 445566\n"
    vals = [normalize_token(m.group(1)) for m in HEAT_LABEL_RE.finditer(text)]
    assert {"H12345", "ABC987", "445566"}.issubset(set(vals))


def test_extract_page_signature_detects_generic_page_fraction_and_zones(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "zones.pdf"
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((40, 40), "Certificate Number: ZX-778899")
    page.insert_text((40, 770), "1/3")
    doc.save(pdf)
    doc.close()

    src = fitz.open(pdf)
    try:
        extraction = extract_page_signature(src.load_page(0), "GENCA")
    finally:
        src.close()

    assert extraction.signature is not None
    assert extraction.signature.cert_id == "ZX-778899"
    assert "page_fraction" in extraction.signature.flags


def test_extract_page_signature_finds_cert_and_heat_in_context_line(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "context.pdf"
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((40, 80), "Document reference CERT-2024/998877 and material data")
    page.insert_text((40, 120), "Datos de colada HX778899 para trazabilidad")
    doc.save(pdf)
    doc.close()

    src = fitz.open(pdf)
    try:
        extraction = extract_page_signature(src.load_page(0), "GENCA")
    finally:
        src.close()

    assert extraction.signature is not None
    assert extraction.signature.cert_id == "CERT-2024/998877"
    assert "HX778899" in extraction.signature.heats
