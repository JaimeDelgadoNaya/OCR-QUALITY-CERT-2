import re
from collections import defaultdict
from typing import Iterable

from typing import Any

from .models import PageExtraction, PageSignature

CERT_LABEL_RE = re.compile(
    r"(?i)(?:certificado|certificate(?:\s*(?:no\.?|number))?|inspection\s*certificate(?:\s*no\.?)?|"
    r"zeugnis\s*[- ]?nr\.?|n[º°o.]|doc(?:ument)?\s*no\.?)\s*[:#\- ]*([A-Z0-9][A-Z0-9/_\-.]{4,30})"
)
CERT_FALLBACK_RE = re.compile(r"\b(?:\d{6,14}|\d{8,10}[-/]\d{1,3}[-/]\d{2,4}|\d{6,12}/\d{1,3})\b")

HEAT_LABEL_RE = re.compile(
    r"(?i)(?:colada|heat\s*(?:no\.?|code)?|schmelze|melt|chargen?|lot|lote)\s*[:#\- ]*([A-Z0-9][A-Z0-9/_\-.]{1,20})"
)
ATTACHMENT_HINT_RE = re.compile(r"(?i)\b(?:supplement|appendix|anexo|annex|ce\b|page\s*\d+\s*/\s*\d+)\b")
PAGE_FRACTION_RE = re.compile(r"(?i)\bpage\s*(\d+)\s*/\s*(\d+)\b")

NOISE_TOKEN_RE = re.compile(r"(?i)^(?:dn\d+|\d{1,2}mm|\d{4}-\d{2}-\d{2}|po\d+|order\d+)$")
GENERIC_PAGE_FRACTION_RE = re.compile(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b")
CERT_CONTEXT_RE = re.compile(r"(?i)\b(?:certificado|certificate|inspection\s*certificate|zeugnis|doc(?:ument)?)\b")
HEAT_CONTEXT_RE = re.compile(r"(?i)\b(?:colada|heat|schmelze|melt|chargen?|lot|lote)\b")
GENERIC_ID_RE = re.compile(r"\b[A-Z0-9][A-Z0-9/_\-.]{5,30}\b")

ZONE_SPECS: tuple[tuple[str, tuple[float, float, float, float], float], ...] = (
    ("header", (0.0, 0.0, 1.0, 0.28), 14.0),
    ("footer", (0.0, 0.72, 1.0, 1.0), 8.0),
    ("left", (0.0, 0.0, 0.35, 1.0), 8.0),
    ("right", (0.65, 0.0, 1.0, 1.0), 8.0),
    ("center", (0.2, 0.2, 0.8, 0.8), 4.0),
)


def normalize_token(token: str) -> str:
    return re.sub(r"\s+", "", token.strip()).upper()


def zone_text(page: Any, box: tuple[float, float, float, float]) -> str:
    r = page.rect
    x0, y0, x1, y1 = box
    import fitz
    clip = fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)
    return page.get_text("text", clip=clip) or ""


def _score_cert(token: str, text: str, idx: int) -> float:
    score = 30.0
    if re.search(r"\d", token):
        score += 30
    if re.search(r"[-/]", token):
        score += 10
    if 6 <= len(re.sub(r"\D", "", token)) <= 14:
        score += 20
    if idx < max(200, len(text) * 0.35):
        score += 8
    return score


def _score_heat(token: str, text: str, idx: int) -> float:
    score = 20.0
    if re.search(r"\d", token):
        score += 35
    if 2 <= len(token) <= 20:
        score += 10
    if idx < max(500, len(text) * 0.7):
        score += 8
    return score


def _filter_noise(values: Iterable[str]) -> list[str]:
    out = []
    for raw in values:
        token = normalize_token(raw)
        if not token:
            continue
        if NOISE_TOKEN_RE.match(token):
            continue
        if token.isdigit() and len(token) < 6:
            continue
        out.append(token)
    return sorted(set(out))


def _looks_like_cert_id(token: str) -> bool:
    digits = len(re.sub(r"\D", "", token))
    if digits < 6:
        return False
    if token.isdigit() and digits <= 14:
        return True
    return bool(re.search(r"[-/]", token)) or bool(re.search(r"[A-Z]", token))


def _expand_context_candidates(text: str, context_re: re.Pattern[str], score_fn, base_boost: float = 0.0) -> list[tuple[str, float]]:
    cands: list[tuple[str, float]] = []
    for line in (ln.strip() for ln in text.splitlines() if ln.strip()):
        if not context_re.search(line):
            continue
        for m in GENERIC_ID_RE.finditer(line.upper()):
            token = normalize_token(m.group(0))
            cands.append((token, score_fn(token, line, m.start()) + 6.0 + base_boost))
    return cands


def extract_page_signature(page: Any, vendor: str) -> PageExtraction:
    full = page.get_text("text") or ""

    zone_chunks: list[str] = []
    weighted_chunks: list[tuple[str, float]] = [(full, 0.0)]
    for _, box, boost in ZONE_SPECS:
        txt = zone_text(page, box)
        if txt.strip():
            zone_chunks.append(txt)
            weighted_chunks.append((txt, boost))
    combined = "\n".join(zone_chunks + [full])

    cert_cands: list[tuple[str, float]] = []
    for txt, boost in weighted_chunks:
        for m in CERT_LABEL_RE.finditer(txt):
            token = normalize_token(m.group(1))
            cert_cands.append((token, _score_cert(token, txt, m.start()) + boost))

    if not cert_cands:
        for txt, boost in weighted_chunks:
            for m in CERT_FALLBACK_RE.finditer(txt):
                token = normalize_token(m.group(0))
                cert_cands.append((token, _score_cert(token, txt, m.start()) - 15 + max(0.0, boost / 2)))

    cert_cands.extend(
        (tok, sc)
        for tok, sc in _expand_context_candidates(full, CERT_CONTEXT_RE, _score_cert)
        if _looks_like_cert_id(tok)
    )

    heat_cands: list[tuple[str, float]] = []
    for txt, boost in weighted_chunks:
        for m in HEAT_LABEL_RE.finditer(txt):
            token = normalize_token(m.group(1))
            heat_cands.append((token, _score_heat(token, txt, m.start()) + boost))

    heat_cands.extend(_expand_context_candidates(full, HEAT_CONTEXT_RE, _score_heat))

    cert_by_token = defaultdict(float)
    for tok, sc in cert_cands:
        cert_by_token[tok] = max(cert_by_token[tok], sc)
    heat_by_token = defaultdict(float)
    for tok, sc in heat_cands:
        heat_by_token[tok] = max(heat_by_token[tok], sc)

    cert_ranked = [(tok, sc) for tok, sc in cert_by_token.items() if _looks_like_cert_id(tok)]
    cert_id = max(cert_ranked, key=lambda i: i[1])[0] if cert_ranked else ""
    heats = tuple(_filter_noise(heat_by_token.keys()))

    is_cert = bool(cert_id) or bool(re.search(r"(?i)certificate|certificado|inspection", combined))
    is_att = bool(ATTACHMENT_HINT_RE.search(combined)) and not cert_id

    score = 0.0
    if cert_id:
        score += cert_by_token[cert_id]
    if heats:
        score += max(heat_by_token.values())
    if is_cert:
        score += 10

    flags = []
    if is_att:
        flags.append("attachment")
    if is_cert:
        flags.append("certificate")
    if PAGE_FRACTION_RE.search(combined) or any(int(a) <= int(b) <= 99 for a, b in GENERIC_PAGE_FRACTION_RE.findall(combined)):
        flags.append("page_fraction")

    sig = PageSignature(
        cert_id=cert_id,
        heats=tuple(sorted(heats)),
        vendor=(vendor or "").strip().upper() or "UNKNOWN",
        flags=tuple(flags),
        is_attachment=is_att,
        is_certificate_page=is_cert,
        score=round(score, 2),
    )
    return PageExtraction(page_index=page.number, full_text=combined, cert_candidates=sorted(cert_by_token.items()), heat_candidates=sorted(heat_by_token.items()), signature=sig)
