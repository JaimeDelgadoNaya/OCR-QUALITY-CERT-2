import re

from .models import PageExtraction, GroupSignature


PAGE_FRACTION_RE = re.compile(r"(?i)\bpage\s*(\d{1,2})\s*/\s*(\d{1,2})\b|\b(\d{1,2})\s*/\s*(\d{1,2})\b")


def _is_continuation(ext: PageExtraction) -> bool:
    sig = ext.signature
    if not sig:
        return False
    return sig.is_attachment or ("page_fraction" in sig.flags) or (sig.is_certificate_page and not sig.cert_id and not sig.heats)


def group_pages(extractions: list[PageExtraction], split_on_heat_change: bool = True) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_cert = ""
    current_heats: tuple[str, ...] = tuple()
    previous_fraction: tuple[int, int] | None = None

    def parse_page_fraction(ext: PageExtraction) -> tuple[int, int] | None:
        match = PAGE_FRACTION_RE.search(ext.full_text or "")
        if not match:
            return None
        n1, n2, n3, n4 = match.groups()
        a, b = (n1, n2) if n1 and n2 else (n3, n4)
        if not (a and b):
            return None
        return int(a), int(b)

    def flush():
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for ext in extractions:
        sig = ext.signature
        if not sig:
            continue
        page_fraction = parse_page_fraction(ext)

        if not current:
            current = [ext.page_index]
            current_cert = sig.cert_id
            current_heats = sig.heats
            previous_fraction = page_fraction
            continue

        cert_changed = bool(sig.cert_id and current_cert and sig.cert_id != current_cert)
        heat_changed = bool(split_on_heat_change and sig.heats and current_heats and sig.heats != current_heats and (sig.cert_id == current_cert or (not sig.cert_id and not current_cert)))
        page_fraction_restarted = bool(
            page_fraction
            and previous_fraction
            and page_fraction[0] == 1
            and page_fraction != previous_fraction
            and not sig.cert_id
            and not current_cert
        )

        if cert_changed or heat_changed or page_fraction_restarted:
            flush()
            current = [ext.page_index]
            current_cert = sig.cert_id
            current_heats = sig.heats
            previous_fraction = page_fraction
            continue

        if not sig.cert_id and not sig.heats and _is_continuation(ext):
            current.append(ext.page_index)
            continue

        if not sig.cert_id and not sig.heats:
            current.append(ext.page_index)
            continue

        if sig.cert_id and not current_cert:
            current_cert = sig.cert_id
        if sig.heats:
            current_heats = sig.heats
        current.append(ext.page_index)
        if page_fraction:
            previous_fraction = page_fraction

    flush()
    return groups


def build_group_signature(group_pages: list[int], extractions: list[PageExtraction], vendor: str) -> GroupSignature:
    subset = [e for e in extractions if e.page_index in set(group_pages)]
    certs = [e.signature.cert_id for e in subset if e.signature and e.signature.cert_id]
    heats = sorted(set(h for e in subset if e.signature for h in e.signature.heats))
    cert_id = certs[0] if certs else ""
    score = round(sum(e.signature.score for e in subset if e.signature) / max(len(subset), 1), 2)
    return GroupSignature(cert_id=cert_id, heats=tuple(heats), vendor=vendor.upper() or "UNKNOWN", score=score, pages=group_pages)
