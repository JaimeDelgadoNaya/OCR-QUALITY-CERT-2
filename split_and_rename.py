import os
import re
import csv
import sys
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Dict, List
from collections import Counter

# PyMuPDF import robusto (fitz)
try:
    import fitz  # PyMuPDF clásico
except ImportError:  # pragma: no cover
    import pymupdf as fitz  # PyMuPDF nuevo

import tkinter as tk
from tkinter import filedialog, messagebox


# ============================================================
# CONFIGURACIÓN (muchas pasan a ser ajustables desde GUI)
# ============================================================

@dataclass
class Config:
    # Naming
    DEFAULT_NORMA: str = "ASME_BPE"
    DEFAULT_SF: str = "SF?"
    MAX_NAME: int = 180

    # OCR
    OCR_MODE: str = "auto"          # auto | none | skip | redo | force
    OCR_OVERSAMPLE: str = "450"
    OCR_TIMEOUT: str = "0"
    OCR_JOBS: str = "1"
    OCR_LANG: str = "eng"          # OJO: si no hay idioma instalado, OCR fallará.

    # Decisión OCR (auto)
    SAMPLE_PAGES_MAX: int = 8
    MIN_TEXT_LEN_PAGE: int = 30
    MIN_QUALITY_FORCE: float = 25.0
    MIN_CERT_HIT_RATIO: float = 0.40

    # Ventanas de líneas
    CERT_LOOKBACK_LINES: int = 4
    CERT_LOOKAHEAD_LINES: int = 14
    HEAT_LOOKAHEAD_LINES: int = 18

    # Colada: regla base (mín. 1 dígito) + excepción controlada
    ALLOW_ALPHA_HEAT_IF_LABEL: bool = True
    ALPHA_HEAT_MAXLEN: int = 6
    ALPHA_HEAT_DISALLOW: Tuple[str, ...] = (
        "OK", "SI", "NO", "ASME", "BPE", "AISI", "ASTM", "EN", "ISO"
    )

    # Agrupación
    ATTACH_ORPHANS_POLICY: str = "next"  # Asumido: adjuntar al siguiente. (No especificado: otras políticas)
    SPLIT_WITHIN_CERT_ON_HEAT_CHANGE: bool = False  # No especificado -> default conservador

    # Rotación
    NORMALIZE_ROTATION: bool = False  # No especificado -> se preserva por defecto

    # Zonas (fracciones del tamaño de página): mejora captura de campos
    ZONES: Dict[str, Tuple[float, float, float, float]] = field(default_factory=lambda: {
        "header": (0.00, 0.00, 1.00, 0.30),
        "top_right": (0.62, 0.00, 1.00, 0.28),
        "mid_left": (0.00, 0.22, 0.55, 0.75),
        "mid_right": (0.45, 0.22, 1.00, 0.75),
        "bottom": (0.00, 0.70, 1.00, 1.00),
    })


# ============================================================
# REGEX: patrones observados (ES/EN/DE/IT + variantes)
# ============================================================

RE_BOXCODE = re.compile(r"^[A-Z]\d{2,3}$")  # A02 / B07 / etc.

# Certificado
RE_CT_LOOSE = re.compile(r"(?i)\bCT\s*[- ]?\s*(\d{6,12})\b")
RE_CERT_PREFIX = re.compile(r"(?i)\bCERT\d{6,12}\b")  # p.ej. CERT25007297 (Genca)
RE_CERT_LABEL_VALUE = re.compile(
    r"(?i)\b(?:Certificado|Certificate|Certificato|Zeugnis(?:\s*[- ]?Nr\.?)?|"
    r"Inspection\s*Certificate|Mill\s*Test|Document\s*No\.?)\b"
    r"[^\w]{0,12}"
    r"([A-Z0-9][A-Z0-9\/\-\._]{2,40})"
)
RE_CERT_LABEL_ONLY = re.compile(
    r"(?i)^\s*(?:Certificado|Certificate|Certificato|Zeugnis(?:\s*[- ]?Nr\.?)?|"
    r"Inspection\s*Certificate|Mill\s*Test|Document\s*No\.?)\s*[:\-\.]?\s*$"
)

# “PCxxxx + número” (muy frecuente en albaranes)
RE_PC_PLUS_NUM = re.compile(r"(?i)\bPC\d{6,}\s+(\d{7,12})\b")

# “Nº 0000369904/1” (ceros + barra)
RE_CERT_N_ES = re.compile(r"(?i)\bN[º°o\.]*\s*([0-9]{6,12}/\d{1,3})\b")

# “No : 4820/5”
RE_CERT_NO_COLON = re.compile(r"(?i)\bNo\b\s*:\s*([0-9]{2,8}/\d{1,4})\b")
CERT_CONTEXT_HINTS = ("INSPECTION", "CERTIFICATE", "CERTIFICADO", "ZEUGNIS", "EN 10204", "EN10204", "MILL TEST", "3.1")

# Colada / Heat
RE_HEAT_INLINE = re.compile(
    r"(?i)\b(?:HEAT\s*NO\.?|COLADA|COLATA|SCHMELZE(?:\s*NR\.?)?|COULEE)\b"
    r"(?!\s*TREAT(?:MENT)?)"
    r"[\s\.\-:·_]*"
    r"([A-Z0-9][A-Z0-9\-\/]{1,20})\b"
)
RE_HEAT_LABEL_ONLY = re.compile(
    r"(?i)^\s*(?:HEAT\s*NO\.?|COLADA|COLATA|SCHMELZE(?:\s*NR\.?)?|COULEE)\s*[:\-\.]?\s*$"
)
RE_HEAT_TABLE_HEADER = re.compile(r"(?i)^\s*Heat#?\s*$")

RE_PUNCT_LINE = re.compile(r"^[\s\.\-:·_]{2,}$")
RE_STOP_HEAT_SCAN = re.compile(
    r"(?i)^(?:Artículo|Articulo|Article|Item|Albar[aá]n|Delivery|Description|Descripción|Cliente|Customer|"
    r"Referencia|Reference|Pedido|Order)\b"
)
COMMON_NON_HEAT_TOKENS = {
    "10204", "EN10204", "10253", "EN10253", "10217", "EN10217", "10255", "EN10255",
    "2004", "2014", "2015", "2020", "2021", "2022", "2023", "2024", "2025", "2026",
}

# “Page 1/4”, “PAG. 1/3”, “1 de 7”
RE_PAGE_FRAC = re.compile(r"(?i)\b(?:PAG\.?|PAGE|SHEET)\s*(\d+)\s*/\s*(\d+)\b|\b(\d+)\s*de\s*(\d+)\b")


# ============================================================
# MODELOS
# ============================================================

@dataclass
class Fields:
    cert_raw: str
    colada_raw: str
    prov_raw: str
    desc: str
    norma: str
    sf: str
    dn_mm: str
    dn_in: str
    confidence: float


@dataclass
class PageSignal:
    page_index: int              # 0-based
    text_len: int
    quality: float
    is_cert_like: bool
    has_heat_label: bool
    page_cur: int
    page_total: int
    cert_raw: str
    cert_norm: str
    cert_score: int
    heat_raw: str
    heat_norm: str
    heat_score: int


# ============================================================
# UTILIDADES
# ============================================================

def normalize_value(x: Any) -> str:
    if x is None:
        return ""
    s = x if isinstance(x, str) else str(x)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_id_for_compare(s: str) -> str:
    t = normalize_value(s)
    t = re.sub(r"\s+", "", t)
    return t.upper()


def safe_filename_component(s: str, max_len: int) -> str:
    """
    Sanitiza para nombre de archivo:
      - preserva ceros; no convierte a int
      - reemplaza / y \\ por -
      - elimina caracteres inválidos (Windows)
    """
    t = normalize_value(s)
    t = t.replace("/", "-").replace("\\", "-")
    t = t.replace(":", "-").replace("*", "").replace("?", "")
    t = t.replace('"', "").replace("<", "").replace(">", "").replace("|", "")
    t = t.replace("’", "'").replace("´", "'")
    t = re.sub(r"\s+", "_", t)
    t = t.strip("._- ")
    if not t:
        t = "NA"
    return t[:max_len].rstrip("_")


def iter_clean_lines(text: str, max_lines: int = 4000) -> List[str]:
    out: List[str] = []
    for line in (text or "").splitlines():
        l = line.strip()
        if l:
            out.append(l)
        if len(out) >= max_lines:
            break
    return out


def write_debug(out_dir: str, name: str, content: str) -> None:
    try:
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(content or "")
    except Exception:
        pass


# ============================================================
# OCR: comprobaciones sin asumir instalación
# ============================================================

def _prepend_to_path(dir_path: str) -> None:
    if not dir_path:
        return
    cur = os.environ.get("PATH", "")
    parts = cur.split(os.pathsep)
    if dir_path not in parts:
        os.environ["PATH"] = dir_path + os.pathsep + cur


def ensure_tesseract_in_path() -> Optional[str]:
    exe = shutil.which("tesseract")
    if exe:
        return exe

    # rutas típicas Windows
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\ProgramData\chocolatey\bin\tesseract.exe",
    ]
    user = os.environ.get("USERPROFILE", "")
    if user:
        candidates.append(os.path.join(user, r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe"))

    for p in candidates:
        if os.path.exists(p):
            _prepend_to_path(os.path.dirname(p))
            return p
    return None


def ensure_ghostscript_in_path() -> Optional[str]:
    if os.name != "nt":
        return shutil.which("gs")
    return shutil.which("gswin64c") or shutil.which("gswin32c") or shutil.which("gs")


def ensure_ocr_tools() -> Dict[str, str]:
    info: Dict[str, str] = {}
    try:
        import ocrmypdf  # noqa: F401
        info["ocrmypdf"] = "OK"
    except Exception as e:
        raise RuntimeError("No encuentro el módulo 'ocrmypdf'. Instala: pip install ocrmypdf") from e

    tpath = ensure_tesseract_in_path()
    if shutil.which("tesseract") is None and not tpath:
        raise RuntimeError(
            "No encuentro 'tesseract' en PATH.\n"
            "Instálalo o añade su carpeta al PATH.\n"
            "Ruta típica: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )
    info["tesseract"] = shutil.which("tesseract") or tpath or "OK"

    gspath = ensure_ghostscript_in_path()
    info["ghostscript"] = gspath or "NO_ENCONTRADO (según versión puede ser opcional)"
    return info


def run_ocr(input_pdf: str, out_pdf: str, cfg: Config, mode: str) -> Dict[str, str]:
    """
    mode:
      - skip  -> --skip-text
      - redo  -> --redo-ocr
      - force -> --force-ocr
    """
    info = ensure_ocr_tools()

    cmd = [
        sys.executable, "-m", "ocrmypdf",
        "--rotate-pages",
        "--deskew",
        "--oversample", cfg.OCR_OVERSAMPLE,
        "--tesseract-timeout", cfg.OCR_TIMEOUT,
        "--jobs", cfg.OCR_JOBS,
        "--language", cfg.OCR_LANG,
        "--optimize", "0",
    ]

    if mode == "force":
        cmd += ["--force-ocr"]
    elif mode == "redo":
        cmd += ["--redo-ocr"]
    else:
        cmd += ["--skip-text"]

    cmd += [input_pdf, out_pdf]

    p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        hint = ""
        if os.name == "nt":
            hint = (
                "\n\nTip Windows: si el error menciona 'gs'/'gswin64c' o rasterización, "
                "instala Ghostscript o usa una versión de OCRmyPDF que soporte rasterización alternativa."
            )
        raise RuntimeError("OCR falló:\n" + err + hint)

    return info


# ============================================================
# PyMuPDF: texto completo + zonas
# ============================================================

def page_text(doc: Any, i: int) -> str:
    return normalize_value(doc.load_page(i).get_text("text"))


def zone_text(page: Any, zone: Tuple[float, float, float, float]) -> str:
    r = page.rect
    x0, y0, x1, y1 = zone
    clip = fitz.Rect(
        r.x0 + x0 * r.width,
        r.y0 + y0 * r.height,
        r.x0 + x1 * r.width,
        r.y0 + y1 * r.height
    )
    return normalize_value(page.get_text("text", clip=clip))


def build_page_texts(doc: Any, page_index: int, cfg: Config) -> Dict[str, str]:
    p = doc.load_page(page_index)
    out: Dict[str, str] = {}
    out["full"] = page_text(doc, page_index)
    for k, z in cfg.ZONES.items():
        out[k] = zone_text(p, z)
    return out


def build_page_combo(texts: Dict[str, str]) -> str:
    parts = [texts.get("full", "")]
    # Orden intencional: header -> top_right -> mid/bottom
    for k in ("header", "top_right", "mid_left", "mid_right", "bottom"):
        if texts.get(k):
            parts.append(texts[k])
    return "\n".join([p for p in parts if p])


# ============================================================
# Calidad / OCR auto
# ============================================================

def text_quality_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    up = t.upper()
    anchors = 0
    for a in ("CERT", "CERTIFICATE", "CERTIFICADO", "INSPECTION", "EN 10204", "ZEUGNIS", "COLADA", "HEAT"):
        if a in up:
            anchors += 1
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", t)
    good_words = len(words)
    long_nums = len(re.findall(r"\d{3,}", t))
    score = 0.0
    score += min(40.0, good_words * 2.0)
    score += min(35.0, anchors * 7.0)
    score += min(15.0, long_nums * 3.0)
    return max(0.0, min(100.0, score))


def looks_like_certificate(text: str) -> bool:
    up = (text or "").upper()
    if not up.strip():
        return False
    return any(k in up for k in ("CERTIFICATE", "CERTIFICADO", "INSPECTION", "EN 10204", "ZEUGNIS", "MILL TEST"))


def extract_page_fraction(text: str) -> Tuple[int, int]:
    m = RE_PAGE_FRAC.search(text or "")
    if not m:
        return (0, 0)
    if m.group(1) and m.group(2):
        return (int(m.group(1)), int(m.group(2)))
    if m.group(3) and m.group(4):
        return (int(m.group(3)), int(m.group(4)))
    return (0, 0)


def sample_page_indices(n: int, max_k: int) -> List[int]:
    if n <= 0:
        return []
    if n <= max_k:
        return list(range(n))
    picks = {0, 1, 2, n - 1, n // 2, n // 3, (2 * n) // 3}
    picks = [p for p in sorted(picks) if 0 <= p < n]
    return picks[:max_k]


def ocr_decision_auto(doc: Any, cfg: Config) -> Tuple[str, float, str]:
    """
    Decide:
      - none: sin OCR
      - skip: OCR solo en páginas sin texto
      - redo: OCR adicional para páginas con texto pero con campos clave no extraíbles
      - force: OCR total (rasteriza todo)
    """
    idxs = sample_page_indices(doc.page_count, cfg.SAMPLE_PAGES_MAX)
    if not idxs:
        return ("none", 0.0, "documento vacío")

    combos: List[str] = []
    empty = 0
    worst = 100.0

    cert_like = 0
    cert_hits = 0
    heat_hits = 0

    for i in idxs:
        texts = build_page_texts(doc, i, cfg)
        combo = build_page_combo(texts)
        combos.append(combo)

        if len(combo.strip()) < cfg.MIN_TEXT_LEN_PAGE:
            empty += 1

        q = text_quality_score(combo)
        worst = min(worst, q)

        if looks_like_certificate(combo):
            cert_like += 1
            cert, _, _ = extract_cert_number(combo, texts)
            heat, _, _ = extract_heat(combo)
            if cert:
                cert_hits += 1
            if heat:
                heat_hits += 1

    # Ratio de éxito en páginas que parecen certificados
    ratio = (cert_hits / cert_like) if cert_like > 0 else 1.0

    # Escaneado puro o casi sin texto
    if empty == len(combos) and worst <= 1.0:
        return ("skip", worst, "todas las páginas muestreadas sin texto")

    # Texto pobre
    if worst < cfg.MIN_QUALITY_FORCE:
        # si hay mezcla con páginas sin texto, skip puede bastar
        if empty > 0 and ratio >= 0.60:
            return ("skip", worst, f"páginas sin texto pero cert OK (ratio {ratio:.2f})")
        return ("force", worst, f"texto muy pobre (worst {worst:.1f})")

    # Caso clave: páginas con texto, pero no se detecta certificado (p.ej. nº dentro de imagen)
    if cert_like > 0 and ratio < cfg.MIN_CERT_HIT_RATIO:
        # preferimos redo antes que force: conserva texto existente
        return ("redo", worst, f"pocas páginas con cert detectable (ratio {ratio:.2f})")

    # Mixto texto + escaneo
    if empty > 0:
        return ("skip", worst, "hay páginas sin texto")

    return ("none", worst, "texto suficiente y señales extraíbles")


# ============================================================
# Extracción (CERT / HEAT) con candidatos + scoring
# ============================================================

def add_candidate(cands: List[Tuple[str, int, str]], value: str, score: int, src: str) -> None:
    v = normalize_value(value)
    if not v:
        return
    v = re.sub(r"\s+", "", v)  # para comparar y evitar variantes con espacios
    if len(v) < 3:
        return
    if RE_BOXCODE.match(v):
        return
    cands.append((v, score, src))


def pick_best(cands: List[Tuple[str, int, str]]) -> Tuple[str, int, str]:
    if not cands:
        return ("", 0, "")
    cands.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    return cands[0]


def extract_cert_number(combo_text: str, zone_texts: Dict[str, str]) -> Tuple[str, List[Tuple[str, int, str]], int]:
    """
    Devuelve:
      best_cert_raw, lista_candidatos, best_score

    Importante: no convierte a int (mantiene ceros y '/').
    """
    t = combo_text or ""
    lines = iter_clean_lines(t, max_lines=3000)
    cands: List[Tuple[str, int, str]] = []

    # 1) CT interno
    for m in RE_CT_LOOSE.finditer(t):
        add_candidate(cands, "CT" + m.group(1), 100, "ct")

    # 2) CERT + dígitos (p.ej. CERT25007297)
    for m in RE_CERT_PREFIX.finditer(t):
        add_candidate(cands, m.group(0), 96, "cert_prefix")

    # 3) Nº 0000369904/1
    for m in RE_CERT_N_ES.finditer(t):
        add_candidate(cands, m.group(1), 95, "n_es")

    # 4) Etiquetas con valor inline (ES/EN/IT/DE)
    for m in RE_CERT_LABEL_VALUE.finditer(t):
        val = m.group(1)
        if re.search(r"\d", val):
            add_candidate(cands, val, 92, "label_inline")

    # 5) PC + número (muy fuerte en albaranes)
    header = zone_texts.get("header", "")
    for m in RE_PC_PLUS_NUM.finditer(header):
        add_candidate(cands, m.group(1), 94, "pc_plus_num")

    # 6) No : 4820/5 con contexto
    for m in RE_CERT_NO_COLON.finditer(t):
        val = m.group(1)
        ctx = t[max(0, m.start() - 140):min(len(t), m.end() + 140)].upper()
        if any(k in ctx for k in CERT_CONTEXT_HINTS):
            add_candidate(cands, val, 85, "no_colon_ctx")

    # 7) Label-only en líneas: buscar alrededor (lookback/lookahead)
    for i, line in enumerate(lines[:1200]):
        if not RE_CERT_LABEL_ONLY.match(line):
            continue
        j0 = max(0, i - 1 - cfg_global.CERT_LOOKBACK_LINES)
        j1 = min(len(lines), i + 1 + cfg_global.CERT_LOOKAHEAD_LINES)
        window = lines[j0:j1]
        for cand_line in window:
            if RE_CERT_LABEL_ONLY.match(cand_line):
                continue
            # prioriza tokens con dígitos y que no sean “ruido puro”
            tok = re.search(r"\b([A-Z0-9][A-Z0-9\/\-\._]{2,40})\b", cand_line, re.IGNORECASE)
            if tok and re.search(r"\d", tok.group(1)):
                add_candidate(cands, tok.group(1), 80, "label_nearby")
                break

    best, best_sc, _ = pick_best(cands)
    return best, cands, best_sc


def _allow_alpha_heat(cfg: Config, token: str) -> bool:
    if not cfg.ALLOW_ALPHA_HEAT_IF_LABEL:
        return False
    if not token.isalpha():
        return False
    if len(token) > cfg.ALPHA_HEAT_MAXLEN:
        return False
    if token.upper() in cfg.ALPHA_HEAT_DISALLOW:
        return False
    return True


def extract_heat(combo_text: str) -> Tuple[str, List[Tuple[str, int, str]], int]:
    """
    Regla base: exige >=1 dígito.
    Excepción: token alfabético corto solo si viene tras etiqueta fuerte.
    """
    t = combo_text or ""
    lines = iter_clean_lines(t, max_lines=4000)
    cands: List[Tuple[str, int, str]] = []

    has_label = bool(re.search(r"(?i)\b(?:HEAT|COLADA|COLATA|SCHMELZE|COULEE)\b", t))

    # 1) Inline
    for m in RE_HEAT_INLINE.finditer(t):
        tok = m.group(1)
        up = tok.upper()
        if up in COMMON_NON_HEAT_TOKENS or up.startswith("EN102"):
            continue
        if re.search(r"\d", tok):
            add_candidate(cands, tok, 90, "heat_inline")
        elif has_label and _allow_alpha_heat(cfg_global, tok):
            add_candidate(cands, tok, 70, "heat_inline_alpha")

    # 2) Label-only lines -> lookahead
    for i, line in enumerate(lines[:1600]):
        if not RE_HEAT_LABEL_ONLY.match(line):
            continue
        for j in range(i + 1, min(len(lines), i + 1 + cfg_global.HEAT_LOOKAHEAD_LINES)):
            cand_line = lines[j]
            if RE_PUNCT_LINE.match(cand_line):
                continue
            # Saltar “Cantidad/Quantity”
            if re.match(r"(?i)^(?:Cantidad|Quantity)\b", cand_line):
                continue
            # Cortar si entramos en otra sección fuerte
            if RE_STOP_HEAT_SCAN.match(cand_line):
                break
            tokm = re.search(r"\b([A-Z0-9][A-Z0-9\-\/]{1,20})\b", cand_line, re.IGNORECASE)
            if not tokm:
                continue
            tok = tokm.group(1)
            up = tok.upper()
            if RE_BOXCODE.match(up):
                continue
            if up in COMMON_NON_HEAT_TOKENS or up.startswith("EN102"):
                continue
            if re.search(r"\d", tok):
                add_candidate(cands, tok, 92, "heat_label_nearby")
                break
            if has_label and _allow_alpha_heat(cfg_global, tok):
                add_candidate(cands, tok, 72, "heat_label_alpha")
                break

    # 3) Tabla Heat#
    for i, line in enumerate(lines[:2500]):
        if not RE_HEAT_TABLE_HEADER.match(line):
            continue
        window = lines[i + 1:i + 650]
        toks: List[str] = []
        for wl in window:
            if RE_PUNCT_LINE.match(wl):
                continue
            for tok in re.findall(r"\b[A-Z0-9][A-Z0-9\-\/]{2,20}\b", wl):
                up = tok.upper()
                if RE_BOXCODE.match(up):
                    continue
                if up in COMMON_NON_HEAT_TOKENS or up.startswith("EN102"):
                    continue
                if not re.search(r"\d", tok):
                    continue
                if tok.isdigit() and len(tok) < 4:
                    continue
                toks.append(tok)
        if toks:
            cnt = Counter(toks)
            val, freq = cnt.most_common(1)[0]
            if freq >= 2:
                add_candidate(cands, val, 90 + min(8, freq), "heat_table_mode")
        break

    # Filtrado final: si no hay etiqueta, no “inventar” colada
    filtered: List[Tuple[str, int, str]] = []
    for v, sc, src in cands:
        up = v.upper()
        if up in COMMON_NON_HEAT_TOKENS or up.startswith("EN102"):
            continue
        if not re.search(r"\d", v):
            if not (has_label and _allow_alpha_heat(cfg_global, v)):
                continue
        filtered.append((v, sc, src))

    best, best_sc, _ = pick_best(filtered)
    if not best and not has_label:
        return ("", filtered, 0)
    return best, filtered, best_sc


# ============================================================
# Size / DN (opcional, no es el núcleo del split)
# ============================================================

RE_GRADE = re.compile(r"\b(316L|304L|316|304|904L|S31603|1\.4404|TP316L|TP304L)\b", re.IGNORECASE)
RE_SF = re.compile(r"\b(SF1|SF2)\b", re.IGNORECASE)
RE_SIZE_MM_IN = re.compile(
    r"([0-9]{1,3}(?:[\,\.][0-9]{1,3})?)\s*mm\s*\(\s*([0-9]{1,2}(?:[\,\.][0-9]{1,3})?)\s*\"?\s*\)\s*x\s*([0-9]{1,2}(?:[\,\.][0-9]{1,3})?)\s*mm\s*\(\s*([0-9]{1,2}(?:[\,\.][0-9]{1,3})?)\s*\"?\s*\)",
    re.IGNORECASE
)
RE_SIZE_MM = re.compile(r"(?:Ø|OD)?\s*([0-9]{1,4}(?:[\,\.][0-9])?)\s*mm\s*(?:x|×)\s*([0-9]{1,3}(?:[\,\.][0-9])?)\s*mm", re.IGNORECASE)
RE_INCH_TOKEN = re.compile(r"\b(\d+\s*\d*\/?\d*)\s*(?:\"|''|\bin\b)\b", re.IGNORECASE)

ODMM_TO_INCH = [(12.7, "1/2"), (19.05, "3/4"), (25.4, "1"), (31.75, "1 1/4"), (38.1, "1 1/2"), (50.8, "2"), (63.5, "2 1/2"), (76.2, "3"), (101.6, "4"), (152.4, "6")]
INCH_TO_DN = {"1/2": "15", "3/4": "20", "1": "25", "1 1/4": "32", "1 1/2": "40", "2": "50", "2 1/2": "65", "3": "80", "4": "100", "5": "125", "6": "150"}


def best_inch_from_odmm(od: float) -> Optional[str]:
    best, best_diff = None, 1e9
    for ref, inch in ODMM_TO_INCH:
        diff = abs(ref - od)
        if diff < best_diff:
            best_diff, best = diff, inch
    return best if best and best_diff <= 2.5 else None


def inch_to_dn(inch: str) -> str:
    key = re.sub(r"\s+", " ", inch.strip())
    return INCH_TO_DN.get(key, "DN?")


def inch_token_for_filename(inch: str) -> str:
    s = re.sub(r"\s+", " ", inch.strip())
    s = s.replace(" ", "-").replace("/", "_")
    return f"{s}in"


def extract_size_dn(text: str) -> Tuple[str, str, str]:
    t = text or ""
    dn_in, dn_mm, size_desc = "", "", ""

    ms = RE_SIZE_MM_IN.search(t)
    if ms:
        mm_od = ms.group(1).replace(",", ".")
        in_od = ms.group(2).replace(",", ".")
        mm_thk = ms.group(3).replace(",", ".")
        size_desc = f"OD{mm_od}mm_THK{mm_thk}mm"
        try:
            od_in = float(in_od)
            common = [(0.5, "1/2"), (0.75, "3/4"), (1.0, "1"), (1.25, "1 1/4"), (1.5, "1 1/2"), (2.0, "2"), (2.5, "2 1/2"), (3.0, "3"), (4.0, "4"), (6.0, "6")]
            best = min(common, key=lambda x: abs(x[0] - od_in))
            if abs(best[0] - od_in) <= 0.08:
                dn_in = best[1]
                dn_mm = inch_to_dn(dn_in)
        except Exception:
            pass

    if not dn_in:
        mi = RE_INCH_TOKEN.search(t)
        if mi:
            dn_in = re.sub(r"\s+", " ", mi.group(1).strip()).replace("-", " ")
            dn_mm = inch_to_dn(dn_in)

    if not dn_in:
        ms2 = RE_SIZE_MM.search(t)
        if ms2:
            try:
                odmm = float(ms2.group(1).replace(",", "."))
                thk = float(ms2.group(2).replace(",", "."))
                inch = best_inch_from_odmm(odmm)
                if inch:
                    dn_in = inch
                    dn_mm = inch_to_dn(inch)
                    size_desc = f"OD{odmm:g}mm_THK{thk:g}mm"
            except Exception:
                pass

    dn_in_fname = inch_token_for_filename(dn_in) if dn_in else "IN?"
    dn_mm_out = dn_mm if dn_mm else "DN?"
    return dn_mm_out, dn_in_fname, size_desc


# ============================================================
# Señal por página
# ============================================================

def build_page_signal(doc: Any, page_index: int, cfg: Config) -> Tuple[PageSignal, Dict[str, Any]]:
    texts = build_page_texts(doc, page_index, cfg)
    combo = build_page_combo(texts)

    tlen = len(combo.strip())
    q = text_quality_score(combo)
    cert_like = looks_like_certificate(combo)
    has_heat_label = bool(re.search(r"(?i)\b(?:HEAT|COLADA|COLATA|SCHMELZE|COULEE)\b", combo))
    cur, total = extract_page_fraction(combo)

    cert_best, cert_cands, cert_score = extract_cert_number(combo, texts)
    heat_best, heat_cands, heat_score = extract_heat(combo)

    sig = PageSignal(
        page_index=page_index,
        text_len=tlen,
        quality=q,
        is_cert_like=cert_like,
        has_heat_label=has_heat_label,
        page_cur=cur,
        page_total=total,
        cert_raw=cert_best,
        cert_norm=normalize_id_for_compare(cert_best),
        cert_score=int(cert_score),
        heat_raw=heat_best,
        heat_norm=normalize_id_for_compare(heat_best),
        heat_score=int(heat_score),
    )
    dbg = {"texts": texts, "combo_sample": combo[:6000], "cert_candidates": cert_cands, "heat_candidates": heat_cands}
    return sig, dbg


# ============================================================
# Agrupación (cert primario, heat fallback, orphans -> siguiente)
# ============================================================

def group_pages(signals: List[PageSignal], cfg: Config) -> List[List[int]]:
    groups: List[List[int]] = []
    current: List[int] = []
    orphans: List[int] = []

    active_cert = ""
    active_heat = ""

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    def start_new_group(first_page: int, cert_norm: str, heat_norm: str) -> None:
        nonlocal current, orphans, active_cert, active_heat
        flush()
        current = orphans + [first_page]
        orphans = []
        active_cert = cert_norm
        active_heat = heat_norm

    n = len(signals)
    for i, s in enumerate(signals):
        next_s = signals[i + 1] if i + 1 < n else None

        # Caso A: hay certificado (clave primaria)
        if s.cert_norm:
            if not current:
                start_new_group(s.page_index, s.cert_norm, s.heat_norm)
            else:
                if active_cert and s.cert_norm != active_cert:
                    start_new_group(s.page_index, s.cert_norm, s.heat_norm)
                else:
                    current.append(s.page_index)
            if s.heat_norm:
                active_heat = s.heat_norm
            continue

        # Caso B: sin certificado
        if not current:
            orphans.append(s.page_index)
            continue

        # Si estamos en un grupo con cert:
        if active_cert:
            # Páginas secuenciales dentro del mismo documento (Page 2/4, 3/4, etc.) suelen pertenecer al cert
            if s.page_total > 0 and s.page_cur > 1:
                current.append(s.page_index)
                continue

            # Si comparte heat, es buena señal
            if s.heat_norm and active_heat and s.heat_norm == active_heat:
                current.append(s.page_index)
                continue

            # Si la siguiente tiene otro cert, este “sin cert” es prefacio del siguiente
            if next_s and next_s.cert_norm and next_s.cert_norm != active_cert:
                orphans.append(s.page_index)
                continue

            # Si el contenido parece certificado, mantener; si no, enviar a orphans
            if s.is_cert_like or s.has_heat_label:
                current.append(s.page_index)
            else:
                orphans.append(s.page_index)
            continue

        # Si NO hay cert activo (grupo basado en heat)
        if s.heat_norm:
            if not active_heat:
                # primer heat del grupo
                current.append(s.page_index)
                active_heat = s.heat_norm
            elif s.heat_norm != active_heat:
                # cambia heat -> nuevo grupo
                start_new_group(s.page_index, "", s.heat_norm)
            else:
                current.append(s.page_index)
            continue

        # Sin cert y sin heat: por defecto orphans (se adjunta al siguiente grupo que arranque)
        orphans.append(s.page_index)

    flush()

    # Orphans finales: no hay “siguiente” -> se adjuntan al último grupo
    if orphans:
        if groups:
            groups[-1].extend(orphans)
        else:
            groups.append(orphans)

    return [sorted(g) for g in groups if g]


# ============================================================
# Campos por grupo + naming
# ============================================================

def mode_best(signals: List[PageSignal], which: str) -> str:
    if which == "cert":
        vals = [s.cert_raw for s in signals if s.cert_norm]
        norms = [s.cert_norm for s in signals if s.cert_norm]
    else:
        vals = [s.heat_raw for s in signals if s.heat_norm]
        norms = [s.heat_norm for s in signals if s.heat_norm]
    if not norms:
        return ""
    best_norm = Counter(norms).most_common(1)[0][0]
    # devuelve un raw que corresponda al norm (preserva formato)
    for s in signals:
        if (which == "cert" and s.cert_norm == best_norm):
            return s.cert_raw
        if (which == "heat" and s.heat_norm == best_norm):
            return s.heat_raw
    return vals[0] if vals else ""


def build_filename(fields: Fields, cfg: Config) -> str:
    cert = safe_filename_component(fields.cert_raw, 50)
    prov = safe_filename_component(fields.prov_raw, 25) if fields.prov_raw else ""
    desc = safe_filename_component(fields.desc, 90)
    norma = safe_filename_component(fields.norma, 20)
    sf = safe_filename_component(fields.sf, 8)
    dn_mm_tag = fields.dn_mm if fields.dn_mm.upper().startswith("DN") else f"DN{fields.dn_mm}"
    dn_mm_tag = safe_filename_component(dn_mm_tag, 12)
    dn_in_tag = safe_filename_component(fields.dn_in, 20)
    colada = safe_filename_component(fields.colada_raw, 25)

    left = cert
    if prov:
        left = f"{left}({prov})"

    name = f"{left}_{desc}_{norma}_{sf}_{dn_mm_tag}_{dn_in_tag}_{colada}.pdf"
    if len(name) > cfg.MAX_NAME:
        fixed = f"{left}_{norma}_{sf}_{dn_mm_tag}_{dn_in_tag}_{colada}.pdf"
        keep = max(10, cfg.MAX_NAME - len(fixed) - 1)
        name = f"{left}_{desc[:keep]}_{norma}_{sf}_{dn_mm_tag}_{dn_in_tag}_{colada}.pdf"
    return name


# ============================================================
# Proceso principal: OCR -> señales -> grupos -> PDFs -> audit/debug
# ============================================================

def insert_pages(newdoc: Any, src: Any, pages: List[int]) -> None:
    """
    Inserta páginas; si son consecutivas, usa rango; si no, inserta una a una.
    """
    if not pages:
        return
    pages = sorted(pages)
    consecutive = all(pages[i] == pages[i - 1] + 1 for i in range(1, len(pages)))
    if consecutive:
        newdoc.insert_pdf(src, from_page=pages[0], to_page=pages[-1])
    else:
        for p in pages:
            newdoc.insert_pdf(src, from_page=p, to_page=p)


def process_pdf(pdf_path: str, proveedor: str, out_dir: str, cfg: Config, status_cb=None) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    # Fallback desde nombre del archivo: CT#### si existiera
    mfn = re.search(r"(?i)(CT\d{6,12})", base.replace(" ", ""))
    fallback_cert = mfn.group(1).upper() if mfn else ""

    if status_cb:
        status_cb("Abriendo PDF...")

    doc0 = fitz.open(pdf_path)
    try:
        if cfg.OCR_MODE == "auto":
            mode, worst_q, rationale = ocr_decision_auto(doc0, cfg)
        else:
            mode = cfg.OCR_MODE
            worst_q, rationale = 0.0, "modo manual"
        sample = "\n\n".join(page_text(doc0, i)[:2000] for i in range(min(3, doc0.page_count)))
    finally:
        doc0.close()

    write_debug(
        out_dir,
        f"{base}__debug_ocr_decision.txt",
        f"ocr_mode={mode}\nworst_quality_score={worst_q:.1f}\nrationale={rationale}\n\nSAMPLE_FIRST_PAGES:\n{sample}"
    )

    work_pdf = pdf_path
    ocr_info: Dict[str, str] = {}
    if mode in ("skip", "redo", "force"):
        if status_cb:
            status_cb(f"OCR ({mode})...")

        ocr_pdf = os.path.join(out_dir, f"{base}__OCR.pdf")
        ocr_info = run_ocr(pdf_path, ocr_pdf, cfg, mode=mode)
        work_pdf = ocr_pdf

        write_debug(out_dir, f"{base}__debug_ocr_env.txt", "\n".join([f"{k}={v}" for k, v in ocr_info.items()]))
    else:
        if status_cb:
            status_cb("Sin OCR.")

    doc = fitz.open(work_pdf)

    if status_cb:
        status_cb("Extrayendo señales por página...")

    page_signals: List[PageSignal] = []
    page_dbg: Dict[int, Dict[str, Any]] = {}

    for i in range(doc.page_count):
        sig, dbg = build_page_signal(doc, i, cfg)
        page_signals.append(sig)
        page_dbg[i] = dbg

    # Debug por página
    pages_csv = os.path.join(out_dir, f"{base}__page_signals.csv")
    with open(pages_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([
            "page", "text_len", "quality", "is_cert_like", "has_heat_label",
            "page_cur", "page_total",
            "cert_raw", "cert_score",
            "heat_raw", "heat_score"
        ])
        for s in page_signals:
            w.writerow([
                s.page_index + 1,
                s.text_len,
                f"{s.quality:.1f}",
                int(s.is_cert_like),
                int(s.has_heat_label),
                s.page_cur,
                s.page_total,
                s.cert_raw,
                s.cert_score,
                s.heat_raw,
                s.heat_score,
            ])

    if status_cb:
        status_cb("Agrupando páginas (split)...")

    groups = group_pages(page_signals, cfg)

    # Debug mapa página -> grupo
    page_to_group_csv = os.path.join(out_dir, f"{base}__page_to_group.csv")
    with open(page_to_group_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["page", "group"])
        for gi, pages in enumerate(groups, start=1):
            for p in pages:
                w.writerow([p + 1, gi])

    outputs: List[str] = []
    audit_csv = os.path.join(out_dir, "audit.csv")
    if not os.path.exists(audit_csv):
        with open(audit_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["input", "output", "pages", "confidence", "cert_raw", "colada_raw", "sf", "dn_mm", "dn_in"])

    used_names = set()

    for gi, pages in enumerate(groups, start=1):
        if status_cb:
            status_cb(f"Generando grupo {gi}/{len(groups)}...")

        sigs = [page_signals[p] for p in pages]
        cert_raw = mode_best(sigs, "cert") or fallback_cert or f"CERT_{gi:03d}"
        heat_raw = mode_best(sigs, "heat") or "COLADA?"

        # Construir “desc” a partir de texto de 1-2 páginas representativas
        sample_pages = pages[:2] + ([pages[-1]] if len(pages) >= 3 else [])
        combo_for_desc = "\n".join(build_page_combo(build_page_texts(doc, p, cfg)) for p in dict.fromkeys(sample_pages))

        m_sf = RE_SF.search(combo_for_desc)
        sf = m_sf.group(1).upper() if m_sf else cfg.DEFAULT_SF

        grade_m = RE_GRADE.search(combo_for_desc)
        grade = grade_m.group(1).upper() if grade_m else ""

        dn_mm, dn_in, size_desc = extract_size_dn(combo_for_desc)
        if grade and size_desc:
            desc = f"TUBERIA_{grade}_{size_desc}"
        elif grade:
            desc = f"TUBERIA_{grade}"
        elif size_desc:
            desc = f"TUBERIA_{size_desc}"
        else:
            desc = "TUBERIA"

        # Confianza simple (cobertura)
        n = max(1, len(sigs))
        cert_cov = sum(1 for s in sigs if s.cert_norm) / n
        heat_cov = sum(1 for s in sigs if s.heat_norm) / n
        confidence = min(100.0, 45.0 * cert_cov + 35.0 * heat_cov + (10.0 if dn_mm != "DN?" else 0.0))

        fields = Fields(
            cert_raw=cert_raw,
            colada_raw=heat_raw,
            prov_raw=proveedor,
            desc=desc,
            norma=cfg.DEFAULT_NORMA,
            sf=sf,
            dn_mm=dn_mm,
            dn_in=dn_in,
            confidence=confidence,
        )

        out_name = build_filename(fields, cfg)
        base_out = out_name
        k = 1
        while out_name in used_names or os.path.exists(os.path.join(out_dir, out_name)):
            out_name = base_out.replace(".pdf", f"__{k}.pdf")
            k += 1
        used_names.add(out_name)

        out_path = os.path.join(out_dir, out_name)

        # Crear PDF del grupo
        newdoc = fitz.open()
        insert_pages(newdoc, doc, pages)

        # Rotación (no especificado: por defecto se preserva)
        if cfg.NORMALIZE_ROTATION:
            for pi in range(newdoc.page_count):
                try:
                    newdoc.load_page(pi).set_rotation(0)
                except Exception:
                    pass

        newdoc.save(out_path)
        newdoc.close()
        outputs.append(out_path)

        # Debug por grupo: candidatos + muestra de texto
        dbg_path = os.path.join(out_dir, f"{base}__group{gi:03d}__candidates.txt")
        lines = []
        lines.append(f"=== GROUP {gi} PAGES: {pages[0]+1}-{pages[-1]+1} ===")
        lines.append(f"cert_raw={cert_raw}")
        lines.append(f"colada_raw={heat_raw}")
        lines.append(f"confidence={confidence:.0f}")

        # Mostrar candidatos de la primera página del grupo (útil para entender errores)
        first_dbg = page_dbg[pages[0]]
        lines.append("\n=== CERT CANDIDATES (value | score | source) ===")
        for v, sc, src in first_dbg["cert_candidates"]:
            lines.append(f"{v} | {sc} | {src}")
        lines.append("\n=== HEAT/COLADA CANDIDATES (value | score | source) ===")
        for v, sc, src in first_dbg["heat_candidates"]:
            lines.append(f"{v} | {sc} | {src}")
        lines.append("\n=== TEXT SAMPLE (first page combo) ===\n")
        lines.append(first_dbg["combo_sample"])
        write_debug(out_dir, os.path.basename(dbg_path), "\n".join(lines))

        with open(audit_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([
                os.path.basename(pdf_path),
                out_name,
                f"{pages[0] + 1}-{pages[-1] + 1}",
                f"{confidence:.0f}",
                cert_raw,
                heat_raw,
                sf,
                dn_mm,
                dn_in,
            ])

    doc.close()
    return outputs


# ============================================================
# GUI (Tkinter) - se mantiene y se añaden opciones ajustables
# ============================================================

cfg_global = Config()  # se usará también en extractores para parámetros de ventana

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Certificados → Split/Rename (Optimizado)")
        self.geometry("860x420")
        self.resizable(False, False)

        self.pdf_path = tk.StringVar(value="")
        self.proveedor = tk.StringVar(value="")
        self.out_dir = tk.StringVar(value=os.path.join(os.getcwd(), "salida_gui"))

        # Opciones avanzadas (GUI)
        self.ocr_mode = tk.StringVar(value=cfg_global.OCR_MODE)
        self.ocr_lang = tk.StringVar(value=cfg_global.OCR_LANG)
        self.min_quality = tk.StringVar(value=str(cfg_global.MIN_QUALITY_FORCE))
        self.allow_alpha_heat = tk.BooleanVar(value=cfg_global.ALLOW_ALPHA_HEAT_IF_LABEL)
        self.max_name = tk.StringVar(value=str(cfg_global.MAX_NAME))

        self.status = tk.StringVar(value="Listo.")
        self.running = False

        # ------- Básico -------
        tk.Label(self, text="PDF:").grid(row=0, column=0, padx=12, pady=10, sticky="w")
        tk.Entry(self, textvariable=self.pdf_path, width=78).grid(row=0, column=1, padx=8, pady=10, sticky="w")
        tk.Button(self, text="Elegir…", command=self.pick_pdf, width=12).grid(row=0, column=2, padx=8, pady=10)

        tk.Label(self, text="Proveedor (código o nombre):").grid(row=1, column=0, padx=12, pady=6, sticky="w")
        tk.Entry(self, textvariable=self.proveedor, width=44).grid(row=1, column=1, padx=8, pady=6, sticky="w")

        tk.Label(self, text="Carpeta salida:").grid(row=2, column=0, padx=12, pady=6, sticky="w")
        tk.Entry(self, textvariable=self.out_dir, width=78).grid(row=2, column=1, padx=8, pady=6, sticky="w")
        tk.Button(self, text="Cambiar…", command=self.pick_out_dir, width=12).grid(row=2, column=2, padx=8, pady=6)

        # ------- Avanzado -------
        lf = tk.LabelFrame(self, text="Opciones avanzadas (recomendado tocar solo si hace falta)")
        lf.grid(row=3, column=0, columnspan=3, padx=12, pady=10, sticky="we")

        tk.Label(lf, text="OCR modo:").grid(row=0, column=0, padx=10, pady=6, sticky="w")
        tk.OptionMenu(lf, self.ocr_mode, "auto", "none", "skip", "redo", "force").grid(row=0, column=1, padx=6, pady=6, sticky="w")

        tk.Label(lf, text="OCR idiomas (tesseract):").grid(row=0, column=2, padx=10, pady=6, sticky="w")
        tk.Entry(lf, textvariable=self.ocr_lang, width=16).grid(row=0, column=3, padx=6, pady=6, sticky="w")

        tk.Label(lf, text="Umbral calidad → force:").grid(row=1, column=0, padx=10, pady=6, sticky="w")
        tk.Entry(lf, textvariable=self.min_quality, width=8).grid(row=1, column=1, padx=6, pady=6, sticky="w")

        tk.Checkbutton(lf, text="Permitir colada alfabética (solo con etiqueta)", variable=self.allow_alpha_heat).grid(
            row=1, column=2, columnspan=2, padx=10, pady=6, sticky="w"
        )

        tk.Label(lf, text="Longitud máx. nombre:").grid(row=2, column=0, padx=10, pady=6, sticky="w")
        tk.Entry(lf, textvariable=self.max_name, width=8).grid(row=2, column=1, padx=6, pady=6, sticky="w")

        tk.Label(self, textvariable=self.status, fg="blue").grid(row=4, column=0, columnspan=2, padx=12, pady=14, sticky="w")

        self.btn = tk.Button(self, text="Procesar", command=self.run_process, width=16, height=2)
        self.btn.grid(row=4, column=2, padx=8, pady=10)

        tip = "Salida: PDFs renombrados + audit.csv + __page_signals.csv + __group___candidates.txt"
        tk.Label(self, text=tip, fg="gray").grid(row=5, column=0, columnspan=3, padx=12, pady=6, sticky="w")

    def pick_pdf(self):
        path = filedialog.askopenfilename(title="Selecciona un PDF", filetypes=[("PDF files", "*.pdf")])
        if path:
            self.pdf_path.set(path)

    def pick_out_dir(self):
        path = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if path:
            self.out_dir.set(path)

    def set_status(self, msg: str):
        self.status.set(msg)
        self.update_idletasks()

    def _effective_config(self) -> Config:
        cfg = Config()
        cfg.OCR_MODE = self.ocr_mode.get().strip()
        cfg.OCR_LANG = self.ocr_lang.get().strip() or cfg.OCR_LANG
        cfg.ALLOW_ALPHA_HEAT_IF_LABEL = bool(self.allow_alpha_heat.get())

        try:
            cfg.MIN_QUALITY_FORCE = float(self.min_quality.get().strip())
        except Exception:
            cfg.MIN_QUALITY_FORCE = cfg_global.MIN_QUALITY_FORCE

        try:
            cfg.MAX_NAME = int(self.max_name.get().strip())
        except Exception:
            cfg.MAX_NAME = cfg_global.MAX_NAME

        return cfg

    def _worker(self, pdf: str, prov: str, out_dir: str, cfg: Config):
        global cfg_global
        cfg_global = cfg  # para que extractores usen la ventana configurada

        try:
            outs = process_pdf(pdf, prov, out_dir, cfg, status_cb=lambda m: self.after(0, self.set_status, m))
            self.after(0, self.set_status, "Terminado.")
            self.after(0, lambda: messagebox.showinfo(
                "OK",
                f"Generados {len(outs)} PDF(s).\n\nRevisa audit.csv y los archivos __debug__ en la carpeta de salida."
            ))
        except Exception as e:
            msg = str(e)
            self.after(0, self.set_status, "Error.")
            self.after(0, lambda m=msg: messagebox.showerror("Fallo", m))
        finally:
            self.running = False
            self.after(0, lambda: self.btn.config(state="normal"))

    def run_process(self):
        if self.running:
            return

        pdf = self.pdf_path.get().strip()
        prov = self.proveedor.get().strip()
        out_dir = self.out_dir.get().strip()

        if not pdf or not os.path.exists(pdf):
            messagebox.showerror("Error", "Selecciona un PDF válido.")
            return

        if not prov:
            if not messagebox.askyesno("Proveedor vacío", "No has puesto proveedor. ¿Continuar igualmente?"):
                return

        cfg = self._effective_config()

        self.running = True
        self.btn.config(state="disabled")
        self.set_status("Iniciando...")

        th = threading.Thread(target=self._worker, args=(pdf, prov, out_dir, cfg), daemon=True)
        th.start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
