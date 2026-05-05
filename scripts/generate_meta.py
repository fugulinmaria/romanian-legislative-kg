"""
Auto-generate .meta.json sidecar files from Romanian legislative .txt headers.

Usage:
    python scripts/generate_meta.py [--dir data/raw_laws]

The script reads the first ~10 lines of each .txt file, extracts the structured
header (act type, number, date, emitter, Monitorul Oficial reference), and writes
a companion .meta.json file alongside each .txt.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

MONTH_MAP = {
    "ianuarie": "01",
    "februarie": "02",
    "martie": "03",
    "aprilie": "04",
    "mai": "05",
    "iunie": "06",
    "iulie": "07",
    "august": "08",
    "septembrie": "09",
    "octombrie": "10",
    "noiembrie": "11",
    "decembrie": "12",
}

# Maps header prefix → canonical tip_act value
TIP_ACT_MAP = {
    "ORDONANȚĂ DE URGENȚĂ": "Ordonanță de urgență",
    "ORDONANȚĂ": "Ordonanță",
    "HOTĂRÂRE": "Hotărâre de guvern",
    "LEGE": "Lege",
    "CONSTITUȚIE": "Constituție",
    # Codes are published as laws
    "CODUL": "Lege",
    "CODULUI": "Lege",
}


def _ro_date(day: str, month_ro: str, year: str) -> str:
    """Convert Romanian date parts to ISO 8601 (YYYY-MM-DD)."""
    month = MONTH_MAP.get(month_ro.strip().lower())
    if not month:
        return f"{year}-01-01"  # fallback; flag for manual review
    return f"{year}-{month}-{day.zfill(2)}"


def _normalize(text: str) -> str:
    """NFC-normalize and strip extra whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def _clean_header_line(line: str) -> str:
    """Remove republication markers like (*republicată*) (**republicat**)."""
    return re.sub(r"\([\*]+[^)]*[\*]+\)", "", line).strip()


def extract_meta(txt_path: Path) -> dict:
    """
    Parse the header of a Romanian legislative text file and return a metadata dict.
    """
    raw = txt_path.read_text(encoding="utf-8")
    # Work only with the first 15 non-empty lines for speed
    lines = [_normalize(ln) for ln in raw.split("\n") if _normalize(ln)][:15]

    law_id = txt_path.stem  # e.g. "lege_190_2018"
    tip_act = ""
    numar = ""
    an = ""
    data_act = ""
    titlu = ""
    emitent = ""
    data_publicare = ""
    monitorul_oficial_nr = ""

    # -------------------------------------------------------------------------
    # Line 0: act type + number + date
    # e.g. "HOTĂRÂRE nr. 214 din 9 aprilie 2026"
    # e.g. "ORDONANȚĂ DE URGENȚĂ nr. 13 din 5 martie 2026"
    # e.g. "CODUL MUNCII din 24 ianuarie 2003 (**republicat**)"
    # -------------------------------------------------------------------------
    header_line = _clean_header_line(lines[0]) if lines else ""

    # Detect tip_act by longest matching prefix (order matters: OUG before OG)
    for prefix, canonical in TIP_ACT_MAP.items():
        norm_prefix = unicodedata.normalize("NFC", prefix)
        if header_line.upper().startswith(norm_prefix):
            tip_act = canonical
            break

    # Extract "nr. X" pattern for act number
    nr_match = re.search(r"\bnr\.?\s*(\d+)", header_line, re.IGNORECASE)
    if nr_match:
        numar = nr_match.group(1)

    # Special case: code laws like "CODUL MUNCII din ... \n (Legea nr. 53/2003)"
    if not numar and tip_act == "Lege":
        # Try next line: "(Legea nr. 53/2003)"
        for ln in lines[1:4]:
            alt = re.search(r"Legea\s+nr\.?\s*(\d+)/(\d{4})", ln, re.IGNORECASE)
            if alt:
                numar = alt.group(1)
                an = alt.group(2)
                break

    # Extract date from "din DD luna YYYY"
    date_match = re.search(
        r"\bdin\s+(\d{1,2})\s+(ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\s+(\d{4})",
        header_line,
        re.IGNORECASE,
    )
    if date_match:
        data_act = _ro_date(date_match.group(1), date_match.group(2), date_match.group(3))
        if not an:
            an = date_match.group(3)

    # -------------------------------------------------------------------------
    # Line 1: title (starts with "privind", "pentru", "cu privire la", or is the
    # code name like "CODUL MUNCII" itself)
    # -------------------------------------------------------------------------
    if len(lines) > 1:
        candidate = lines[1]
        # Skip if it looks like the Legea nr. X/YYYY reference for codes
        if not re.match(r"^\(Legea", candidate, re.IGNORECASE):
            titlu = candidate

    # -------------------------------------------------------------------------
    # Find emitter: line after "EMITENT"
    # -------------------------------------------------------------------------
    for i, ln in enumerate(lines):
        if re.match(r"^EMITENT", ln, re.IGNORECASE):
            if i + 1 < len(lines):
                emitent = lines[i + 1]
            break

    # -------------------------------------------------------------------------
    # Find Monitorul Oficial: "Publicat în  MONITORUL OFICIAL nr. X din ..."
    # -------------------------------------------------------------------------
    for ln in lines:
        mo_match = re.search(
            r"MONITORUL\s+OFICIAL\s+nr\.?\s*(\d+)\s+din\s+(\d{1,2})\s+(ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\s+(\d{4})",
            ln,
            re.IGNORECASE,
        )
        if mo_match:
            monitorul_oficial_nr = mo_match.group(1)
            data_publicare = _ro_date(mo_match.group(2), mo_match.group(3), mo_match.group(4))
            break

    return {
        "law_id": law_id,
        "tip_act": tip_act,
        "numar": numar,
        "an": an,
        "titlu": titlu,
        "emitent": emitent,
        "data_act": data_act,
        "data_publicare": data_publicare,
        "monitorul_oficial_nr": monitorul_oficial_nr,
        "source_url": "https://legislatie.just.ro",
        "in_force": True,
    }


def generate_all(directory: str = "data/raw_laws", overwrite: bool = False) -> None:
    base = Path(directory)
    txt_files = sorted(base.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {base.resolve()}")
        return

    print(f"Found {len(txt_files)} .txt file(s) in {base.resolve()}\n")

    for txt_path in txt_files:
        meta_path = txt_path.with_suffix(".meta.json")

        if meta_path.exists() and not overwrite:
            print(f"[SKIP]  {meta_path.name} already exists (use --overwrite to replace)")
            continue

        meta = extract_meta(txt_path)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK]    {meta_path.name}")
        # Quick sanity print
        print(
            f"        {meta['tip_act']} nr.{meta['numar']}/{meta['an']} | "
            f"MO nr.{meta['monitorul_oficial_nr']} | pub:{meta['data_publicare']}"
        )

    print("\nDone. Review any empty fields and fill in 'source_url' manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate .meta.json for Romanian law .txt files")
    parser.add_argument("--dir", default="data/raw_laws", help="Directory containing .txt files")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing .meta.json files"
    )
    args = parser.parse_args()
    generate_all(args.dir, args.overwrite)
