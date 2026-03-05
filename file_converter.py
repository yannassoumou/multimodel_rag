"""
File conversion step: PPTX and XLSX → PDF.
Uses LibreOffice in headless mode for conversion so the rest of the pipeline
can process documents as PDFs (e.g. pdf2image → embeddings).
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Extensions supported for conversion to PDF
CONVERTIBLE_EXTENSIONS = (".pptx", ".xlsx")

# LibreOffice executable names per platform
LIBREOFFICE_NAMES = ("libreoffice", "soffice")


def _find_libreoffice() -> str | None:
    """Return path to LibreOffice executable, or None if not found."""
    for name in LIBREOFFICE_NAMES:
        path = shutil.which(name)
        if path:
            return path
    return None


def convert_to_pdf(
    source_path: str, out_dir: str | None = None
) -> tuple[str | None, bool, str | None]:
    """
    Convert a PPTX or XLSX file to PDF.

    Uses LibreOffice in headless mode. The output PDF is written to out_dir
    (default: a temporary directory). When out_dir is None, the caller should
    remove temp_dir (e.g. shutil.rmtree) after use.

    Args:
        source_path: Absolute or relative path to a .pptx or .xlsx file.
        out_dir: Directory for the output PDF. If None, a temp dir is used.

    Returns:
        (pdf_path, is_temp, temp_dir): Path to the generated PDF; True if
        a temp dir was used; and that temp dir path (for cleanup). On failure
        returns (None, False, None). When is_temp is False, temp_dir is None.
    """
    path = Path(source_path).resolve()
    if not path.is_file():
        print(f"[convert] File not found: {path}")
        return None, False, None

    suffix = path.suffix.lower()
    if suffix not in CONVERTIBLE_EXTENSIONS:
        print(f"[convert] Unsupported extension {suffix!r}. Supported: {CONVERTIBLE_EXTENSIONS}")
        return None, False, None

    libreoffice = _find_libreoffice()
    if not libreoffice:
        print("[convert] LibreOffice not found. Install it (e.g. apt install libreoffice) for PPTX/XLSX conversion.")
        return None, False, None

    is_temp = out_dir is None
    temp_dir = None
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="doc_convert_")
        temp_dir = out_dir

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # LibreOffice writes <outdir>/<stem>.pdf
    try:
        subprocess.run(
            [
                libreoffice,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(out_path),
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=120,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[convert] LibreOffice failed: {e.stderr or e}")
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, False, None
    except subprocess.TimeoutExpired:
        print("[convert] LibreOffice timed out (120s)")
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, False, None
    except FileNotFoundError:
        print("[convert] LibreOffice executable not found")
        return None, False, None

    pdf_file = out_path / f"{path.stem}.pdf"
    if not pdf_file.is_file():
        print(f"[convert] Output PDF not found: {pdf_file}")
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return None, False, None

    print(f"[convert] Converted {path.name} → {pdf_file}")
    return str(pdf_file), is_temp, temp_dir


def is_convertible(path: str) -> bool:
    """Return True if the file is a supported type for conversion (pptx, xlsx)."""
    return Path(path).suffix.lower() in CONVERTIBLE_EXTENSIONS
