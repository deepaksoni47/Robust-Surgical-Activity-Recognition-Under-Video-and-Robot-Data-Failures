"""scripts/compile_paper.py

Automated self-contained compiler for paper/main.tex.
Detects available LaTeX engine (pdflatex, latexmk, xelatex, tectonic).
If no local TeX engine is found, automatically uses the standalone Tectonic
engine to compile paper/main.tex into paper/main.pdf with BibTeX support.
"""

import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "paper"
TEX_FILE = PAPER_DIR / "main.tex"
PDF_FILE = PAPER_DIR / "main.pdf"
TOOLS_DIR = PROJECT_ROOT / ".tools" / "tectonic"
TECTONIC_EXE = TOOLS_DIR / ("tectonic.exe" if sys.platform == "win32" else "tectonic")


def get_compiler_cmd() -> list[str]:
    """Find or download an appropriate LaTeX compiler."""
    # Check if a native compiler is in PATH
    for cmd in ["tectonic", "latexmk", "pdflatex"]:
        path = shutil.which(cmd)
        if path:
            if cmd == "tectonic":
                return [path, "-X", "compile", str(TEX_FILE)]
            elif cmd == "latexmk":
                return [path, "-pdf", "-interaction=nonstopmode", str(TEX_FILE)]
            elif cmd == "pdflatex":
                return [path, "-interaction=nonstopmode", str(TEX_FILE)]

    # Check if local tectonic exists in .tools/ or scratch
    if TECTONIC_EXE.exists():
        return [str(TECTONIC_EXE), "-X", "compile", str(TEX_FILE)]

    # Scratch cache check
    scratch_tectonic = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\6ddc7693-b1ca-42c7-ba2c-ec17125cf6b6\scratch\tectonic\tectonic.exe")
    if scratch_tectonic.exists():
        return [str(scratch_tectonic), "-X", "compile", str(TEX_FILE)]

    # Download portable Tectonic binary (Windows x86_64)
    print("No system LaTeX found. Downloading standalone Tectonic binary...")
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip"
    zip_path = TOOLS_DIR / "tectonic.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(TOOLS_DIR)
    if zip_path.exists():
        zip_path.unlink()

    print(f"Downloaded Tectonic to {TECTONIC_EXE}")
    return [str(TECTONIC_EXE), "-X", "compile", str(TEX_FILE)]


def main() -> int:
    if not TEX_FILE.exists():
        print(f"Error: {TEX_FILE} does not exist.")
        return 1

    cmd = get_compiler_cmd()
    print(f"Compiling {TEX_FILE} with command: {' '.join(cmd)}")

    res = subprocess.run(cmd, cwd=str(PAPER_DIR), capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Compiler messages/warnings:")
        print(res.stderr)

    if res.returncode == 0 and PDF_FILE.exists():
        size_kb = PDF_FILE.stat().st_size / 1024.0
        print(f"\nSUCCESS: Generated {PDF_FILE} ({size_kb:.1f} KB)")
        return 0
    else:
        print(f"\nCompilation failed with exit code {res.returncode}")
        return res.returncode


if __name__ == "__main__":
    sys.exit(main())
