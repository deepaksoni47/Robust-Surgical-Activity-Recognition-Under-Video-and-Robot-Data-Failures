# Research Paper Source (LaTeX)

This directory contains the live, publication-ready research paper LaTeX source code for our project.

## Structure
- `main.tex`: Master LaTeX document formatted according to standard IEEE Transactions / Medical Robotics guidelines.
- `references.bib`: BibTeX citations including JIGSAWS, SurgMotion, GRAD, SurgSync, Endo-C6 / RobustEndoCLIP, and foundational surgical workflow literature.

## Continuous Update Policy
As defined in `AGENTS.md` (Rule 4) and `memory.md`, this paper document is maintained and updated incrementally as project milestones and benchmarks complete:
- Pipeline specifications, dataset counts, and mathematical formulations are established.
- Empirical results tables (Tables I and II) will be populated directly from saved evaluation metrics as baselines and fusion models finish execution.
- Figures and qualitative analysis will be inserted during Phase 10.

## How to Compile
### Option A: Overleaf
1. Zip the `paper/` directory.
2. Upload to [Overleaf](https://www.overleaf.com) and click **Recompile**.

### Option B: Local TeX Live / MiKTeX
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
