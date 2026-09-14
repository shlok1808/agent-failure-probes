# Overleaf upload

1. New Project -> Upload Project -> drop this zip in.
2. Compiler is pdfLaTeX (Menu -> Compiler). That is Overleaf's default.
3. Main document is `pitch.tex`. Compiles to 2 pages.

Packages used are all in Overleaf's standard TeX Live: geometry, graphicx,
booktabs, microtype, xcolor, amsmath, float, hyperref.

## Figures

The three PDFs are vector output from `make_figures.py`, generated directly from
the run artifacts in `runs/stage2-002/` (not committed here - they live in the
project repo). Regenerate rather than hand-editing if any number changes:

    python make_figures.py        # run from the repo root

- `fig1_confound.pdf`  per-task mean score on failures vs successes, reported config
- `fig2_layers.pdf`    within-task AUC by layer, rounds 1 and 3
- `fig3_modes.pdf`     detection by failure mode, reported config vs layer 12 / round 3
