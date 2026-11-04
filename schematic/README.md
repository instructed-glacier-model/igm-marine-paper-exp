# Schematic (Fig. 1)

Cross-section of a marine-terminated glacier with the notation of Sect. 2 of the paper (TikZ).

```bash
pdflatex fig_01_schematic.tex      # needs TikZ, bm and the standalone class; writes fig_01_schematic.pdf
```

`fig_01_schematic.tex` sets the fonts, colours and notation macros of the paper and inputs `schematic_body.tex`, the
drawing. The built `fig_01_schematic.pdf` is Fig. 1.
