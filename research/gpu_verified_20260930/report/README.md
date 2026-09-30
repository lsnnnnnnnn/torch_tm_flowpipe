# English Beamer deck and Chinese verbatim script

> Publication context (2026-09-30): this deck and its evidence remain a dated reporting snapshot. New terminal records establish a 600-step complete-period prefix and a six-hour native QUAD timeout; see [current status](../STATUS.md). The exact QUAD ONNX model is now included in [the source snapshot](../source/benchmark/), while compiled binaries and large PT/SR assets remain external. Slides and their historical numerical evidence were not rewritten during publication. Only this README context note and its checksum were updated.

This bundle translates the September 29, 2026 progress deck into English. It preserves 42 slides: 27 presentation slides and 15 reference slides. The accompanying Chinese script follows slides 1–27 and includes optional Q&A prompts for slides 28–42. The English deck uses the same saved experimental evidence and numerical values; no experiments were rerun.

## Files

- `slides.pdf`: ready to present.
- `slides.tex`: editable, standalone Beamer source; tables and the QUAD timing chart remain native LaTeX.
- `chinese_verbatim_script.docx`: editable spoken Chinese script, keyed to slide number.
- `CHINESE_VERBATIM_SCRIPT.md`: plain-text copy of the script.
- `data/`, `configs/`, `evidence/`, `SOURCES.json`, `SOURCES.md`: the original evidence package with data, configurations, source paths, and SHA-256 hashes.

Use XeLaTeX and `slides.tex` as the main file in Overleaf or run `sh build.sh` with a TeX Live installation containing Beamer and pgfplots. The deck does not require external images, data files, a GPU, SSH, or model weights to compile. The historical `evidence/previous_slides_20260923.tex` is included as evidence and is not an input to this deck.

The seven complete ARCH configurations have five fresh-process timing runs per method. The full QUAD P3 and trig-reuse results are single completion runs, while failed or early-stopped attempts have separate statuses. Old native ACC widths fail a saved analytic enclosure check. Huan parity and strict P3 use different numerical contracts. The last archived native QUAD snapshot is 400/1000 steps at 20:19:53 China time on September 29; this is not a live status. The experimental task remains paused.

Original evidence reports retain paths from their source repository layouts. Use `SOURCES.json` to find each included copy. The bundle is for presentation and audit, not a complete solver environment: model weights, binaries, and large PT/SR artifacts remain on the server.
