#!/bin/sh
set -eu
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error slides.tex
xelatex -interaction=nonstopmode -halt-on-error slides.tex
