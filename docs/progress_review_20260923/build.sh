#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p .build
for document in report slides; do
 xelatex -interaction=nonstopmode -halt-on-error -output-directory=.build "$document.tex" > ".build/${document}_pass1.txt"
 xelatex -interaction=nonstopmode -halt-on-error -output-directory=.build "$document.tex" > ".build/${document}_pass2.txt"
 cp ".build/$document.pdf" "$document.pdf"
done
