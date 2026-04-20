#!/bin/bash
# ICGNC 论文编译脚本
# 使用方法: bash build.sh

TEXLIVE_DIR="/home/dataset-assist-0/gxr/conda-pkgs/texlive2024"
export PATH="$TEXLIVE_DIR/bin/x86_64-linux:$PATH"
export TEXINPUTS="./styles//:./:"

cd "$(dirname "$0")"

echo "=== Compiling atom_language_paper.tex ==="
pdflatex -interaction=nonstopmode atom_language_paper.tex
pdflatex -interaction=nonstopmode atom_language_paper.tex

echo ""
echo "=== Done ==="
ls -lh atom_language_paper.pdf
