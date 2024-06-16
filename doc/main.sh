#!/bin/bash
pandoc --template ./eisvogel.tex main.md --number-sections --listings --pdf-engine=lualatex	-o ./pdfs/Zusammenfassung_MAA.pdf
