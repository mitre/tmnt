
ntm_arch:
	pdflatex ntm_arch1.tex
	convert -density 300 ntm_arch1.pdf -quality 100 -sharpen 0x1.0 ntm_arch1.png

tmnt_arch:
	pdflatex tmnt_main_arch.tex
	convert -density 300 tmnt_main_arch.pdf -quality 100 -sharpen 0x1.0 tmnt_main.png

all_bib:
	pdflatex all_bib.tex
	bibtex all_bib
	pdflatex all_bib.tex
	pdflatex all_bib.tex


