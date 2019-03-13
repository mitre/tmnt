
ntm_arch:
	pdflatex ntm_arch1.tex
	convert -density 300 ntm_arch1.pdf -quality 100 -sharpen 0x1.0 ntm_arch1.png

tmnt_arch:
	pdflatex tmnt_main_arch.tex
	convert -density 300 tmnt_main_arch.pdf -quality 100 -sharpen 0x1.0 tmnt_main.png

tmnt_cov_arch:
	pdflatex tmnt_covariate_arch.tex
	convert -density 300 tmnt_covariate_arch.pdf -quality 100 -sharpen 0x1.0 tmnt_covariate.png

results:
	pdflatex results_overview.tex
	convert -density 300 results_overview.pdf -quality 100 -sharpen 0x1.0 results_1.png

formulae:
	pdflatex eval_formulae.tex
	convert -density 300 eval_formulae.pdf -quality 100 -sharpen 0x1.0 eval_formulae.png

topic_words:
	pdflatex topic_words.tex
	convert -density 300 topic_words.pdf -quality 100 -sharpen 0x1.0 topic_words.png
	pdflatex topic_words_imdb.tex
	convert -density 300 topic_words_imdb.pdf -quality 100 -sharpen 0x1.0 topic_words_imdb.png

all_bib:
	pdflatex all_bib.tex
	bibtex all_bib
	pdflatex all_bib.tex
	pdflatex all_bib.tex
	convert -density 300 all_bib.pdf -quality 100 -sharpen 0x1.0 all_bib.png

clean:
	rm *.blg *.bbl *.aux *.dvi *.aux *.pdf *.png *.log rm *~
