imdb_trace:
	python3 gen_plots.py --in_files data/IMDB_gaussian_tuned-glove_0_0_20_0.001.csv,data/IMDB_vmf_tuned-glove_0_0_20_0.001.csv,data/IMDB_logistic-gaussian_tuned-glove_0_0_20_0.001.csv --names 'Gaussian,von-MisesFisher,Logistic-Normal' --title 'IMDB Perplexity vs. Coherence' --plot_file plot.imdb.png

20news_trace:
	python3 gen_plots.py --in_files data/20news_gaussian_tuned-glove_0_0_20_0.001.csv,data/20news_vmf_tuned-glove_0_0_20_0.001.csv,data/20news_logistic-gaussian_tuned-glove_0_0_20_0.001.csv --names 'Gaussian,von-MisesFisher,Logistic-Normal' --title '20 News Perplexity vs. Coherence' --plot_file plot.20news.png

twitter_trace:
	python3 gen_plots.py --in_files data/twitter_gaussian_tuned-glove_0_0_20_0.001.csv,data/twitter_vmf_tuned-glove_0_0_20_0.001.csv,data/twitter_logistic-gaussian_tuned-glove_0_0_20_0.001.csv --names 'Gaussian,von-MisesFisher,Logistic-Normal' --title 'Twitter Perplexity vs. Coherence' --plot_file plot.twitter.png

nih_trace:
	python3 gen_plots.py --in_files data/nih_gaussian_tuned-glove_0_0_50_0.001.csv,data/nih_vmf_tuned-glove_0_0_50_0.001.csv,data/nih_logistic-gaussian_tuned-glove_0_0_50_0.001.csv --names 'Gaussian,von-MisesFisher,Logistic-Normal' --title 'NIH Perplexity vs. Coherence' --plot_file plot.nih.png

#imdb_vmf_l1_trace:
#	python3 gen_plots.py --in_files data/IMDB_vmf_tuned-glove_0_0_20_0.001.csv,data/IMDB_vmf_tuned-glove_0.05_0_20_0.001.csv,data/IMDB_vmf_tuned-glove_0.1_0_20_0.001.csv,data/IMDB_vmf_tuned-glove_0.2_0_20_0.001.csv --names 'Sparsity=0,Sparsity=0.05,Sparsity=0.1,Sparsity=0.2' --title 'IMDB Perplexity vs. Coherence' --plot_file plot.imdb.vmf.l1.png

all:	imdb_trace 20news_trace twitter_trace nih_trace
