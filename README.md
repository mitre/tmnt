# Topic Modeling Neural Toolkit

The following README contains very minimal documentation to get started.  Site documentation is underway
and will be made accessible shortly.

## Installation

After cloning the repository, TMNT should be installed as a package locally by running:

```
python setup.py develop
```

## Training a model with the 20 news data

The commonly used 20 News Dataset is included in a sparse vector representation for testing and for purpose of example.
It's possible to train a topic model on the data as follows:

```
mkdir _model_dir
python bin/train_bow_vae.py --tr_vec_file ./data/train.2.vec --tst_vec_file ./data/test.2.vec --vocab_file ./data/train.2.vocab --n_latent 20 --lr 0.001 --batch_size 200 --epochs 40 --model_dir ./_model_dir 
```

The resulting model will be placed in the `_model_dir` directory.

## Using a trained model to 

To load a saved model via the API, do:

```
python -i tmnt/bow_vae/runtime.py
infer = BowNTMInference('_model_dir/model.params','_model_dir/model.specs', '_model_dir/vocab.json')
top_k_terms = infer.get_top_k_words_per_topic(10) # top 10 words per topic
encodings = infer._test_inference_on_directory('<some directory of .txt files>', '*.txt') ## encode documents

# get encodings for a list of token lists
encodings2 = infer.encode_texts([['first', 'document', 'tokenized', 'about', 'guns', 'weapons', 'defense'],
	                         ['second', 'document', 'about', 'nasa', 'orbit', 'rocket']])
```

To assess a trained model against a dataset, run the `evaluate.py` script as below.  Note that depending
on your backend for handling visualization/plotting you may need to run `pythonw` instead of `python` when
executing the script.

```
pythonw bin/evaluate.py --train_file ./data/train.2.vec --eval_file ./data/test.2.vec \
                        --vocab_file ./data/train.2.vocab --model_dir ./_model_dir/ \
			--num_topics 20 --plot_file ./p1.png
```

## Building a model from raw text files

Run this via something like the following where the argument to `train_dir` is a directory containing a
single document in each file.

```
python bin/train_bow_vae.py --train_dir ./input-data --file_pat '*.txt' --epochs 120 --n_latent 20 --batch_size 200 --lr 0.001 --latent_distribution vmf
```




