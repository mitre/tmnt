# Topic Modeling Neural Toolkit

The following README contains very minimal documentation to get started.  See full documentation
for details, examples and API documentation.

## Installation

After cloning the repository, TMNT should be installed as a package locally by running:

```
python setup.py develop
```

## Training a model with the 20 news data

The commonly used 20 News Dataset is included in a sparse vector representation for testing and for purpose of example.
It's possible to train a topic model on the data as follows:

```
mkdir -p _model_dir
mkdir -p _experiments
python bin/train_model.py --tr_vec_file ./data/train.2.vec --tst_vec_file ./data/test.2.vec --vocab_file ./data/train.2.vocab --save_dir ./_experiments/ --model_dir ./_model_dir_final/ --config ./examples/train_model/model.config --trace_file ./TRACE.csv 
```

The resulting model will be placed in the `_model_dir` directory.

## Using a trained model to encode texts

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

## Evaluation

To assess a trained model against a dataset, run the `evaluate.py` script as below.  Note that depending
on your backend for handling visualization/plotting you may need to run `pythonw` instead of `python` when
executing the script.

```
pythonw bin/evaluate.py --train_file ./data/train.2.vec --eval_file ./data/test.2.vec \
                        --vocab_file ./data/train.2.vocab --model_dir ./_model_dir/ \
			--num_topics 20 --plot_file ./p1.png
```

## Preparing data

```
python bin/prep_data.py --train_dir ./input-data --file_pat '*.txt' ...
```




