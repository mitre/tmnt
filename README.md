# Topic Modeling Neural Toolkit

The following README contains very minimal documentation to get started.  See full documentation
for details, examples and API documentation.

## Installation

### Getting TMNT

TMNT can be obtained via GitLab:

```
git clone https://gitlab.mitre.org/milu/tmnt.git
```

Note that the test data provided in the TMNT archive uses `git lfs`, see details here for installing
and setting up in your git environment: `https://git-lfs.github.com`

### Dependencies and Environment

TMNT is easiest to use by installing all necessary dependencies with Conda (Miniconda or Anaconda). If
Conda is not installed, please install by grabbing the necessary install script from:

https://docs.conda.io/en/latest/miniconda.html

Once Conda is installed properly, install a new conda environment for TMNT as follows:


```
conda create --name TMNT pip numpy==1.16.4
```

For some platforms, it may be preferred to install the necessary
C compiler/environment via conda by adding the `gxx_linux-64`
and `gcc_linux-64` targets:

```
conda create --name TMNT pip gxx_linux-64 gcc_linux-64 numpy==1.16.4
```

Then, from the tmnt top-level directory, execute


```
conda activate TMNT
pip install -r requirements.txt
```

Finally, TMNT should be installed as a package locally by running:

```
python setup.py develop
```

## Training a model with the 20 news data

The commonly used 20 News Dataset is included in a sparse vector representation for testing and for purpose of example.
It's possible to train a topic model on the data as follows:

```
mkdir -p _model_dir
mkdir -p _experiments
python bin/train_model.py --tr_vec_file ./data/train.2.vec --tst_vec_file ./data/test.2.vec --vocab_file ./data/train.2.vocab --save_dir ./_experiments/ --model_dir ./_model_dir/ --config ./examples/train_model/model.config --trace_file ./TRACE.csv 
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
on your backend for handling visualization/plotting you may need to run `python` instead of `python` when
executing the script.

```
python bin/evaluate.py --train_file ./data/train.2.vec --test_file ./data/test.2.vec \
                        --vocab_file ./data/train.2.vocab --model_dir ./_model_dir/ 
			
```

## Preparing data

There are two input formats (currently).  The first assumes a single JSON object per line in a file.  The value of the key 'text' will
be used as the document string.  All other fields are ignored. So, for example:

```
{"id": "1052322266514673664", "text": "This is the text of one of the documents in the corpus."}
{"id": "1052322266514673664", "text": "This is the text of another of the documents in the corpus."}
....
```

Two directories of such files should be provided, one for training and one for test.  Assuming the files end with `.json` extensions, the
following example invocation would prepare the data for the training and test sets, creating vector representations with a vocabulary
size of 2000.  Note that this script uses the built in pre-processing which tokenizes, downcases and removes common English stopwords.

```
python bin/prepare_corpus.py --vocab_size 2000 --file_pat '*.json' --tr_input_dir ./train-json-files/ --tst_input_dir ./test-json-files/ --tr_vec_file ./train.2k.vec --vocab_file ./2k.vocab  --tst_vec_file ./test.2k.vec 
```

Another input format assumes directories for training and test sets, where each file is a separate plain text document. This should be
invoked by adding the `--txt_mode` option:


python bin/prepare_corpus.py --vocab_size 2000 --file_pat '*.txt' --tr_input_dir ./train-txt-files/ --tst_input_dir ./test-txt-files/ --tr_vec_file ./train.2k.vec --vocab_file ./2k.vocab  --tst_vec_file ./test.2k.vec --txt_mode

