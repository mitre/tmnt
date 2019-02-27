# Topic Modeling Neural Toolkit

Topic modeling with Neural Variational Models

Project layout using ./tmnt for general source code and ./bin for command-line executable scripts.

Run this via something like the following where the argument to `train_dir` is a directory containing a
single document in each file.

```
python3 train_bow_vae.py --train_dir ./input-data --file_pat '*.txt' --epochs 400 --n_latent 32 --batch_size 32 --lr 0.00005
```

To train using the example data in sparse vector (libSVM) format, run:

```
mkdir _model_dir
python3 train_bow_vae.py --tr_vec_file ./data/train.2.vec --tst_vec_file ./data/test.2.vec --vocab_file ./data/train.2.vocab --n_latent 20 --lr 0.001 --batch_size 64 --epochs 60 --model_dir ./_model_dir
```

To load a saved model via the API, do:

```
python3 -i tmnt/bow_vae/runtime.py
infer = BowNTMInference('_model_dir/model.params','_model_dir/model.specs', '_model_dir/vocab.json')
top_k_terms = infer.get_top_k_words_per_topic(10) # top 10 words per topic
encodings = infer._test_inference_on_directory('<some directory of .txt files>', '*.txt') ## encode documents 
```

To evaluate a model, run:

```
python3 evaluate.py --model_dir model --eval_file data/test.feat.vec --train_file data/train.feat.vec --vocab_file data/20news.vocab
```

# Experimental Rubric

Below provides info and outline for experiments prior to Tech Talk on 03/14/2019

## Hyper-parameter Settings

### Learning Rate
When using the `gaussian` or `logistic_gaussian` distribution the recommended default learning rate is in the range: `0.0001` to `0.001`.

When using the `vmf` latent distribution, it is best to use a higher learning rate ranging from `0.01` to `0.1`

### Batch Size
Batch sizes should be in the range of 128 to 256 in most cases.

### Number of Epochs

Generally training for 100 to 200 epochs is sufficient.

### Experimental factors

factor | values
-------| ------
dataset | 20news, IMDB, Yahoo
latent dims | 20, 50, 100, 200
latent distribution | gaussian, logistic_gaussian, vmf
pre trained embeddings | fixed-glove, tuned-glove, none
L1 regularizer (target sparsity) | 0, 0.2, 0.4, 0.75
WETC regularizer | 0, 10, 50, 100

