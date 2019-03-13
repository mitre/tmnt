# Topic Modeling Neural Toolkit

Topic modeling with Neural Variational Models

Project layout using ./tmnt for general source code and ./bin for command-line executable scripts.

Run this via something like the following where the argument to `train_dir` is a directory containing a
single document in each file.

```
python3 train_bow_vae.py --train_dir ./input-data --file_pat '*.txt' --epochs 120 --n_latent 20 --batch_size 200 --lr 0.001 --latent_distribution vmf
```

To train using the example data in sparse vector (libSVM) format, use the flags `tr_vec_file`, `tst_vec_file` and `vocab_file`, e.g. run:

```
mkdir _model_dir
python3 train_bow_vae.py --tr_vec_file ./data/train.2.vec --tst_vec_file ./data/test.2.vec --vocab_file ./data/train.2.vocab --n_latent 20 --lr 0.001 --batch_size 200 --epochs 120 --model_dir ./_model_dir
```

To load a saved model via the API, do:

```
python3 -i tmnt/bow_vae/runtime.py
infer = BowNTMInference('_model_dir/model.params','_model_dir/model.specs', '_model_dir/vocab.json')
top_k_terms = infer.get_top_k_words_per_topic(10) # top 10 words per topic
encodings = infer._test_inference_on_directory('<some directory of .txt files>', '*.txt') ## encode documents 
```

To evaluate a trained model against a dataset, run the `evaluate.py` script, e.g.:

```
python3 evaluate.py --model_dir model --eval_file data/test.feat.vec --train_file data/train.feat.vec --vocab_file data/20news.vocab
```


