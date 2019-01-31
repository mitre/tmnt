# Topic Modeling Neural Toolkit

Topic modeling with Neural Variational Models

Project layout using ./tmnt for general source code and ./bin for command-line executable scripts.

Run this via something like the following where the argument to `train_dir` is a directory containing a
single document in each file.

```
 > cd bin
 > python3 train_bow_vae.py --train_dir ./input-data --file_pat '*.txt' --epochs 400 --n_latent 32 --batch_size 32 --lr 0.00005
```
