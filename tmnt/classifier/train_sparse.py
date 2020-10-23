# codeing: utf-8

import argparse
import logging
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon.data import DataLoader
import gluonnlp as nlp
from sklearn.metrics import precision_recall_curve, average_precision_score

from tmnt.classifier.load_data import load_sparse_dataset
from tmnt.classifier.model import DANTextClassifier
from tmnt.utils.log_utils import logging_config


def get_args():
    parser = argparse.ArgumentParser(description='Train a (short) text classifier - via convolutional or other standard architecture')
    parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data')
    parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data', default=None)
    parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
    parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
    parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
    parser.add_argument('--lr',type=float, help='Learning rate', default=0.0005)
    parser.add_argument('--batch_size',type=int, help='Training batch size', default=128)
    parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.4)
    parser.add_argument('--emb_dropout', type=float, help='Dropout ratio for embedding layer', default=0.4)
    parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')
    parser.add_argument('--voc_size', type=int, default=2000, help='Vocab items')
    parser.add_argument('--max_length', type=int, default=64, help='Maximum length')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Embedding dimension (default = 50)')
    parser.add_argument('--hidden_dims', type=str, default='50', help='List of integers with dimensions for hidden layers (e.g. "50,50"). Default = 50 (single layer)')
    return parser.parse_args()
    
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

def train_classifier(vocab_size, emb_output_dim, transformer, data_train, data_val, data_test, ctx=mx.cpu()):

    data_train = gluon.data.SimpleDataset(data_train).transform(transformer)
    data_val   = gluon.data.SimpleDataset(data_val).transform(transformer)
    data_test  = gluon.data.SimpleDataset(data_test).transform(transformer)    
    
    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader   = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    test_dataloader  = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    emb_input_dim = vocab_size
    hidden_dims = [int(i) for i in args.hidden_dims.split(',')]
    model = DANTextClassifier(emb_input_dim, emb_output_dim, dropout = args.dropout, emb_dropout=args.emb_dropout,
                              dense_units = hidden_dims,
                              seq_length=args.max_length)

    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)  ## initialize model parameters on the context ctx
    
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    start_ap, start_acc = evaluate(model, val_dataloader)
    logging.info("Starting AP = {} Acc = {}".format(start_ap, start_acc))
    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, pair in enumerate(train_dataloader):
            data, label, mask = pair
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = model(data, mask)
                l = loss_fn(output, label).mean()
            l.backward()
            trainer.step(1)
            epoch_loss += l.asscalar()
        logging.info("Epoch [{}] loss = {}".format(epoch+1, epoch_loss))
        tr_ap, tr_acc = evaluate(model, train_dataloader)
        logging.info("TRAINING AP = {} Acc = {}".format(tr_ap, tr_acc))        
        val_ap, val_acc = evaluate(model, val_dataloader)
        logging.info("VALIDATION AP = {} Acc = {}".format(val_ap, val_acc))
    dev_ap, dev_acc = evaluate(model, test_dataloader)
    logging.info("***** Training complete. *****")
    logging.info("Test AP = {} Acc = {}".format(dev_ap, dev_acc))
        

def evaluate(model, dataloader, ctx=mx.cpu()):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    """
    acc = 0
    total_correct = 0
    total = 0
    all_scores = []
    all_labels = []
    for i, (data, label, mask) in enumerate(dataloader):
        out = model(data, mask)
        predictions = mx.nd.argmax(out, axis=1).astype('int32')
        for j in range(out.shape[0]):
            probs = mx.nd.softmax(out[j])
            lab = int(label[j].asscalar())
            all_scores.append(probs[1].asscalar())
            all_labels.append(lab)
            if probs[1] > probs[0] and lab == 1:
                total_correct += 1
            elif probs[1] < probs[0] and lab == 0:
                total_correct += 1
            total += 1
    acc = total_correct / float(total)
    ap = average_precision_score(all_labels, all_scores)
    return ap, acc
    

if __name__ == '__main__':
    args = get_args()
    logging_config(args.log_dir, 'train', level=logging.INFO, console_level=logging.INFO)
    train_dataset, val_dataset, test_dataset, transform = \
        load_sparse_dataset(args.train_file, args.val_file, args.test_file, voc_size=args.voc_size, max_length=args.max_length)
    ctx = mx.cpu()
    train_classifier(args.voc_size, args.embedding_dim, transform, train_dataset, val_dataset, test_dataset, ctx)
