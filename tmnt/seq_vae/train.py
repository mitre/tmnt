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
import datetime
from tmnt.seq_vae.seq_data_loader import load_dataset, BasicTransform
from tmnt.seq_vae.seq_models import SeqVAE
from tmnt.utils.log_utils import logging_config


parser = argparse.ArgumentParser(description='Train a Convolutional Variational AutoEncoder on Context-aware Encodings')

parser.add_argument('--train_file', type=str, help='Directory containing a RecordIO file representing the input data')
parser.add_argument('--val_file', type=str, help='Directory containing a RecordIO file representing the validation data')
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.00001)
parser.add_argument('--gpu',type=int, help='GPU device ids', default=-1)
parser.add_argument('--save_dir',type=str, help='Target directory for trained model parameters', default='cvae_model_out')
parser.add_argument('--batch_size',type=int, help='Training batch size', default=8)
parser.add_argument('--num_filters',type=int, help='Number of filters in first layer (each subsequent layer uses x2 filters)', default=64)
parser.add_argument('--decrease_filters', action='store_true')
parser.add_argument('--increase_filters', action='store_true')
parser.add_argument('--latent_dim',type=int, help='Encoder dimensionality', default=256)
parser.add_argument('--kld_wt',type=float, help='Weight of the KL divergence term in variational loss', default=1.0)
parser.add_argument('--sent_size',type=int, help='Fixed/max length of sentence (zero padded); should be power of 2', default=16)
parser.add_argument('--batch_report_freq', type=int, help='Frequency to report batch stats during training', default=100)
parser.add_argument('--save_model_freq', type=int, help='Number of epochs to save intermediate model', default=100)
parser.add_argument('--use_hotel_data', action='store_true', help='Special test using hotel review data')
parser.add_argument('--embedding_source', type=str, default='glove.twitter.27B.100d', help='Pre-trained embedding source name')
#parser.add_argument('--learn_embedding', action='store_true', help='Assume straight token input and learn embeddings via training')
parser.add_argument('--ngram_buckets', type=int, help='Number of ngram buckets in the pre-embedding file')

args = parser.parse_args()
i_dt = datetime.datetime.now()
train_out_dir = '{}/train_{}_{}_{}_{}_{}_{}'.format(args.save_dir,i_dt.year,i_dt.month,i_dt.day,i_dt.hour,i_dt.minute,i_dt.second)
logging_config(folder=train_out_dir, name='train_cvae', level=logging.INFO, no_console=False)
logging.info(args)


def get_model(emb_dim, vocab_dim, ctx):
    model = SeqVAE(emb_dim, vocab_dim, n_latent=64, model_ctx=ctx)
    return model
    

def train_cvae(vocabulary, data_transform, data_train, data_val, report_fn, ctx=mx.cpu()):

    data_train = gluon.data.SimpleDataset(data_train).transform(data_transform)
    #data_val   = gluon.data.SimpleDataset(data_val).transform(data_transform)

    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    #val_dataloader   = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False)

    vocab_dim, emb_dim = vocabulary.embedding.idx_to_vec.shape
    model = get_model(emb_dim, vocab_dim, ctx)
    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)  ## initialize model parameters on the context ctx

    model.embedding.weight.set_data(vocab.embedding.idx_to_vec) ## set the embedding layer parameters to pre-trained embedding

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})

    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, data in enumerate(train_dataloader):
            data = data.as_in_context(ctx)
            with mx.autograd.record():
                loss, predictions = model(data)
                loss.backward()
            ls = loss.sum()
            trainer.step(data.shape[0])
            epoch_loss += ls.asscalar()
            if i % 20 == 0:
                if report_fn:
                    mx.nd.waitall()
                    report_fn(data, predictions)
        logging.info("Epoch loss = {}".format(epoch_loss))
        if report_fn:
            mx.nd.waitall()
            report_fn(data, predictions)




def get_report_reconstruct_data_fn(vocab, pad_id=0):
    def report_reconstruct_data_fn(data, predictions):
        reconstructed_sent_ids = mx.nd.argmax(predictions[0],1).asnumpy() ## get the first item of batch and arg_max over vocab size
        input_sent_ids = data[0].asnumpy()
        rec_sent = [vocab.idx_to_token[int(i)] for i in reconstructed_sent_ids if i != pad_id]   # remove <PAD> token from rendering
        in_sent = [vocab.idx_to_token[int(i)] for i in input_sent_ids if i != pad_id]
        in_ids = [str(i) for i in input_sent_ids]
        logging.info("---------- Reconstruction Output/Comparison --------")
        logging.info("Input Ids = {}".format(' '.join(in_ids)))
        logging.info("Input = {}".format(' '.join(in_sent)))
        logging.info("Reconstructed = {}".format(' '.join(rec_sent)))
    return report_reconstruct_data_fn


if __name__ == '__main__':

    train_dataset, val_dataset, vocab = load_dataset(args.train_file, args.val_file, max_len=32)

    glove_twitter = nlp.embedding.create('glove', source=args.embedding_source)
    vocab.set_embedding(glove_twitter)

    _, emb_size = vocab.embedding.idx_to_vec.shape
    pad_id = vocab['<pad>']

    transform = BasicTransform(max_len=32, pad_id=pad_id)

    ## set embeddings to random for out of vocab items
    oov_items = 0
    for word in vocab.embedding._idx_to_token:
        if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
            oov_items += 1
            vocab.embedding[word] = mx.nd.random.normal(0.0, 0.1, emb_size)

    report_fn = get_report_reconstruct_data_fn(vocab, pad_id)
    logging.info("** There are {} out of vocab items **".format(oov_items))
    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu() 
    train_cvae(vocab, transform, train_dataset, None, report_fn, ctx)


    
    
