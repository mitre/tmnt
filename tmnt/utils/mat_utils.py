# coding: utf-8
"""
Copyright (c) 2019 The MITRE Corporation.
"""


import numpy as np
import io

__all__ = ['export_sparse_matrix', 'export_vocab']

def export_sparse_matrix(mat, ofile, label=-1):
    data = mat.data.asnumpy()
    indices = mat.indices.asnumpy()
    ptrs = mat.indptr.asnumpy()
    with io.open(ofile, 'w') as fp:
        for i in range(len(ptrs)-1):
            p0 = ptrs[i]
            p1 = ptrs[i+1]
            fp.write(str(label))
            vls = data[p0:p1]
            inds = indices[p0:p1]
            for i in range(len(inds)):
                fp.write(' ')
                fp.write(str(inds[i]))
                fp.write(':')
                fp.write(str(vls[i]))
            fp.write('\n')

def export_vocab(vocab, ofile):
    with io.open(ofile, 'w') as fp:
        for i in range(len(vocab.idx_to_token)):
            fp.write(vocab.idx_to_token[i])
            fp.write('\n')
            
