#!/usr/bin/env python

# topic 015 in ls models/lda_c_2011_10_16/final.* is about restaurants and dining

# I classified the 10k document collection and put those gammas in ccode/lda-c/classify10000-gamma.dat
# topic 015 in ls models/lda_c_2011_10_16/final.* is about restaurants and dining

# this will find the documents which have that topic
import jsondata

gamma_filename = 'models/lda_c_2011_10_16/final.gamma'
gamma_filename = 'ccode/lda-c/classify10000-gamma.dat'

gammas = []
with open(gamma_filename, 'r') as f:
    gammas = [[float(f) for f in g.split(' ')] for g in f.readlines()]

restaurant_gammas = [(i,g) for i,g in enumerate(gammas) if g[15] > 100.0]
restaurant_indices = [i for (i,g) in restaurant_gammas]



data_filename = 'data/lda/nytimes_10000_sparse_lda_2011_10_16.dat'
with open(data_filename, 'r') as f:
    docs = f.readlines()

restaurant_docs = [docs[i] for i in restaurant_indices]

sparse_docs = [[(int(e.split(':')[0]),int(e.split(':')[1])) for e in d.split(' ')[1:]] for d in restaurant_docs]

import pdb; pdb.set_trace()
jsondata.save('background.nyt_med.json', sparse_docs)
