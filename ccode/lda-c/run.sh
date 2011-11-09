#!/bin/sh

#time ./lda est 0.1 40 settings.txt /Users/josephperla/projects/projects/happynews/data/nytimes_2000_sparse_lda_2011_10_17.dat random /Users/josephperla/projects/projects/happynews/models/lda_c_2011_10_17

./topics.py ../../models/lda_c_2011_10_16/final.beta ../../data/nytimes_med_common_vocab.json 25
