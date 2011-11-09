#!/bin/sh

#time ./lda est 0.1 40 settings.txt ../../data/lda/nytimes_2000_sparse_lda_2011_10_16.dat ../../testmodel/testmodel ../../models/yelptopics_nytimes_lda_c_2011_10_17

./topics.py ../../models/yelptopics_nytimes_lda_c_2011_10_17/final.beta ../../data/nytimes_med_common_vocab.json 40
