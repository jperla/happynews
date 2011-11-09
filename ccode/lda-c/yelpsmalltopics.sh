#!/bin/sh

time ./lda est 0.1 40 settings.txt ../../data/lda/yelpsmall_nytimes_2000_lda_2011_10_16.dat ../../testmodel/testmodel ../../models/yelpsmall_nytimes_lda_c_2011_10_17

./topics.py ../../models/yelpsmall_nytimes_lda_c_2011_10_17/final.beta ../../data/yelp_lexicon_small.json 40
