#!/bin/sh

#time ./lda est 0.1 20 settings.txt ../../data/yelpsmall_nytimes_5000_lda_2011_10_16.dat random ../../models/yelpsmall_emotive_lda_c_2011_10_18

./topics.py ../../models/emotive_lda_c_2011_10_17/final.beta ../../data/yelp_lexicon_med.json 30

#./topics.py ../../models/yelpsmall_emotive_lda_c_2011_10_17/final.beta ../../data/yelp_lexicon_small.json 30

#./topics.py ../../models/yelpsmall_emotive_lda_c_2011_10_18/final.beta ../../data/yelp_lexicon_small.json 30
