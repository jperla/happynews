#!/bin/sh

#rm -rf ../../models/yelp_slda_2011_10_17/

#time ./slda --fit --docs ../../data/lda/yelp_10000_sparse_lda_2011_10_16.dat / --resp ../../data/lda/yelp_10000_annotations_slda_2011_10_16.dat --vocab ../../data/nytimes_med_common_vocab.json --modeldir ../../models/yelp_slda_2011_10_17/ --type slda --vb 0 --ntopics 50

#../lda-c/topics.py ../../testbeta.dat ../../data/nytimes_med_common_vocab.json 25


time ./slda --infer --docs ../../data/nytimes_2000_foreign_desk_2011_10_25.dat  --vocab ../../data/nytimes_med_common_vocab.json --infdir ../../models/nytimes_slda_infer_from_yelp_2011_10_25/ --modeldir ../../models/yelp_slda_2011_10_16/final-model/ --type lda --vb 0 --ntopics 50
