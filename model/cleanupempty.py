#!/usr/bin/env python
import jsondata

num_docs = 10000
labeled_documents = jsondata.read('data/yelp.nyt_med.json')[:num_docs]
for i,l in enumerate(labeled_documents):
    if len(l) == 0:
        print i
