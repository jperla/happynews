#!/usr/bin/env python
from itertools import izip

import jsondata
import partial_slda

if __name__=='__main__':
    num_docs = 10000

    # use my big generated dataset
    labeled_documents = jsondata.read('data/yelp.nyt_med.json')[:num_docs]
    y = jsondata.read('data/yelp.labels.json')[:num_docs]

    #// filter out documents with no words
    all_data = [(l,y) for l,y in izip(labeled_documents,y) if len(l) > 0]
    print len(all_data)
    labeled_documents = [a[0] for a in all_data]

    # norm this to around 2.0
    # so that things without sentimental topics end up being neutral!
    y = [(a[1] - 3.0) for a in all_data]

    real_data = (labeled_documents, y)

    var = partial_slda.PartialSupervisedLDAVars(real_data, Ks=5, Kb=15)

    try:
        output = partial_slda.run_partial_slda(var)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

