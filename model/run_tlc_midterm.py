#!/usr/bin/env python
import numpy as np

import jsondata
import tlc

from itertools import izip

if __name__=='__main__':
    num_labeled = 9994

    cut_short = 100000000000

    # use my tlc synthetically generated dataset
    documents = jsondata.read('data/documents.dc.nyt.json')[:cut_short]
    comments = jsondata.read('data/comments.dc.nyt.json')[:cut_short]
    labeled_documents = jsondata.read('data/yelp.nyt_med.json')[:num_labeled][:cut_short]
    background = jsondata.read('data/background.nyt_med.json')[:cut_short]

    y = jsondata.read('data/yelp.labels.json')[:num_labeled][:cut_short]
    y = [(i - 3.0) for i in y] # center around 0

    real_data = (documents, comments, labeled_documents, background, y)

    var = tlc.TLCVars(real_data, Ku=25, Ks=5, Kb=25)
    var.eta = np.array([3.0, 1.5, 0.5, -1.5, -3.0])

    try:
        output = tlc.run_tlc(var)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

