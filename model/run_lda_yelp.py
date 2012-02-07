#!/usr/bin/env python
import jsondata

import lda

if __name__=='__main__':
    labeled_documents = jsondata.read('data/yelp.nyt_med.json')[:100]
    # filter out documents with no words
    labeled_documents = [l for l in labeled_documents if len(l) > 0]

    var = lda.LDAVars(labeled_documents, K=20)
    try:
        output = lda.run_lda(var)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

