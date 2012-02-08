#!/usr/bin/env python
"""
    Functions useful in doing Posterior Predictive Checks of models.
    See Gelman et al. 1996.
    Uses graphlib and topiclib.
    
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import ppc

import numpy as np

import jsondata


if __name__=='__main__':
    s = 'medsldamodel/med-slda.final-%s.dat'

    eta = np.array(jsondata.read(s % 'eta'))
    beta = np.array(jsondata.read(s % 'beta'))
    phi = [np.array(p) for p in jsondata.read(s % 'phi')]
    sigma_squared = jsondata.read(s % 'sigma_squared')[0]

    print 'finished reading in params...'
    global_params = {'eta': eta, 'beta': beta, 'sigma_squared': sigma_squared}
    local_params = [{'phi': p} for p in phi]

    # get the data
    num_docs = 1000
    #labeled_documents = jsondata.read('data/yelp.nyt_med.json')[:num_docs]
    y = jsondata.read('data/yelp.labels.json')[:num_docs]

    #// filter out documents with no words
    #all_data = [(l,y) for l,y in izip(labeled_documents,y) if len(l) > 0]
    #print 'num docs: ' + len(all_data)
    #labeled_documents = [a[0] for a in all_data]

    # norm this to around 2.0
    # so that things without sentimental topics end up being neutral!
    #y = [(a[1] - 3.0) for a in all_data]
    print 'finished reading in docs...'

    # note, only y needs to be observed; not the words
    #observed_values = [y for l,y in izip(labeled_documents, y)]
    observed_values = [(i - 3.0) for i in y]

    print 'finished reading in docs...'

    p = ppc.YelpSentimentPartialSLDAPPC()

    scatterplot = p.scatterplot(global_params, local_params, observed_values)
    lines = p.simulated_lines(global_params, local_params, observed_values)

