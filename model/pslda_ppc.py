#!/usr/bin/env python
"""
    Runs PPC on a partial slda model I made and graphs them.
    
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
    lines = p.simulated_lines(global_params, local_params, observed_values, 300, 3)

    print len([a for a,b in scatterplot if a < b])

    first = lambda x: list(a[0] for a in x)
    second = lambda x: list(a[1] for a in x)

    import matplotlib.pyplot as plot
    # make graphs
    plot.figure(1)
    plot.grid(True)
    plot.scatter(first(scatterplot), second(scatterplot), 20, 'k', '.')
    plot.axis([0, 3.5, 0, 3.5])
    plot.plot([0, 3.5], [0, 3.5], ':')
    plot.xlabel(r'D(y,$\theta$)')
    plot.ylabel(r'D($y^{rep}$,$\theta$)')
    plot.title('Posterior Predictive Check Scatterplot')
    plot.savefig('slda-ppc-scatterplot.png')
    #plot.legend(('sample1','sample2'))

    plot.figure(2)
    plot.axis([-2, 2, -2, 2])
    plot.plot(first(lines[0]), second(lines[0]), 'k-')
    for p in lines[1]:
        plot.plot(first(p), second(p), 'b-.')
    plot.xlabel(r'$\eta^T E[\bar z]$')
    #plot.xlabel(r'Positive vs Negative words')
    plot.ylabel(r'Observed Rating')
    plot.title('Posterior Predictive Check Simulated Draws')
    plot.savefig('slda-ppc-lines.png')

