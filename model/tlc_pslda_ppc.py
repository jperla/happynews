#!/usr/bin/env python
"""
    Runs PPC on a partial slda model I made and graphs them.
    
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import glob

import numpy as np

import jsondata
import ppc


if __name__=='__main__':
    s = 'midterm/mytlc-output-20-%s'

    name = 'tlc-pslda'

    eta = jsondata.read(glob.glob(s % 'eta*')[0])
    sigma_squared = jsondata.read(glob.glob(s % 'sigma_squared*')[0])[0]
    beta = jsondata.read(glob.glob(s % 'beta*')[0])
    phi = jsondata.read(glob.glob(s % 'phiC*')[0])
    # cut beta down to what we need
    Nd,Kc = phi[0].shape
    beta = beta[:Kc,:]

    # in comments, last phi is the sentiment data
    Ks = len(eta)
    phi = [p[:,-Ks:] for p in phi]

    print 'finished reading in params...'
    global_params = {'eta': eta, 'beta': beta, 'sigma_squared': sigma_squared}
    local_params = [{'phi': p} for p in phi]

    # get the data
    num_docs = 9994
    y = jsondata.read('data/yelp.labels.json')[:num_docs]

    # note, only y needs to be observed; not the words
    observed_values = [(i - 3.0) for i in y]
    print 'finished reading in docs...'

    p = ppc.YelpSentimentPartialSLDAPPC()

    scatterplot = p.scatterplot(global_params, local_params, observed_values)
    lines = p.simulated_lines(global_params, local_params, observed_values, 300, 5)

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
    plot.savefig(name + '-ppc-scatterplot.png')
    #plot.legend(('sample1','sample2'))

    plot.figure(2)
    plot.axis([-2, 2, -2, 2])
    for p in lines[1]:
        plot.plot(first(p), second(p), 'b-.')
    plot.plot(first(lines[0]), second(lines[0]), 'k-')
    plot.xlabel(r'$\eta^T E[\bar z]$')
    #plot.xlabel(r'Positive vs Negative words')
    plot.ylabel(r'Observed Rating')
    plot.title('Posterior Predictive Check Simulated Draws')
    plot.savefig(name + '-ppc-lines.png')

