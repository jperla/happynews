#!/usr/bin/env python
"""
    Functions useful in doing Posterior Predictive Checks of models.
    See Gelman et al. 1996.
    Uses graphlib and topiclib.
    
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import math
from itertools import izip

import numpy as np

import jsondata
import graphlib
import topiclib

class PosteriorPredictiveChecks(object):
    def __init__(self):
        pass

    def simulate(self, posterior, observed):
        """Accepts the posterior model parameters.  
            Also accepts the observed value, 
                but only to figure out the auxiliary variables 
                (like number of words to generate in this document).

            Returns a new simulated observation from the posterior e.g. using MCMC.
        """
        raise NotImplementedError
    
    def observed_norm(self, observed):
        """Accepts the observed data.
            Returns a real number.

            e.g. in Linear regression, return the observed y
        """
        raise NotImplementedError

    def posterior_norm(self, posterior):
        """Accepts the observed data.
            Returns a real number.

            e.g. in Linear regression, return the Ax+b predicted value without noise
        """
        raise NotImplementedError

    def discrepancy(self, posterior, observed):
        """Accepts the posterior model parameters, and observed values.

            These may be real observed values, or simulated.
            Returns a real number.

            e.g. in Linear regression, would be standardized residual
        """
        raise NotImplementedError

    def iterpost(self, global_params, local_params, observed_values):
        for (local, observed) in izip(local_params, observed_values):
            posterior = {}
            posterior.update(global_params)
            posterior.update(local)
            yield posterior, local, observed

    def scatterplot(self, global_params, local_params, observed_values):
        """Accepts global parameters dictionary.
            Also accepts a list of dictionaries of local parameters.
            Finally a list of observed values which should be same size as local parameters.

            Generates a list of 2-tuples (D(y, theta), D(yrep, theta)).
            Can be used to generate p-value graph to check model fit.
        """
        output = []
        for posterior, local, observed in self.iterpost(global_params, local_params, observed_values):
            Dy    = self.discrepancy(posterior, observed)
            Dyrep = self.discrepancy(posterior, self.simulate(posterior, observed))
            output.append((Dy, Dyrep))
        return list(sorted(output))

    def min_discrepancy_histogram(self, global_params, local_params, observed_values):
        # NOTE: I misunderstood this graph, This is wrong.
        # Computing this is much harder, and requires fitting the simulation.
        """Accepts global parameters dictionary.
            Also accepts a list of dictionaries of local parameters.
            Finally a list of observed values which should be same size as local parameters.

            Generates a histogram of discrepancies for Dmin
            See Gelman et al. 1996
            
            Returns 2-tuple of (float, [floats]). 
            Representing the minimum discrepancy of observation to its posterior,
                and a list of the simulated discrepancies from that same minimum posterior.
        """
        # first find Dmin
        minimum_discrepancy = float('inf')
        minimum = (None, None)
        for posterior, local, observed in self.iterpost(global_params, local_params, observed_values):
            d = self.discrepancy(posterior, observed)
            if d < minimum_discrepancy:
                minimum_discrepancy = d
                minimum = (local, observed)

        # now simulate 1000 tosses
        simulations = 1000
        min_posterior = {}
        min_posterior.update(global_params)
        min_posterior.update(minimum[0])
        simulated = [self.simulate(min_posterior, minimum[1]) for i in xrange(simulations)]

        # now return all discrepancies
        return (minimum_discrepancy, [self.discrepancy(min_posterior, s) for s in simulated])

    def simulated_lines(self, global_params, local_params, observed_values):
        """Accepts global parameters dictionary.
            Also accepts a list of dictionaries of local parameters.
            Finally a list of observed values which should be same size as local parameters.

            See Gelman et al. 1996
            Generates a series of plots which can be graphed to see 
                over which types of data the model may go wrong.
        """
        def generate_linegraph(global_params, local_params, observed_values):
            line = [(self.posterior_norm(p), self.observed_norm(o))
                        for p,l,o in self.iterpost(global_params, local_params, observed_values)]
            return list(sorted(line))

        num_lines = 9
        real_line = generate_linegraph(global_params, local_params, observed_values)

        simulated = [[self.simulate(p, o)
                        for p,l,o in self.iterpost(global_params, local_params, observed_values)]
                            for i in xrange(num_lines)]

        simulated_lines = [generate_linegraph(global_params, local_params, s) for s in simulated]
        return (real_line, simulated_lines)
        

class TLCPPC(PosteriorPredictiveChecks):
    def simulate(self, posterior, observed):
        """Accepts posterior, which is dictionary of phi, beta, eta, sigma squared.
            Observed is a sparse vector of word, list of (word int,count) 2-tuples.

            Returns observation in same sparse vector type.
        """
        # number of words to generate
        No = sum(o[1] for o in observed) 

        beta = posterior['beta']
        phi = posterior['phi']
        N,K = phi.shape
        assert No == N
        assert K == beta.shape[0]

        topics = [np.random.multinomial(1, phi) for n in xrange(N)]
        words = [np.random.multinomial(1, beta[t]) for t in topics]
    
        d = {}
        for w in words:
            if w in d:
                d[w] += 1
            else:
                d[w] = 1

        return list(sorted((w,c) for w,c in d))
        
    def posterior_norm(self, posterior):
        """Accepts posterior, which is dictionary of phi, beta, eta, sigma squared.
            Returns real number.
        """
        eta = posterior['eta']
        phi = posterior['phi']

        # partial slda, so only use first few topics
        N,K = phi.shape
        Ks = len(eta)
        partial_phi = phi[:,:Ks]

        EZ = np.sum(partial_phi, axis=0) / N
        assert len(EZ) == len(eta)
        return np.dot(eta, EZ)

    def observed_norm(self, observed):
        """How to generate a statistic based on text is 
            probably different for each application.
        """
        raise NotImplementedError

    def discrepancy(self, posterior, observed):
        """Accepts posterior, which is dictionary of phi, beta, eta, sigma squared.
            Observed is a sparse vector of word, list of (word int,count) 2-tuples.

            Returns a real number.

            Just uses observed and posterior norm divided by sigma squared.
        """
        #TODO: jperla: maybe can generalize, sigma is a def standardizer() ?
        s = np.sqrt(posterior['sigma_squared'])
        return abs(self.posterior_norm(posterior) - self.observed_norm(observed)) / s


vocab = dict((w,i) for i,w in enumerate(jsondata.read('../data/nytimes_med_common_vocab.json')))
pos = jsondata.read('../data/liu_pos_words.json')
neg = jsondata.read('../data/liu_neg_words.json')

posi = set([vocab[w] for w in pos if w in vocab])
negi = set([vocab[w] for w in neg if w in vocab])

class YelpSentimentPartialSLDAPPC(TLCPPC):
    def simulate(self, posterior, observed):
        """Accepts posterior vars which include phi and eta.
            As well as observed value which is just a real number.
            Returns a new observation.

            Observation is from a normal from expected mean, like regression.
        """
        s = np.sqrt(posterior['sigma_squared'])
        mean = self.posterior_norm(posterior)
        return np.random.normal(mean, s)

    def observed_norm(self, observed):
        """Accepts a real value between -2 and 2.
            Returns real number between -2 and 2.

            Itself.  This is just like regression.
        """
        return observed

class YelpSentimentTLCPPC(TLCPPC):
    def observed_norm(self, observed):
        """Accepts a sparse vector of word, list of (word int,count) 2-tuples.
            Returns real number between -2 and 2.
        """
        numpos = 0
        numneg = 0
        for n,word,count in topiclib.iterwords(observed):
            if word in posi:
                numpos += 1
            if word in negi:
                numneg += 1

        ratio = 1.0 
        normratio = 0.0
        if numpos == 0 and numneg == 0:
            return 0.0
        elif numneg == 0:
            return 2.0
        elif numpos == 0:
            return -2.0
        else:
            if numpos >= numneg:
                ratio = float(numpos) / numneg
                normratio = (ratio - 1)
            else:
                ratio = -1.0 * float(numneg) / numpos
                normratio = (ratio + 1)
            o = graphlib.logistic_sigmoid(normratio)
            return (4 * o) - 2 # norm to -2 to 2



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

    ppc = YelpSentimentPartialSLDAPPC()

    scatterplot = ppc.scatterplot(global_params, local_params, observed_values)
    lines = ppc.simulated_lines(global_params, local_params, observed_values)

