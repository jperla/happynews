"""
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import json

import jsondata

def p_of_feature_given_label(pd, label, fname):
    """Accepts NLTK probability distribution, label string, and feature name string.
        Returns float of probability of the feature given label on pd.
    """
    return pd[label, fname].prob(True)

def maxratio(pd, fname):
    """Accepts NLTK probability distribution and feature name string.
        Calculates biggest ratio of likelihood given one label vs the other.
        Assumes only two labels, pos and neg.
        If one probability is 0, returns ratio of 10000.
        Returns float.
    """
    l0 = 'pos'
    l1 = 'neg'
    highest = 10000

    posprob = p_of_feature_given_label(pd, l0, fname)
    negprob = p_of_feature_given_label(pd, l1, fname)
    if negprob > 0:
        r1 = posprob / negprob
    else:
        r1 = highest

    if posprob > 0:
        r2 = negprob / posprob
    else:
        r2 = highest

    return max(r1, r2)


def savefeatures(classifier, filename):
    """Accepts feature probability distributions, filename string.
        Saves features to the file in json format.
        Returns features list of 6-tuples.  
        First is name of feature,
            second is predictive ratio,
            third is probability of showing up given positive label,
            4th is ... negative.
            5th is the label of maximum P(label|feature).
            6th is P(label|feature) for that feature.
    """
    features = []
    pd = classifier._feature_probdist

    for (label, fname) in pd:
        if label == 'neg':
            ratio = maxratio(pd, fname)
            posprob = p_of_feature_given_label(pd, 'pos', fname)
            negprob = p_of_feature_given_label(pd, 'neg', fname)

            ld = classifier.prob_classify({fname: True})
            max_class = ld.max()
            pclass = ld.prob(max_class)

            features.append((fname, ratio, posprob, negprob, max_class, pclass))

    jsondata.save(filename, features)
    return features

'''
f = savefeatures(classifier, 'features.json')
'''

