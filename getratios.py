"""
    Copyright (C) 2011 Joseph Perla

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import json

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

    save_data(filename, features)
    return features

'''
pd = classifier._feature_probdist
f = savefeatures(pd, 'features.json')
'''

