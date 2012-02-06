"""
    code i used to generate histograms of some data
    visualizing them
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import pylab

import vlex
import jsondata

f = 'data/yelp_4cat_naive_full_mytoken_74.json'
f = 'data/yelp_4cat_naive_full_standardtoken.json'
f = 'data/yelp_2cat_naive_full_mytoken_783.json'

data = list(jsondata.read(f))
words = vlex.parse_bayes_into_scores(data)

values = [w[1] for w in words]

#remove the modes, +/-.75
#values = [v for v in values if abs(v) != .75]
values = [v for v in values if 30 > abs(v) and abs(v) != 3]


pylab.hist(values, bins=50)

pylab.show()
'''
'''
