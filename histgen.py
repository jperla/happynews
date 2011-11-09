"""
    code i used to generate histograms of some data
    visualizing them
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
