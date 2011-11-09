# -*- coding: utf-8 -*-
"""
    tools for taking a list of strings on a real number line and
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

import codecs
import random
import itertools
from functools import partial

import jsondata

def create_intervals(num_intervals=20, max_abs=1):
    """Accepts an integer number of intervals and a max_abs integer.
        Creates a list of floats from -max_abs to max_abs, describing num_intervals intervals.
        Returns the list.
    """
    intervals = [(2 * max_abs * (r - (num_intervals / 2.0)) / num_intervals) for r in range(num_intervals + 1)]
    return intervals

def choose_words(point, words, num_to_choose=20):
    """Accepts float point score to choose words nearby it, list of 2-tuples (word, score), and integer number of words to choose.
        Finds the closest num_to_choose words closest to point.
        Returns a list of those words.
    """
    # use random.random() to sort within a distance
    s = sorted((abs(w[1] - point), random.random(), w[0]) for w in words)
    return [j[2] for j in itertools.islice(s, 0, num_to_choose)]

def print_lists(lists, width=12):
    """Accepts a list of two-tuples (r, list).  
        First element is a floating point number score, 
        second is a list of strings.
        Returns a string representation of this in a readable way.
    """
    s = []
    depth_of_lists = max([len(i[1]) for i in lists])

    header = []
    for (r, l) in lists:
        header.append(unicode(r).center(width))
    s.append(''.join(header))

    for d in xrange(depth_of_lists):
        line = []
        for (r, l) in lists:
            word = l[d]
            # we want to shorten words that are too long
            shortword = (word[:width-4] + '...') if len(word) > (width-2) else word
            line.append(shortword.center(width))
        s.append(''.join(line))
    return '\n'.join(s)
            

def parse_bayes_into_scores(data):
    """Accepts a list of 6-tuples of data.
        The first element is a string, the word.
        The second to last element is either 'pos' or 'neg' label string.
        The second element is the score of confidence to that label 
            (probability of feature given the label versus the opposite label).
    """
    # specific to my data; turn the file into a list of (word, score) 2-tups
    '''
    posd = [(d[0], d[-1]) for d in data if d[-2] == 'pos']
    negd = [(d[0], -1 * d[-1]) for d in data if d[-2] == 'neg']
    '''
    posd = [(d[0], d[1]) for d in data if d[2] > d[3]]
    negd = [(d[0], -1 * d[1]) for d in data if d[2] < d[3]]
    words = posd + negd
    return words

def normscore(max_abs, score):
    """Accepts number max_abs of highest score, that maps to 1.0.
        Also accept score which is a number.
        Converts the score to the range -1.0 to 1.0. 
        max_abs scores and higher map to 1.0/-1.0 depending on sign.
        Returns a float.
    """
    if abs(score) >= max_abs:
        return float(score) / abs(score)
    else:
        return float(score) / max_abs

if __name__ == '__main__':
    import sys
    #filename = sys.argv[1]
    #num_intervals = sys.argv[2] if len(sys.argv) > 2 else 10

    fbase,ext = filename.rsplit('.', 1)
    intervals = create_intervals(num_intervals, 1)
    data = list(jsondata.read(filename))
    
    raw_words = parse_bayes_into_scores(data)

    n = partial(normscore, 30) # 30:1 likelihood pretty extreme already
    words = [(w, n(s)) for w,s in raw_words]

    m = [(p, choose_words(p, words, 200)) for p in intervals]

    html = '<html><body><code><pre>%s</pre></code></body></html>'
    p = html % print_lists(m, 15)

    with codecs.open(fbase + '.html', 'w', 'utf8') as f:
        f.write(p)

