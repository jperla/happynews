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

import random
import itertools

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
            



if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    num_intervals = sys.argv[2] if len(sys.argv) > 2 else 10

    fbase,ext = filename.rsplit('.', 1)
    data = list(jsondata.read(filename))
    intervals = create_intervals(num_intervals, 1)

    # specific to my data; turn the file into a list of (word, score) 2-tups
    posd = [(d[0], d[-1]) for d in data if d[-2] == 'pos']
    negd = [(d[0], -1 * d[-1]) for d in data if d[-2] == 'neg']
    words = posd + negd

    m = [(p, choose_words(p, words, 200)) for p in intervals]

    html = '<html><body><code><pre>%s</pre></code></body></html>'
    p = html % print_lists(m, 15)

    with open(fbase + '.html', 'w') as f:
        f.write(p)

