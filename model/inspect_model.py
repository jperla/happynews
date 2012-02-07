#!/usr/bin/env python
"""
    Looks inside outputed model.
    Prints out topics and other variables.
    Useful for inspecting a model's fit.

    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""

import sys
import jsondata

if __name__=='__main__':
    data_filename = sys.argv[1]
    vocab_filename = sys.argv[2]
    words_per_topic = int(sys.argv[3])

    associated_filename = sys.argv[5] if len(sys.argv) > 5 else None

    data = jsondata.read(data_filename)[0]
    lexicon = jsondata.read(vocab_filename)

    beta = data['beta']

    eta = None
    if 'eta' in data:
        eta = data['eta']

    print 'eta: %s' % eta

    for t,topic in enumerate(beta):
        top_words = list(reversed(sorted((p,i) for i,p in enumerate(topic))))[:words_per_topic]
        print 'topic #%s' % t
        for p,i in top_words:
            w = lexicon[i]
            print '%s   (%s)' % (w, p)
        print '\n'

    if associated_filename is not None:
        #associated = jsondata.read(associated_filename)[:num_docs]
        pass

