#!/usr/bin/env python
"""
    Looks inside json data. Prints out first few lines of words.
    Useful for making sure I have the data I want.

    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""

import sys
import jsondata

if __name__=='__main__':
    data_filename = sys.argv[1]
    vocab_filename = sys.argv[2]
    num_docs = int(sys.argv[3])
    words_per_doc = int(sys.argv[4])
    associated_filename = sys.argv[5] if len(sys.argv) > 5 else None

    data = jsondata.read(data_filename)[:num_docs]
    lexicon = jsondata.read(vocab_filename)
    words = [[lexicon[w] for (w,c) in sorted(doc, key=lambda w:-w[1])][:words_per_doc] for doc in data]

    if associated_filename is not None:
        associated = jsondata.read(associated_filename)[:num_docs]

    for i in xrange(num_docs):
        if associated_filename is not None:
            print associated[i], words[i]
        else:
            print words[i]
