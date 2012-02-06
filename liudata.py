#!/usr/bin/env python
"""
    converts liu opinion sentiment word lists into json format
    visualizing them
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import codecs
import jsondata

filename = 'data/liu_neg_words.txt'

f,ext = filename.rsplit('.', 1)

lines = codecs.open(filename, 'r', 'utf8').readlines()
real = [l.strip('\r\n ') for l in lines if not l.startswith(';')]
real = [l for l in real if l]

try:
    jsondata.save(f + '.json', real)
except:
    import pdb;pdb.post_mortem()

