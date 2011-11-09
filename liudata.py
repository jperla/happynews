#!/usr/bin/env python
"""
    converts liu opinion sentiment word lists into json format
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

