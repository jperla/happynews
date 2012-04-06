#!/usr/bin/env python

import csv

import numpy

filename = 'secondbatch.csv'
r = csv.reader(open(filename, 'r'))


articles = {}
for line in r:
    n = line[-3]
    v = float(line[-1])
    if n in articles:
        articles[n].append(v)
    else:
        articles[n] = [v]

import re

w = csv.writer(open('silver.csv', 'w'))

for k,v in articles.iteritems():
    maximum = max(v)
    minimum = min(v)
    std = numpy.std(v)
    mean = numpy.mean(v)
    m = re.match(r'.*\((\d+)\)', k)
    if m:
        row = k,m.group(1),mean,std,minimum,maximum
        w.writerow(row)
        print row
