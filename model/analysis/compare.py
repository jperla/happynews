#!/usr/bin/env python

import jsondata
from inspect_slda_model import predict


phi_filename = '../balancedtlc/mytlc-output-20-phiC.dat.npy.list.npz'
vocab_filename = ''
eta_filename = '../balancedtlc/mytlc-output-20-eta.dat.npy.gz'

titles_filename = '../data/titles.dc.nyt.json'

phi = jsondata.read(phi_filename)
eta = jsondata.read(eta_filename)
titles = jsondata.read(titles_filename)

print 'read in data...'
predicted_ratings = list(sorted((i, predict(eta, p)) for i,p in enumerate(phi)))
print 'predicted ratings...'

import csv
reader = csv.reader(open('gold.csv', 'r'))

#v = [i[0] for i in predicted_ratings]
#import pdb; pdb.set_trace()

rall = []

for line in reader:
    index = int(line[1])
    mean = float(line[2])
    std = float(line[3])
    minimum = float(line[4])
    maximum = float(line[5])

    p = predicted_ratings[index][1]
    print index, predicted_ratings[index][0], p, mean, std, line[0]
    if abs(mean) <= 0.2 and abs(p) <= 0.1:
        good = 'neut'
    elif p * mean >= 0:
        good = 'good'
    else:
        good = 'bad '
    
    r = (p, mean, std, minimum, maximum, good, line[0], titles[index][1])
    rall.append(r)

for r in sorted(rall):
    print r

first = lambda x: list(a[0] for a in x)
second = lambda x: list(a[1] for a in x)

# filter out the ones with very low predicted rating
#rall = [r for r in rall if abs(r[0]) > 0.2]

scatterplot = [(r[0], r[1]) for r in rall]

import matplotlib.pyplot as plot
# make graphs
plot.figure(0)
plot.grid(True)
s = [(r[0], r[1]) for r in rall if r[2] <= 0.5]
plot.scatter(first(s), second(s), 20, 'k', '.')
s = [(r[0], r[1]) for r in rall if r[2] > 0.5]
plot.scatter(first(s), second(s), 20, 'r', 'x')
plot.axis([-0.5, 0.9, -2.0, 1.5])
#plot.plot([-0.5, 0.9], [-2.0, 1.5], ':')
plot.xlabel(r'predicted label')
plot.ylabel(r'mean human rating')
plot.title('Scatterplot of human label versus machine label')
plot.savefig('label-scatterplot.png')


scatterplot = [(abs(r[1] - r[0]), r[2]) for r in rall]

# make graphs
plot.figure(1)
plot.grid(True)
plot.scatter(first(scatterplot), second(scatterplot), 20, 'k', '.')
plot.axis([0, 2.0, 0, 1.0])
plot.xlabel(r'residual difference between predicted and human label')
plot.ylabel(r'variance between human labelers')
plot.title('Errors versus variance')
plot.savefig('variance-scatterplot.png')


