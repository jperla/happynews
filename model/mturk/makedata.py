#!/usr/bin/env python
import os
import csv
import codecs

import pymongo
from pymongo.objectid import ObjectId

import jsondata

db = None
try:
    import pymongo
    db = pymongo.Connection('localhost', 27017).nytimes
except:
    print 'did not connect to mongo; not running'

articles = list(jsondata.read('../data/titles.dc.nyt.json'))


w = csv.writer(codecs.open('titles.txt', 'w', 'utf8'))
w.writerow(['id', 'mongoid', 'title', 'article'])

dirname = '../../../../../nytimesscrape/'

for i,a in enumerate(articles):
    a = db.article.find_one({'_id':ObjectId(a[0])})

    fulltext_loc = a['fulltext_loc']
    with codecs.open(os.path.join(dirname, fulltext_loc), 'r', 'utf8') as f:
        fulltext = f.read()
    fulltext.replace('\n', '<p>')
    a['title'] = a['title'].replace(u'\xa0', ' ')
    w.writerow([i, a['_id'], a['title'], fulltext])

