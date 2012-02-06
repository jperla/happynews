"""
    web server that interacts with active learning
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
#!/usr/bin/env python

import hashlib
import random

from flask import request
from flask import Flask

import pymongo
from pymongo import objectid

import jsondata

app = Flask(__name__)

db = pymongo.Connection('localhost', 27017).ydb

def get_new_document():
    num_reviews = db.review.count()
    i = random.randint(0, num_reviews - 1)
    return db.review.find_one({}, skip=i, limit=1)


def set_document_rating(docid, rating):
    r = db.review.find_one({'_id':pymongo.objectid.ObjectId(docid)})
    r['has_rating'] = True
    r['rating'] = rating
    db.review.save(r)
    return True

def recalculate_model():
    reviewed = list(db.review.find({'has_rating': True}))
    # make stuff

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        vote = request.form['vote']
        docid = request.form['docid']
        if vote == 'Positive':
            set_document_rating(docid, 1)
        elif vote == 'Negative':
            set_document_rating(docid, -1)
        else:
            print 'no known vote type...'
    else:
        vote = ''
    
    recalculate_model()
    doc_dict = get_new_document()

    doc = doc_dict['text']
    docid = doc_dict['_id']

    html = open('base.html', 'r').read()
    html = html.format(vote=vote, docid=docid, doc=doc)
    return html

if __name__ == "__main__":
    app.debug = True
    app.run()
    
