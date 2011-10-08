"""
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
#!/usr/bin/env python
import random
import json
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
 
def word_feats(words):
    return dict([(word, True) for word in words])

def words_from_review_text(text):
    text = text.replace('[', ' ').replace(']', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('---', '   ').replace('--', '  ').replace('?', ' ').replace('!', ' ')
    words = nltk.tokenize.word_tokenize(text)
    low_words = [w.lower() for w in words]
    return low_words


if 'data' not in locals(): # useful for running in ipython
    data = [json.loads(r) for r in open('yelp_academic_dataset.json').readlines()]

    all_reviews = [d for d in data if d['type'] == 'review']

reviews = all_reviews#[:80000]
print 'a'

pos_stars = 4
neg_stars = 2

posrevs = [words_from_review_text(r['text']) for r in reviews if r['stars'] >= pos_stars]
negrevs = [words_from_review_text(r['text']) for r in reviews if r['stars'] <= neg_stars]


print 'b'
random.shuffle(posrevs)
random.shuffle(negrevs)

# normalize so we have equal pos and neg reviews
assert(len(posrevs) > len(negrevs))
#posrevs = posrevs[:len(negrevs)]

print 'c'
negfeats = [(word_feats(f), 'neg') for f in negrevs]
posfeats = [(word_feats(f), 'pos') for f in posrevs]


print 'number of positive examples: %s, negative examples: %s' % (len(posrevs), len(negrevs))
 
negcutoff = len(negfeats) * 3 / 4
poscutoff = len(posfeats) * 3 / 4
 
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()
