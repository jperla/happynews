"""
    converts text into a vector of word counts (or bools)
    visualizing them
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import random
import itertools
from itertools import izip

import numpy
import nltk

import jsondata

def words_from_text(text):
    text = text.replace(':', ' ').replace(';', ' ').replace('|', ' ').replace('}', ' ').replace('{', ' ').replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ').replace('~', ' ').replace('\\', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('!', ' ').replace('---', '   ').replace('--', '  ')
    words = nltk.tokenize.word_tokenize(text)
    low_words = [w.lower().strip('-') for w in words]
    return low_words

def words_from_review_text(text):
    #totr = '|{}()[]/,\\.?!~'
    #text = text.translate(string.maketrans(totr, ' ' * len(totr)))
    #text = text.replace('---', '   ').replace('--', '  ').replace('- ', '  ').replace(' -', '  ')
    return words_from_text(text)



def vectorize(lexicon, text, array):
    """Accepts lexicon dictionary, text string, and array. 
        Lexicon dictionary should map string => int, index in array.
        Array should be an empty dictionary (sparse vector representation).
        Tokenizes string, then computes counts on a numpy vector.
        Returns array.  NOTE: Modifies array in place.
    """
    assert {} == array
    for w in words_from_review_text(text):
        i = lexicon.get(w)
        if i is not None:
            if i in array:
                array[i] += 1
            else:
                array[i] = 1
    return array

def linearize(sparse_vector):
    """Accepts a sparse vector (a dictionary).
        Returns a list of 2-tuples of key-value pairs.
    """
    return [(k,v) for k,v in sparse_vector.iteritems()]

def describe_bag(lexicon, array):
    """Accepts lexicon dictionary, and sparse vector array (dictionary).
        Returns a list of the words in the text of sparse vector..
    """
    revlex = dict((b,a) for a,b in lexicon.iteritems())
    return [revlex[i] for i in array.keys()]

def read_yelp_reviews():
    """Returns generator of dicts of reviews from yelp datset."""
    for d in jsondata.read('data/yelp_academic_dataset.json'):
        if d['type'] == 'review':
            yield d

def pyfeatures_to_sparse_r_docs(features):
    """Accepts a list of dictionaries.
        Each item in list represents a document. 
        Each dictionary represents a sparse matrix of terms.
            Each term is a numeric key in the dictionary, 
                (mapped to a vocabulary).
            The value of the dictionary is the number 
                of times the term appears in the document
        Returns an rpython list of rpython 2-row matrices.
            First row are term id numbers, 
                second row matches to column and contains count of term in doc.
    """
    def matrix(d):
        """Accepts dictionary as above. Returns 2-row matrix."""
        # todo: i think this uses way too much memory
        elements = list(itertools.chain(*d.iteritems()))
        return r.matrix(ro.IntVector(elements), nrow=2)
    matrices = [matrix(d) for d in features]
    return matrices
    #return r.list(matrices)


if __name__ == '__main__':
    lexicon = dict([(a,i) for i,a in enumerate(jsondata.read('data/nytimes_med_common_vocab.json'))])

    '''
    db = None
    try:
        import pymongo
        db = pymongo.Connection('localhost', 27017).nytimes
    except:
        print 'did not connect to mongo; not running'

    docs_with_comments = list(db.article.find({'num_comments':{'$gt': 0}}).sort([('pubdate', -1)]))

    dwc = docs_with_comments

    titles = []
    docs = []
    comments = []
    for d in dwc:
        titles.append([str(d.get('_id')), d.get('title', 'no title')])

        # get document
        vector = {}
        text = open('/Users/josephperla/nytimesscrape/' + d.get('fulltext_loc',''), 'r').read()
        v = linearize(vectorize(lexicon, text, vector))
        docs.append(v)

        # get comment
        text = ' '.join([c.get('commentBody', '') for c in d.get('comments', [])])
        vector = {}
        v = linearize(vectorize(lexicon, text, vector))
        comments.append(v)

    jsondata.save_data('titles.dc.nyt.json', titles)
    jsondata.save_data('documents.dc.nyt.json', docs)
    jsondata.save_data('comments.dc.nyt.json', comments)
    '''

    '''
    #lexicon = dict([(a[0],i) for i,a in enumerate(jsondata.read('data/yelp_lexicon_med.json'))])
    #lexicon = dict([(a[0],i) for i,a in enumerate(jsondata.read('data/yelp_lexicon_small.json'))])
    #lexicon = dict([(a[0],i) for i,a in enumerate(jsondata.read('data/yelp_lexicon.json'))])
    print 'lexi'

    '''
    num_samples = 10000000

    data = [d for i,d in izip(xrange(num_samples), read_yelp_reviews())]
    
    # make sparse vectors
    features = [dict() for i in xrange(len(data))]
    labels = [d['stars'] for d in data]

    print 'data'

    for i,d in enumerate(data):
        f = features[i]
        text = d['text']
        vectorize(lexicon, text, f)

    for i in range(10):
        print describe_bag(lexicon, features[-i])
        print data[-i]

    labeled = [linearize(f) for f in features]
    jsondata.save_data('yelp.nyt_med.json', labeled)
    jsondata.save_data('yelp.labels.json', labels)


    '''
    from rpy2 import robjects as ro
    r = ro.r

    from rpy2.robjects.packages import importr
    lda = importr('lda')

    num_topics = 10
    params = [random.choice([-1, 1]) for i in xrange(num_topics)]
    documents = pyfeatures_to_sparse_r_docs(features)
    vocab = [a[1] for a in sorted((i,w) for w,i in lexicon.iteritems())]
    annotations = ro.FloatVector([((l-1.0)/4) for l in labels])

    print 'starting slda'
    #result = lda.slda_em(documents, num_topics, vocab, 10, 4, 1.0, 0.1, annotations, params, 0.1, 1.0, False, 'sLDA')
    topics = lda.lda_collapsed_gibbs_sampler(documents, num_topics, vocab, 25, 0.1, 0.1, compute_log_likelihood=True)

    print 'done'

    """
    > result <- slda.em(documents=poliblog.documents,
    +                   K=num.topics,
    +                   vocab=poliblog.vocab,
    +                   num.e.iterations=10,
    +                   num.m.iterations=4,
    +                   alpha=1.0, eta=0.1,
    +                   poliblog.ratings / 100,
    +                   params,
    +                   variance=0.25,
    +                   lambda=1.0,
    +                   logistic=FALSE,
    +                   method="sLDA")
    """
    '''
