"""
    helps in creating vocab and data files for lda-c
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import os
import codecs

import jsondata
import vectorize

def create_full_vocab(docgen):
    """Accepts generator of strings, representing text documents.
        Tokenizes documents.
        Returns a list of all lowercased-normalized words in document,
            sorted alphabetically.
    """
    vocab = {}
    for doc in docgen:
        for word in vectorize.words_from_text(doc):
            if word not in vocab:
                vocab[word] = True
    return list(sorted(vocab.keys()))

def filter_vocab(docgen, vocab, highfreq=.4, lowfreq=.001):
    """Accepts string generator, vocab list, 
        and a range of high frequency and low frequency.
        Filters out words from the vocab that are in more
            than highfreq proportion of documents, and
            lower than low freq proportion of docs.
        Returns new list of vocab.
    """
    vocab = dict((v, 0) for v in vocab)
    numdocs = 0
    for doc in docgen:
        numdocs += 1
        for word in set(vectorize.words_from_text(doc)):
            vocab[word] += 1
    highcount = int(numdocs * highfreq)
    lowcount = int(numdocs * lowfreq)
    return list(sorted(k for k,v in vocab.iteritems() if lowcount <= v <= highcount))


def numdocs(docgen):
    """Accepts string generator.
        Returns an integer.
    """
    i = 0
    for d in docgen:
        i += 1
    return i

def numwords_per_doc(docgen):
    """
    Use this to generate histogram:
    hist = ...
    hv = [hist.get(i, 0) for i in xrange(2000)]
    b = 100
    hc = [(i*b, sum(hv[i*b:(i+1)*b])) for i in xrange(100)]
    """
    hist = {}
    for doc in docgen:
        num_words = len(list(vectorize.words_from_text(doc)))
        hist[num_words] = hist.get(num_words, 0) + 1
    return hist

def numwords(docgen):
    """Accepts string generator.
        Returns integer of number of words..
    """
    i=0
    for doc in docgen:
        i += glen(vectorize.words_from_text(doc))
    return i

def glen(g):
    """Accepts generator.  Destroys generator. Returns its length."""
    return sum(1 for i in g)

def nytimes_medium_article_docgen():
    """Returns articles with a reasonable number of words
        (500 - 3000)
    """
    for doc in nytimes_docgen():
        size = glen(vectorize.words_from_text(doc))
        if 500 <= size <= 3000:
            yield doc


#########################
# Docgens return strings (documents) to vectorize
#########################

try:
    import pymongo
    db = pymongo.Connection('localhost', 27017).nytimes
except:
    print 'did not connect to mongo; not running'
dirname = '/Users/josephperla/projects/projects/nytimesscrape/'
def nytimes_docgen():
    for a in db.article.find():
        fulltext_loc = a['fulltext_loc']
        with codecs.open(os.path.join(dirname, fulltext_loc), 'r', 'utf8') as f:
            y = f.read()
        yield y

def nytimes_recent_docgen():
    for a in db.article.find(sort=[('pubdate', -1),]):
        fulltext_loc = a['fulltext_loc']
        with codecs.open(os.path.join(dirname, fulltext_loc), 'r', 'utf8') as f:
            y = f.read()
        yield y

def nytimes_foreign_desk():
    for a in db.article.find({'section': 'A'}, sort=[('pubdate', -1),]):
        if 'Foreign Desk' in a['title']:
            fulltext_loc = a['fulltext_loc']
            with codecs.open(os.path.join(dirname, fulltext_loc), 'r', 'utf8') as f:
                y = f.read()
            yield y

def yelp_docgen():
    for review in vectorize.read_yelp_reviews():
        yield review['text']

def yelp_labelgen():
    for review in vectorize.read_yelp_reviews():
        yield review['stars']


def size_filter(docgen, limit):
    """Accepts string generator, integer limit.
        Only returns up to limit number.
        Returns new string generator.
    """
    i = 0
    for doc in docgen:
        i += 1
        yield doc
        if i >= limit:
            break
    

def medium_size_filter(docgen, limit):
    """Accepts string generator, integer limit.
        Filters out docs of incorrect size 500 - 3000.
        Only returns up to limit number.
        Returns new string generator.
    """
    i = 0
    for doc in docgen:
        size = glen(vectorize.words_from_text(doc))
        if 500 <= size <= 3000:
            i += 1
            yield doc
            if i >= limit:
                break





######################
# save in format for lda-c
######################

def term_vector(doc, vocab):
    """Accepts document string, vocab list.
        Returns a dictionary, sparse vector, of term:count in doc.
        Note that index of term is index of word in vocab list.
    """
    term_counts = {}
    vocab = dict((v,i) for i,v in enumerate(vocab))
    for word in vectorize.words_from_text(doc):
        if word in vocab:
            term_id = vocab[word]
            if term_id in term_counts:
                term_counts[term_id] += 1
            else:
                term_counts[term_id] = 1
    return term_counts

def save_sparse(filename, docgen, vocab):
    """Accepts string generator.
        Vectorizes each string into words, looks those up in a vocabulary.
        Saves each document line by line in the following format:
            [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]
    """
    with open(filename, 'w') as f:
        for doc in docgen:
            term_counts = term_vector(doc, vocab)
            terms = ' '.join('%s:%s' % (t,c) for t,c in term_counts.iteritems())
            f.write('%s %s\n' % (len(term_counts), terms))

'''
# make vocabulary
common_vocab = filter_vocab(nytimes_medium_article_docgen(), create_full_vocab(nytimes_medium_article_docgen()))

jsondata.save('data/nytimes_med_common_vocab.json', common_vocab)
common_vocab = list(jsondata.read('data/nytimes_med_common_vocab.json'))

# save nytimes dataset for lda-c
for n in (100, 200, 500, 1000, 2000, 5000, 10000):
    save_sparse('data/nytimes_%s_emotive_lda_2011_10_17.dat' % n, medium_size_filter(nytimes_recent_docgen(), n), common_vocab)
    print n

for n in (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 1e99):
    save_sparse('data/yelp_%s_sparse_lda_2011_10_16.dat' % n, size_filter(yelp_docgen(), n), common_vocab)
    with open('data/yelp_%s_annotations_slda_2011_10_16.dat' % n, 'w') as f:
        for a in size_filter(yelp_labelgen(), n):
            f.write('%s\n' % a)
    print n



# save nytimes dataset for lda-c
for n in (100, 200, 500, 1000, 2000, 5000, 10000):
    save_sparse('data/nytimes_%s_foreign_desk_2011_10_25.dat' % n, medium_size_filter(nytimes_foreign_desk(), n), common_vocab)
    print n
'''

