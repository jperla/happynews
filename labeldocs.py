"""
    helps me extract topics and label new text documents using topics
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import jsondata

def describe_doc(data_filename, vocab_filename, docid):
    counts = [f for f in (open(data_filename).readlines()[docid]).split(' ')][1:]
    wordids = [int(c.split(':')[0]) for c in counts]
    vocab = list(jsondata.read(vocab_filename))
    words = [vocab[i] for i in wordids]
    return sorted(words)


def grab_topic(beta_filename, vocab_filename, topicid):
    counts = [float(f) for f in (open(beta_filename).readlines()[topicid]).split(' ')]
    minimum = min(counts) # to ignore very irrelevant words
    vocab = list(jsondata.read(vocab_filename))
    words = [(vocab[i], p) for i,p in enumerate(counts) if p > minimum]
    return sorted(words, key=lambda v:-v[1])


t = grab_topic('testbeta.dat', 'data/nytimes_med_common_vocab.json', 4)
print t[:40]

