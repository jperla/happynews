

# finds articles titles in section A of newspaper
[a['title'] for a in list(db.article.find({'section': 'A'}, limit=1000)) if 'Corrections' not in a['title'] and 'Paid Notice' not in a['title']]


# grab the inferred ratings from an slda --infer step
b = list(jsondata.read("./models/nytimes_slda_infer_from_yelp_2011_10_25/fcast.dat"))

import pylab
pylab.hist(b, 30)
# pylab.show()

neg = [(i,z) for i,z in enumerate(b) if z < 3.2]
pos = [(i,z) for i,z in enumerate(b) if z > 3.8]

# get all the foreign docs
allforeigndocs = list(medium_size_filter(nytimes_foreign_desk(), 2000))

# describe document to make sure i have the right one
n = 400
describe_doc('data/nytimes_500_foreign_desk_2011_10_25.dat', 'data/nytimes_med_common_vocab.json', n)
# should be the same article as
allforeigndocs[n]

print len(neg), len(pos)
nchars = 1000
for i,j in (neg + pos):
    print allforeigndocs[i][:nchars] + '\n'
