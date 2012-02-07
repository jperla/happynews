import jsondata
import tlc

from itertools import izip

if __name__=='__main__':
    num_labeled = 10000

    # use my tlc synthetically generated dataset
    documents = jsondata.read('data/documents.dc.nyt.json')
    comments = jsondata.read('data/comments.dc.nyt.json')
    labeled_documents = jsondata.read('data/yelp.nyt_med.json')[:num_labeled]
    background = jsondata.read('data/background.nyt_med.json')

    y = jsondata.read('data/yelp.labels.json')[:num_labeled]
    y = [(i - 3.0) for i in y] # center around 0


    # filter out documents with no words
    all_data = [(l,i) for l,i in izip(labeled_documents,y) if len(l) > 0]
    print len(all_data)
    labeled_documents = [a[0] for a in all_data]
    y = [a[1] for a in all_data]

    real_data = (documents, comments, labeled_documents, background, y)

    var = tlc.TLCVars(real_data, Ku=25, Ks=5, Kb=25)

    try:
        output = tlc.run_tlc(var)
    except Exception,e:
        print e
        import pdb; pdb.post_mortem()

