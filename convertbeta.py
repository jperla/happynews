#!/usr/bin/env python
import sys
from itertools import izip

if __name__ == '__main__':
    vocabfile = sys.argv[1]
    betafile = sys.argv[2]
    output = sys.argv[3]

    num_words = len(open(vocabfile, 'r').readlines())
    print '%s words per topic' % num_words

    with open(output, 'w') as o:
        with open(betafile, 'r') as f:
            def makelines():
                l = True
                while l:
                    l = f.readline()
                    yield l
            lines = makelines()
            while True:
                betas = [line.strip('\r\n ')
                        for i,line in izip(xrange(num_words), lines)]
                if len(betas) == 0 or betas == ['']:
                    break
                else:
                    if (len(betas) == num_words):
                        o.write('%s\n' % ' '.join(betas))
                    else:
                        import pdb; pdb.set_trace()
                        pass

