
import jsondata
liu = jsondata.read('data/liu_pos_words.json')
liu
liu = list(jsondata.read('data/liu_pos_words.json'))
liu = list(jsondata.read('data/liu_pos_words.json'))
liup = liu
liu = list(jsondata.read('data/liu_neg_words.json'))
liun = liu
liu = liun + liup
len(liu)
len(liup)
len(liun)
liu[10]
import vlex
mydata = jsondata.read('data/yelp_4cat_naive_full_mytoken_74.json')
mydata = list(mydata)
myscores = vlex.parse_bayes_into_scores(mydata)
myscores
myscores[:10]
n = partial(normscore, 30) # 30:1 likelihood pretty extreme already
words = [(w, n(s)) for w,s in raw_words]
raw_words = myscores
%paste
from functools import partial
n = partial(vlex.normscore, 30)
%paste
words
words[:10]
words[:30]
len(words)
dw = dict(words)
len(dw)
myliu = [(l, dw.get(l, 'unknown')) for l in liu]
myliu[:10]
myliu = [(l, dw.get(l, 'unknown')) for l in liup]
mliu
myliu
myliup = [(l, dw.get(l, 'unknown')) for l in liup]
myliup
len([m for m in myliup if m[1] != 'unknown'])
len(myliup)
len([m for m in myliup if m[1] != 'unknown' and m[1] > 0])
len([m for m in myliup if m[1] != 'unknown' and m[1] < 0])
[m for m in myliup if m[1] != 'unknown' and m[1] < 0]
[m for m in myliup if m[1] != 'unknown' and m[1] < -.01]
len([m for m in myliup if m[1] != 'unknown' and m[1] < -.01])
len([m for m in myliup if m[1] != 'unknown' and m[1] < -.1])
[m for m in myliup if m[1] != 'unknown' and m[1] < -.1]
[m for m in myliup if m[1] != 'unknown' and m[1] < -.1]
[m for m in myliup if m[1] != 'unknown' and m[1] < -.1]
[m for m in myliup if m[1] != 'unknown' and m[1] > .1]
len([m for m in myliup if m[1] != 'unknown' and m[1] > .1])
[m for m in myliup if m[1] != 'unknown' and m[1] > .02]
len([m for m in myliup if m[1] != 'unknown' and m[1] > 0])
len([m for m in myliup if m[1] != 'unknown' and m[1] > .1])
myliun = [(l, dw.get(l, 'unknown')) for l in liun]
myliun[:10]
len([m for m in myliun if m[1] != 'unknown' and m[1] < 0])
len(myliun)
len([m for m in myliun if m[1] != 'unknown'])
len([m for m in myliun if m[1] != 'unknown' and m[1] > .1])
[m for m in myliun if m[1] != 'unknown' and m[1] > .1]
len([m for m in myliun if m[1] != 'unknown' and m[1] > .1])
len(myliun)
len([m for m in myliun if m[1] != 'unknown'])
len([m for m in myliun if m[1] != 'unknown' and m[1] < -.1])
len([m for m in myliun if m[1] != 'unknown' and m[1] < 0])
myscores
myscores[:100]
mydata[:100]
words[:100]
len(words)
len([w for w in words if abs(w[1]) > .1])
mydata = jsondata.read('data/yelp_4cat_naive_full_mytoken_622.json')
mydata = list(mydata)
words[-10:]
myscores = vlex.parse_bayes_into_scores(mydata)
myscores[-10]
    words = [(w, n(s)) for w,s in raw_words]
words = [(w, n(s) for w,s in myscores]
words = [(w, n(s)) for w,s in myscores]
len(words)
len([m for m in myliun if m[1] != 'unknown' and m[1] < 0])
myliun = [(l, dw.get(l, 'unknown')) for l in liun]
dw = dict(words)
len(dw)
myliun = [(l, dw.get(l, 'unknown')) for l in liun]
myliup = [(l, dw.get(l, 'unknown')) for l in liup]
len([m for m in myliup if m[1] != 'unknown'])
len([m for m in myliun if m[1] != 'unknown'])
len([m for m in myliun if m[1] != 'unknown' and m[1] > .1])
len([m for m in myliun if m[1] != 'unknown' and m[1] < -.1])
len([m for m in myliun if m[1] != 'unknown' and m[1] < 0])
len([m for m in myliup if m[1] != 'unknown' and m[1] < -.1])
len([m for m in myliup if m[1] != 'unknown' and m[1] > .1])
len([m for m in myliup if m[1] != 'unknown' and m[1] > 0])
len([m for m in myliup if m[1] != 'unknown'])
len([w for w in words if abs(w[1]) > .1])
len([w for w in words if abs(w[1]) > .2])

