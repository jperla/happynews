import time
import scipy.special

start = time.time()

a = 0.0
for i in xrange(1, 1000000, 1):
    f = i / 100.0
    a += scipy.special.psi(f) / float(i + 1) * i

end = time.time()
print '%s: took %s seconds' % (a, end - start)


start = time.time()

i = 0.01
while i < 10000:
    scipy.special.psi(i)
    i += 0.01

end = time.time()
print 'bare metal took %s seconds' % (end - start)

start = time.time()
scipy.special.psi(1000)
end = time.time()
print 'one step took %s seconds' % (end - start)
