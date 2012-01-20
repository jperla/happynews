import time
import scipypy

start = time.time()

a = 0.0
for i in xrange(1, 1000000, 1):
    f = i / 100.0
    a += scipypy.psi(f) / float(i + 1) * i

end = time.time()
print '%s: took %s seconds' % (a, end - start)


start = time.time()

i = 0.01
while i < 10000:
    scipypy.psi(i)
    i += 0.01

end = time.time()
print 'bare metal took %s seconds' % (end - start)

start = time.time()
scipypy.psi(1000)
end = time.time()
print 'one step took %s seconds' % (end - start)

start = time.time()
scipypy.psi(999)
end = time.time()
print 'one step again took %s seconds' % (end - start)


start = time.time()
for i in xrange(10):
    scipypy.psi(1000 + i)
end = time.time()
print 'ten steps took %s seconds' % (end - start)

start = time.time()
for i in xrange(100):
    scipypy.psi(1000 + i)
end = time.time()
print 'one hundred steps took %s seconds' % (end - start)
