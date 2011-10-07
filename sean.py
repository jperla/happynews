rows = [r.strip('\r\n ').split(',') for r in open('v4-lm.dat', 'r').readlines()]
rows = [(r[0], float(r[1]), float(r[2])) for r in rows]
v = list(sorted(rows, key=lambda k:k[2]))
