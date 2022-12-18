import os
p = '../../output/msmarco.passage/t5.base.gc.doc.query/'
for f in os.listdir(p):
    s = p + f
    d = p + f.replace('icted_queries.txt0', '.')
    print(f'{s} -> {d}')
    os.rename(s, d)
