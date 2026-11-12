"""Extract the best-fit curves of Fig. 26 (arXiv:2109.11174v2, page 30).

Usage: python scripts/extract_sk4_fig26.py <SK paper PDF> data/sk4_fig26_bestfit.csv
"""
import sys, fitz, numpy as np
pdf, out = sys.argv[1], sys.argv[2]
doc = fitz.open(pdf); page = doc[29]
d = page.get_drawings()
rgb = lambda c: tuple(round(v, 2) for v in (c or ()))
panels = [x['rect'] for x in d if rgb(x.get('fill')) == (0.92, 0.92, 0.95)]
COL = {(0.33, 0.66, 0.41): 'ncqe', (0.51, 0.45, 0.7): 'mupi', (0.77, 0.31, 0.32): 'decaye',
       (0.8, 0.73, 0.45): 'spall', (0.39, 0.71, 0.8): 'dsnb', (0.3, 0.45, 0.69): 'blue'}
REGION = ['IV', 'V', 'VI', 'I', 'II', 'III']  # top row: !=1 tag; bottom: 1 tag (Table VI)
centres = np.arange(16.5, 79.0, 2.0)
rows = []
for ip, P in enumerate(panels):
    inside = [x for x in d if P.contains(fitz.Point((x['rect'].x0 + x['rect'].x1) / 2, (x['rect'].y0 + x['rect'].y1) / 2))]
    white = [x for x in inside if rgb(x.get('color')) == (1.0, 1.0, 1.0)]
    xs = sorted({round(it[1].x, 2) for x in white for it in x['items'] if it[0] == 'l' and abs(it[1].x - it[2].x) < 0.01})
    ys = sorted({round(it[1].y, 2) for x in white for it in x['items'] if it[0] == 'l' and abs(it[1].y - it[2].y) < 0.01}, reverse=True)
    xs = [v for v in xs if P.x0 < v < P.x1]; ys = [v for v in ys if P.y0 < v < P.y1]
    assert len(xs) == 7 and len(ys) == 8, (ip, xs, ys)
    ax = np.polyfit(xs, np.arange(20, 81, 10), 1); ay = np.polyfit(ys, np.arange(0, 17.6, 2.5), 1)
    curves = {}
    for x in inside:
        name = COL.get(rgb(x.get('color')))
        if not name or x['type'] != 's':
            continue
        pts = [x['items'][0][1]] + [it[2] for it in x['items'] if it[0] == 'l']
        if len(pts) < 2:
            continue
        X = np.polyval(ax, [p.x for p in pts]); Y = np.polyval(ay, [p.y for p in pts])
        if X.max() - X.min() < 5 or (len(pts) <= 2 and abs(Y[0] - Y[-1]) < 1e-6 and Y[0] > 1.0):
            continue  # legend sample lines
        curves.setdefault(name, []).append((X, Y))
    blues = curves.pop('blue', [])
    blues.sort(key=lambda c: np.interp(centres, c[0], c[1], left=0, right=0).sum())
    if blues:
        curves['total'] = [blues[-1]]
    if len(blues) > 1:
        curves['nuecc'] = [blues[0]]
    for name, segs in curves.items():
        X = np.concatenate([s[0] for s in segs]); Y = np.concatenate([s[1] for s in segs]); o = np.argsort(X)
        vals = np.interp(centres, X[o], Y[o], left=0.0, right=0.0)
        rows.append((REGION[ip], name, vals))
with open(out, 'w') as f:
    f.write('region,component,' + ','.join(f'{c:.1f}' for c in centres) + '\n')
    for r, n, v in sorted(rows):
        f.write(f'{r},{n},' + ','.join(f'{max(x, 0):.4f}' for x in v) + '\n')
print(len(rows), 'curves')
