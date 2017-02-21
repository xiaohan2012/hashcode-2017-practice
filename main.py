import numpy as np
import random
from scipy.optimize import linprog
from collections import defaultdict


data = 'medium'

path = 'data/{}.in'.format(data)

with open(path, 'r') as f:
    R, C, L, H = map(int, f.readline().strip().split())
    rows = []
    val_map = {'T': 0, 'M': 1}
    for l in f:
        rows.append(list(map(val_map.__getitem__, list(l.strip()))))
        
    m = np.array(rows)

possible_pieces = []

for i in range(R):
    for j in range(C):
        for nrow in range(H):
            for ncol in range(H):
                if i + nrow <= R and j + ncol <= C:
                    if nrow * ncol <= H:
                        sub_m = m[i:i+nrow, j:j+ncol]
                        cnt1 = np.count_nonzero(sub_m)
                        if L <= cnt1:
                            cnt0 = nrow * ncol - cnt1
                            if L <= cnt0:
                                # randomly drop some candidates
                                if np.random.rand() <= (nrow * ncol) / H:
                                    possible_pieces.append(((i, j), (i+nrow-1, j+ncol-1)))


candidates_size = 5000
possible_pieces = random.sample(possible_pieces, candidates_size)

n_vars = len(possible_pieces)

piece2id = {p: i for i, p in enumerate(possible_pieces)}
id2piece = {i: p for i, p in enumerate(possible_pieces)}


# build the linear program
def get_size(data):
    (x1, y1), (x2, y2) = data
    return abs(x2 - x1 + 1) * abs(y2 - y1 + 1)

c = list(map(get_size, possible_pieces))

n_constraints = 500
if R*C < n_constraints:
    n_constraints = R*C
A_ub = np.zeros((n_constraints, n_vars))
print('constraint matrix size {}x{}'.format(n_constraints, n_vars))

# for each coordinate, associate it with the list of rectangles that cover it
coord2pieces = defaultdict(list)
for piece in possible_pieces:
    (x1, y1), (x2, y2) = piece
    for i in range(x1, x2+1):
        for j in range(y1, y2+1):
            coord2pieces[(i, j)].append(piece)

coords = random.sample(list(coord2pieces.keys()), n_constraints)
for i, coord in enumerate(coords):
    pieces = coord2pieces[coord]
    for p in pieces:
        A_ub[i][piece2id[p]] = 1
        
b_ub = np.ones(n_constraints)

assert b_ub.shape[0] == A_ub.shape[0]
assert n_vars == A_ub.shape[1]

res = linprog(-np.array(c), A_ub, b_ub, bounds=(0, 1),
              options={'maxiter': 100, 'disp': True})
# print(res.x)

# randomized rounding
x = (np.random.rand(len(res.x)) <= res.x)


def between(x, y, x1, y1, x2, y2):
    return (x1 <= x) and (x <= x2) and (y1 <= y) and (y <= y2)


def overlap(pi, pj):
    (x1, y1), (x2, y2) = pi
    (a1, b1), (a2, b2) = pj
    return (between(a1, b1, x1, y1, x2, y2) or
            between(a2, b2, x1, y1, x2, y2) or
            between(a2, b1, x1, y1, x2, y2) or
            between(a1, b2, x1, y1, x2, y2))

pieces = list(map(id2piece.__getitem__, np.nonzero(x)[0]))

pieces = list(sorted(pieces, key=lambda p: res.x[piece2id[p]], reverse=True))

i = 0
while i < len(pieces):
    pi = pieces[i]
    tail = pieces[i+1:]
    add = True
    for pj in tail:
        if overlap(pi, pj):
            add = False
            break
    if not add:
        print('remove {}'.format(pi))
        pieces.remove(pi)
    else:
        i += 1

score = 0
for p in pieces:
    (x1, y1), (x2, y2) = p
    score += (x2 - x1 + 1) * (y2 - y1 + 1)
    
print('score = {}'.format(score))
    
with open('output/{}.out'.format(data), 'w') as f:
    f.write('{}\n'.format(len(pieces)))
    for (x1, y1), (x2, y2) in pieces:
        f.write('{} {} {} {}\n'.format(x1, y1, x2, y2))
