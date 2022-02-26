import matplotlib.pyplot as plt
import numpy as np


def get(first, second):
    x0, y0 = np.loadtxt(first).T
    x1, y1 = np.loadtxt(second).T

    l = min(len(x0), len(x1))

    assert np.linalg.norm(x0[:l] - x1[:l]) < 1E-13

    return (x0[:l], y0[:l]), (x1[:l], y1[:l]) 


fig, ax = plt.subplots()
for L in (1., 2., 5.):
    first = f'itersHistory_L{L}_pcTypediag.txt'
    second = f'itersHistory_L{L}_pcTypemg.txt'

    (x0, y0), (x1, y1) = get(first, second)

    l, = ax.plot(x0, y0, label=f'(diag) L = {L}', marker='x', markersize=12)
    l, = ax.plot(x1, y1, label=f'(mg) L = {L}', marker='o', color=l.get_color(), markersize=12)
plt.legend()
plt.savefig('mg_diag.png')
plt.show()

