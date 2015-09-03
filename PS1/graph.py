from __future__ import print_function
import matplotlib.pyplot as plt

RANGES = ((2, 100), (2, 10**6), (2, 10**9))
NPS = (1, 2, 4, 8)

f = open('results.txt', 'r')
results = eval(f.readline())
f.close()


def where(collection, key, value):
    return tuple(filter(lambda x: x[key] == value, collection))

def pluck(collection, key):
    return tuple(map(lambda x: x[key], collection))

def group_by(collection, key):
    gs = sorted(set(pluck(collection, key)))
    return [where(collection, key, g) for g in gs]

for r in RANGES:
    subplot = plt.subplot()
    plt.suptitle("Summing 1/log(n) from 2 to {}".format(r[1]))
    plt.xlabel("Range of sum (2 -> n)")
    plt.ylabel("Time in seconds")

    foo = where(results, 'whole_range', r)
    for np in NPS:
        bar = where(foo, 'np', np)
        ys = pluck(bar, 'time')
        xs = pluck(pluck(bar, 'part_range'), 1)
        subplot.plot(xs, ys, label="{} pcs.".format(np))

    box = subplot.get_position()
    subplot.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    subplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

