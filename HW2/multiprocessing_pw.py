#%%
import multiprocessing
from multiprocessing import Process
import time
import matplotlib.pyplot as plt
import numpy as np

def alg1(data):
    data = list(data)
    changes = True
    while changes:
        changes = False
        for i in range(len(data) - 1):
            if data[i + 1] < data[i]:
                data[i], data[i + 1] = data[i + 1], data[i]
                changes = True
    return data

def data1(n, sigma=10, rho=28, beta=8 / 3, dt=0.01, x=1, y=1, z=1):
    import numpy
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result


def alg2(data):
    if len(data) <= 1:
        return data
    else:
        split = len(data) // 2
        left = iter(alg2(data[:split]))
        right = iter(alg2(data[split:]))
        result = []
        # note: this takes the top items off the left and right piles
        left_top = next(left)
        right_top = next(right)
        while True:
            if left_top < right_top:
                result.append(left_top)
                try:
                    left_top = next(left)
                except StopIteration:
                    # nothing remains on the left; add the right + return
                    return result + [right_top] + list(right)
            else:
                result.append(right_top)
                try:
                    right_top = next(right)
                except StopIteration:
                    # nothing remains on the right; add the left + return
                    return result + [left_top] + list(left)

def worker(data, id, return_dict):
    res = alg2(data)
    return_dict[id] = res

def new_alg2(data):

    split = len(data) // 2
    left = data[:split]
    right = data[split:]

    # never used queue for storing large amount of data!
    # q = Queue()
    # use manger instead
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process_list = []

    p1 = Process(target=worker, args=(left,'left', return_dict))
    process_list.append(p1)
    p1.start()

    p2 = Process(target=worker, args=(right, 'right', return_dict))
    process_list.append(p2)
    p2.start()

    # wait for processes to finished
    p1.join()
    p2.join()

    # combined the result from the processes
    left = iter(list(return_dict['left']))
    right = iter(list(return_dict['right']))
    left_top = next(left)
    right_top = next(right)
    result = []
    while True:
        if left_top < right_top:
            result.append(left_top)
            try:
                left_top = next(left)
            except StopIteration:
                # nothing remains on the left; add the right + return
                return result + [right_top] + list(right)
        else:
            result.append(right_top)
            try:
                right_top = next(right)
            except StopIteration:
                # nothing remains on the right; add the left + return
                return result + [left_top] + list(left)


if __name__ == '__main__':
    data = np.logspace(1,6,20)
    # data = data1(1000000)
    y1 = []
    y2 = []
    for s in data:
        s = int(s)
        x = data1(s)
        start_time = time.time()
        alg2(x)
        end_time = time.time()
        y1.append(end_time - start_time)

        x = data1(s)
        start_time = time.time()
        new_alg2(x)
        end_time = time.time()
        y2.append(end_time - start_time)

plt.loglog(data, y1, label='alg2')
plt.loglog(data, y2, label='parallel alg2')
plt.legend()
plt.show()
plt.savefig('pbnb.png')


# %%

# %%
