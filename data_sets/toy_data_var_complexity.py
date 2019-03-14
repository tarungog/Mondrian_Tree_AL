# Create toy data with constant variance, but differing fequency

import random
import math

class ComplexityStream:

    def __init__(self, *args, **kwargs):
        pass


def toy_data_var_complexity(n,p,high_area,std=1,low_freq=0.2,high_freq=0.05, low_mag=1, high_mag=2,
    set_seed=None, marginal = 'uniform', return_true_labels=False, useless_dims = 0):

    if set_seed is not None:
        random.seed(set_seed)

    points = []
    labels = []
    true_labels = []
    for i in range(n):
        point = []
        is_high_var = []
        for j in range(p-useless_dims):
            if marginal == 'normal':
                val = random.gauss(0.5, 0.5/3)
            else:
                val = random.random()
            point.append(val)
            if val > high_area[j][0] and val < high_area[j][1]:
                is_high_var.append(True)
            else:
                is_high_var.append(False)

        for j in range(useless_dims):
            if marginal == 'normal':
                val = random.gauss(0.5, 0.5/3)
            else:
                val = random.random()
            point.append(val)

        if all(is_high_var):
            labels.append(high_mag*math.sin((2*math.pi)/(high_freq * (p - useless_dims))*
                sum(point[:(p - useless_dims)])) + random.gauss(0, std))
            true_labels.append(high_mag*math.sin((2*math.pi)/(high_freq * (p - useless_dims))*
                sum(point[:(p - useless_dims)])))
        else:
            labels.append(low_mag*math.sin((2*math.pi)/(low_freq * (p - useless_dims))*
                sum(point[:(p - useless_dims)])) + random.gauss(0, std))
            true_labels.append(low_mag*math.sin((2*math.pi)/(low_freq * (p - useless_dims))*
                sum(point[:(p - useless_dims)])))
        points.append(point)


    if return_true_labels:
        return points, labels, true_labels
    else:
        return points, labels


