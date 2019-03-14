import copy
import random
import warnings
from argparse import ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from data_sets.toy_data_var_complexity import toy_data_var_complexity
from Mondrian_Tree import Mondrian_Tree

# matplotlib.use('AGG')

RANDOM = True
ACTIVE = True

PLUS_ONE = False

N_FINALS = 500

class Stream:
    def __init__(self, set_builder_func, p, step_size=1000, **kwargs):
        self.func = set_builder_func
        self.step = step_size
        self.kwargs = kwargs
        self.p = p

        self.idx = 0
        self.refresh()

    def refresh(self):
        self.X, self.Y = self.func(n=self.step, p=self.p, **self.kwargs)
        self.local_idx = 0

    def reset(self):
        self.idx = 0
        self.refresh()

    @property
    def point(self):
        if self.local_idx >= self.step:
            self.refresh()

        x = self.X[self.local_idx]
        y = self.Y[self.local_idx]

        self.idx += 1
        self.local_idx += 1

        return x, y


def seeding(seed_index):
    seeds = list(range(1000))
    seed = seeds[int(seed_index)]
    random.seed(seed)
    np.random.seed(seed)
    print(seed)
    return seed


def evaluate(tree, X_test, y_test):
    tree.set_default_pred_global_mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_hat = tree.predict(X_test)

    return np.mean((y_hat - y_test) ** 2)

def example_var_mt(seed_index):
    seed = seeding(seed_index)

    p = 10
    marginal = 'uniform'
    std = 1
    low_freq = 0.1
    high_freq = 0.05
    low_mag = 5
    high_mag = 20
    high_area = [[0.1,1]]*p

    n_final = N_FINALS

    s = Stream(toy_data_var_complexity, p=p,
            high_area=high_area,
            std=std,
            low_freq=low_freq,
            high_freq=high_freq,
            low_mag=low_mag,
            high_mag=high_mag,
            marginal=marginal
        )

    X_test, y_test = toy_data_var_complexity(n=200,p=p,high_area=high_area,std=std,
        low_freq=low_freq,high_freq=high_freq, low_mag=low_mag, high_mag=high_mag
    )

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    MT_al_MSE = []
    MT_rn_MSE = []

    MT_al = Mondrian_Tree([[0,1]]*p)
    MT_rn = Mondrian_Tree([[0,1]]*p)

    if RANDOM:
        while s.idx < n_final:
            x, y = s.point
            MT_rn.add_data_point(x, y)
            n = s.idx
            MT_rn.update_life_time(n**(1/(2+p))-1)

            if s.idx % 5 == 0:
                mse = evaluate(MT_rn, X_test, y_test)
                MT_rn_MSE.append(mse)

        print(" Done with random, took {} points".format(s.idx))


    if ACTIVE:
        seeding(seed_index)
        s.reset()
        X_test, y_test = toy_data_var_complexity(n=200,p=p,high_area=high_area,std=std,
            low_freq=low_freq,high_freq=high_freq, low_mag=low_mag, high_mag=high_mag
        )

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        n = 0

        # random priming
        while s.idx < 10:
            x, y = s.point
            MT_al.add_data_point(x, y)
            n += 1
            MT_al.update_life_time(n**(1/(2+p))-1)
            if n % 5 == 0:
                mse = evaluate(MT_al, X_test, y_test)
                MT_al_MSE.append(mse)

        # active learning
        while n < n_final:

            if not MT_al._full_leaf_list_up_to_date:
                MT_al.update_leaf_lists()
                MT_al.al_set_default_var_global_var()
                MT_al.al_calculate_leaf_proportions()
                # MT_al.al_calculate_leaf_number_new_labels(n, stream=True)
            MT_al.al_calculate_sk_stream()

            props = copy.copy(MT_al.sk_stream)

            # generate new point
            x, y = s.point
            leaf_idx = MT_al._root.leaf_for_point(x).full_leaf_list_pos
            prop = props[leaf_idx]

            if prop > 0.0 and np.random.rand() < prop:
                MT_al.add_data_point(x, y)
                n += 1

                if not PLUS_ONE:
                    MT_al.update_life_time(n**(1/(2+p))-1)

                    if n % 5 == 0:
                        mse = evaluate(MT_al, X_test, y_test)
                        MT_al_MSE.append(mse)
                else:
                    MT_al.update_life_time((n + 1)**(1/(2+p))-1)

                    if n % 5 == 0:
                        mse = evaluate(MT_al, X_test, y_test)
                        MT_al_MSE.append(mse)


            else:
                MT_al.add_data_point(x, None)

            MT_al.al_calculate_sk_stream()



        print(" Done with active, took {} points".format(s.idx))

    plt.figure()
    if RANDOM:
        n_range = range(5, len(MT_rn_MSE) * 5 + 1, 5)
        plt.plot(n_range, MT_rn_MSE, label="random")
    if ACTIVE:
        n_range = range(5, len(MT_al_MSE) * 5 + 1, 5)
        plt.plot(n_range, MT_al_MSE, label="active")

    plt.legend()

    plt.savefig('test.png')

    # np.savez('streaming_trials/var_mt_' +
    #     str(seed) + '.npz',
    #     MT_al_MSE=MT_al_MSE,
    #     MT_rn_MSE=MT_rn_MSE,
    #     n_grid=n_grid,
    # )


def main():
    import sys
    assert(len(sys.argv) == 2)
    index = sys.argv[1]

    example_var_mt(index)


if __name__ == '__main__':
    main()
