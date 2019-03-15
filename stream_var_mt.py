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

PLUS_ONE = None
N_FINALS = None
DIST = None

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


    if RANDOM:
        MT_rn = Mondrian_Tree([[0,1]]*p, seed=seed)

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
        MT_al = Mondrian_Tree([[0,1]]*p, seed=seed)

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

            state = np.random.get_state()
            if prop > 0.0 and np.random.random_sample() < prop:
                np.random.set_state(state)
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
                np.random.set_state(state)
                MT_al.add_data_point(x, None)

            MT_al.al_calculate_sk_stream()



        print(" Done with active, took {} points".format(s.idx))

    # plt.figure()
    # if RANDOM:
    #     n_range = range(5, len(MT_rn_MSE) * 5 + 1, 5)
    #     plt.plot(n_range, MT_rn_MSE, label="random")
    # if ACTIVE:
    #     n_range = range(5, len(MT_al_MSE) * 5 + 1, 5)
    #     plt.plot(n_range, MT_al_MSE, label="active")

    # plt.legend()

    # plt.savefig('test.png')

    # np.savez('streaming_trials/var_mt_' +
    #     str(seed) + '.npz',
    #     MT_al_MSE=MT_al_MSE,
    #     MT_rn_MSE=MT_rn_MSE,
    #     n_grid=n_grid,
    # )

    return MT_rn_MSE, MT_al_MSE, s.idx


def main():
    parser = ArgumentParser()
    parser.add_argument(
                    'finals', action='store', default=500, type=int
                )
    parser.add_argument(
                    '--plus_one', '-p', dest='plus_one',
                    action='store_true', default=False
                )

    parser.add_argument(
                    '--dist', '-d', dest='dist', type=str,
                    action='store', choices=['het', 'var'], default='var'
                )

    args = parser.parse_args()

    global N_FINALS
    global PLUS_ONE
    global DIST

    N_FINALS = args.finals
    PLUS_ONE = args.plus_one
    DIST = args.dist

    RN_MSE = []
    AL_MSE = []
    sidxs = []

    trials = 500

    for index in range(trials):
        rn_mse, al_mse, sidx = example_var_mt(index)
        RN_MSE.append(rn_mse)
        AL_MSE.append(al_mse)
        sidxs.append(sidx)


    RN_MSE = np.array(RN_MSE)
    AL_MSE = np.array(AL_MSE)
    sidxs = np.array(sidxs)

    rn_mean = RN_MSE.mean(0)
    al_mean = AL_MSE.mean(0)

    al_MSE_var = np.std(AL_MSE, axis=0)
    rn_MSE_var = np.std(RN_MSE, axis=0)

    plt.figure()
    if RANDOM:
        n_range = range(5, len(rn_mean) * 5 + 1, 5)
        plt.plot(n_range, rn_mean, label="random")
        # plt.errorbar(n_range, rn_mean, rn_MSE_var, marker='^', capsize=5)
    if ACTIVE:
        n_range = range(5, len(al_mean) * 5 + 1, 5)
        plt.plot(n_range, al_mean, label="active")
        # plt.errorbar(n_range, al_mean, al_MSE_var, marker='^', capsize=5)

    plt.legend()
    plt.title('{}_trials_alg2vsrand_{}_upbefore'.format(trials, sidxs.mean()))
    plt.savefig('{}_trials_alg2vsrand_{}_upbefore.png'.format(trials, sidxs.mean()))


    plt.figure()
    corrected_mt_al_vals = AL_MSE - RN_MSE

    plt.title("test box")
    plt.boxplot(corrected_mt_al_vals, labels=n_range)
    plt.axhline(linewidth=1, color='r')
    plt.savefig('corrected_box_{}_trials_alg2vsrand_{}_upbefore.png'.format(trials, sidxs.mean()))


if __name__ == '__main__':
    main()
