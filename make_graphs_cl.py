import matplotlib
matplotlib.use('AGG')
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import glob

def main():
    globby = glob.glob('graphs/cl_mt*.npz')
    print('{} data points'.format(len(globby)))

    n_finals = [100, 200, 300, 400, 500, 600, 700]

    data = None

    for file in globby:
        temp_data = np.load(file)

        if data is None:
            data = dict(temp_data)
        else:
            for key in temp_data:
                data[key] += temp_data[key]

    MT_al_MSE = data['MT_al_MSE'] / len(globby)
    MT_rn_MSE = data['MT_rn_MSE'] / len(globby)
    MT_uc_MSE = data['MT_uc_MSE'] / len(globby)

    BT_al_MSE = data['BT_al_MSE'] / len(globby)
    BT_rn_MSE = data['BT_rn_MSE'] / len(globby)
    BT_uc_MSE = data['BT_uc_MSE'] / len(globby)

    f, axarr = plt.subplots(2, sharex=True)

    mt_al = axarr[0].plot(n_finals, MT_al_MSE, color = 'red', label='Mondrian Tree - Active sampling')
    mt_rn = axarr[0].plot(n_finals, MT_rn_MSE, color = 'blue', label = 'Mondrian Tree - Random sampling')
    mt_uc = axarr[0].plot(n_finals, MT_uc_MSE, color = 'green', label = 'Mondrian Tree - Uncertainty sampling')
    axarr[0].set_title('Cl experiment, n={} trials'.format(len(globby)))
    axarr[0].legend(loc='best')

    bt_al = axarr[1].plot(n_finals, BT_al_MSE, color = 'red', linestyle = '--',
        label = 'Breiman Tree - Active sampling')
    bt_rn = axarr[1].plot(n_finals, BT_rn_MSE, color = 'blue', linestyle = '--',
        label = 'Breiman Tree - Random sampling')
    bt_rn = axarr[1].plot(n_finals, BT_uc_MSE, color = 'green', linestyle = '--',
        label = 'Breiman Tree - Uncertainty sampling')
    axarr[1].legend(loc='best')

    f.text(0.01, 0.5, 'MSE', va='center', rotation='vertical')
    f.text(0.5, 0.01, 'Final number of labelled points', ha='center')


    variance_data = {
        'MT_al_MSE': defaultdict(list),
        'MT_rn_MSE': defaultdict(list),
        'MT_uc_MSE': defaultdict(list),

        'BT_al_MSE': defaultdict(list),
        'BT_rn_MSE': defaultdict(list),
        'BT_uc_MSE': defaultdict(list),
    }


    for file in globby:
        temp_data = np.load(file)

        for key in temp_data:
            curr = temp_data[key]

            for i in range(len(curr)):
                variance_data[key][i].append(curr[i])

    for key in variance_data:
        for nidx in variance_data[key]:
            variance_data[key][nidx] = np.array(variance_data[key][nidx])


    MT_al_MSE_var = np.std(np.array(list(variance_data['MT_al_MSE'].values())), axis=1)
    MT_rn_MSE_var = np.std(np.array(list(variance_data['MT_rn_MSE'].values())), axis=1)
    MT_uc_MSE_var = np.std(np.array(list(variance_data['MT_uc_MSE'].values())), axis=1)

    BT_al_MSE_var = np.std(np.array(list(variance_data['BT_al_MSE'].values())), axis=1)
    BT_rn_MSE_var = np.std(np.array(list(variance_data['BT_rn_MSE'].values())), axis=1)
    BT_uc_MSE_var = np.std(np.array(list(variance_data['BT_uc_MSE'].values())), axis=1)


    mt_al_err = axarr[0].errorbar(n_finals, MT_al_MSE, MT_al_MSE_var, color = 'red', marker='^', capsize=10)
    mt_rn_err = axarr[0].errorbar(n_finals, MT_rn_MSE,  MT_rn_MSE_var, color = 'blue', marker='^', capsize=10)
    mt_uc_err = axarr[0].errorbar(n_finals, MT_uc_MSE, MT_uc_MSE_var, color = 'green', marker='^', capsize=10)

    bt_al_err = axarr[1].errorbar(n_finals, BT_al_MSE, BT_al_MSE_var, color = 'red', marker='^', capsize=10)
    bt_rn_err = axarr[1].errorbar(n_finals, BT_rn_MSE, BT_rn_MSE_var, color = 'blue', marker='^', capsize=10)
    bt_rn_err = axarr[1].errorbar(n_finals, BT_uc_MSE, BT_uc_MSE_var, color = 'green', marker='^', capsize=10)

    plt.tight_layout()
    plt.savefig('cl_mt.pdf')


    corrected_mt_al_vals = np.array(list(variance_data['MT_al_MSE'].values())) - np.array(list(variance_data['MT_rn_MSE'].values()))

    corrected_bt_al_vals = np.array(list(variance_data['BT_al_MSE'].values())) - np.array(list(variance_data['BT_rn_MSE'].values()))

    plt.figure()
    plt.title("CL Mondrian Trees boxplot normed MSE")
    plt.boxplot(corrected_mt_al_vals.T, labels=n_finals)
    plt.axhline(linewidth=1, color='r')
    plt.savefig('cl_corrected_boxplot_mt.png')

    plt.figure()
    plt.title("CL Breiman Trees boxplot normed MSE")
    plt.boxplot(corrected_bt_al_vals.T, labels=n_finals)
    plt.axhline(linewidth=1, color='r')
    plt.savefig('cl_corrected_boxplot_bt.png')


if __name__ == '__main__':
    main()
