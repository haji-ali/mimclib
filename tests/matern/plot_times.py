import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

def plot_times(data_in, *args, **kwargs):
    data = data_in[data_in[:, 1] == 0, :]
    exclude_work = kwargs.pop('exclude_work', -1)
    j = kwargs.pop('ind', -1)
    #x = data[:, 2]*data[:, 3]
    x = data[:, 3]
    ind = np.argsort(x)
    plt_d = np.zeros((len(data), 3))
    plt_d[:, 0] = x[ind]
    if exclude_work:
        plt_d[:, 1] = data[ind, j] / data[ind, 2]
    else:
        plt_d[:, 1] = data[ind, j]
    plt_d[:, 2] = data[ind, 3]
    from itertools import groupby

    final_d = []
    for k, itr in groupby(plt_d, key=lambda x:x[0]):
        d = np.array(list(itr))
        #final_d.append([k, np.sum(d[:, 1])/np.sum(d[:, 2])])
        final_d.append([k, np.mean(d[:, 1])])
    final_d = np.array(final_d)
    plt.loglog(final_d[:, 0], final_d[:, 1], *args, **kwargs);
    plt.ylabel('Time')
    plt.xlabel('Number of samples')
    return final_d

def plot_cond(data, *args, **kwargs):
    ind = np.argsort(data[:, 3])
    plt_d = np.zeros((len(data), 2))
    plt_d[:, 0] = data[ind,3]
    plt_d[:, 1] = data[ind, 4]
    from itertools import groupby

    final_d = []
    for k, itr in groupby(plt_d, key=lambda x:x[0]):
        d = list(itr)
        final_d.append(np.mean(d, axis=0))
    final_d = np.array(final_d)
    plt.loglog(final_d[:, 0], final_d[:, 1], *args, **kwargs);
    plt.ylabel('Condition Number')
    plt.xlabel('number of samples')

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
opt = genfromtxt('output/optimal.txt', delimiter=',')
# opt_discard = genfromtxt('output/optimal_discard.txt', delimiter=',')
# arcsine = genfromtxt('output/arcsine.txt', delimiter=',')


from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('output/time.pdf')


for data in [genfromtxt('output/optimal.txt', delimiter=','),
             genfromtxt('output/arcsine.txt', delimiter=',')]:
    fig = plt.figure()
    for ind, ylabel in [ [4, 'Sampling'], [5, 'Point Sampling'],
                         [6, 'Basis Assembly'], [7, 'Projection Assembly'],
                         [8, 'Projection']]:
        d_opt = plot_times(data, '-', ind=ind, label=ylabel, exclude_work=(ind==4))
        if ind == 4:
            c = np.mean(d_opt[-5:, 1] / d_opt[-5:, 0])
            x = np.logspace(np.log10(np.min(d_opt[:, 0])), np.log10(np.max(d_opt[:, 0])))
            plt.loglog(x, c*x, 'k--', label='1')
        #break

    plt.legend(loc='upper left')

    # fig = plt.figure()
    # plot_cond(opt, '-o', label="Optimal")
    # plot_cond(arcsine, '-*', label="Arcsine")
    # plot_cond(opt_discard, '-*', label="Optimal (discarding)")
    # plt.legend(loc='upper left')

    pdf.savefig(fig)
pdf.close()
