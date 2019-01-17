#!python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import mimclib.plot as miplot
import matplotlib.pyplot as plt
from mimclib import ipdb
import mimclib.setutil as setutil
import itertools
from collections import OrderedDict

mpl.rc('text', usetex=True)
mpl.rc('font', **{'family': 'normal', 'weight': 'demibold',
                  'size': 15})
# mpl.rc('lines', markersize=10, markeredgewidth=1.)
# mpl.rc('markers', fillstyle='none')

def plotWorkVsMaxError(ax, runs, *args, **kwargs):
    iter_stats_args = kwargs.pop('iter_stats_args', dict())
    fnWork = kwargs.pop('fnWork', lambda itr: itr.calcTotalWork())
    fnAggError = kwargs.pop("fnAggError", np.max)
    flip = kwargs.pop('flip', False)

    if flip:
        ax.set_ylabel('Work')
        ax.set_xlabel('$L^2$ Error')
    else:
        ax.set_xlabel('Work')
        ax.set_ylabel('$L^2$ Error')

    ax.set_yscale('log')
    ax.set_xscale('log')

    def fnItrStats(run, i, in_fn=fnWork, in_flip=flip):
        itr = run.iters[i]
        work = in_fn(run, i)
        if in_flip:
            return [np.log(itr.exact_error), itr.exact_error, work,
                    itr.total_error_est]
        else:
            return [np.log(work), work, itr.exact_error,
                    itr.total_error_est]

    xy_binned = miplot.computeIterationStats(runs,
                                             fnItrStats=fnItrStats,
                                             arr_fnAgg=[np.mean, np.mean,
                                                        fnAggError, np.min],
                                             **iter_stats_args)
    if len(xy_binned) == 0:
        return None, []

    xy_binned = xy_binned[:, 1:]
    plotObj = []
    ErrEst_kwargs = kwargs.pop('ErrEst_kwargs', None)
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    Ref_ErrEst_kwargs = kwargs.pop('Ref_ErrEst_kwargs', Ref_kwargs)
    sel = np.nonzero(np.logical_and(np.isfinite(xy_binned[:, 1]), xy_binned[:, 1] >=
                                    np.finfo(float).eps))[0]
    if len(sel) == 0:
        plotObj.append(None)
    else:
        plotObj.append(miplot.plot(ax, xy_binned[:, 0], xy_binned[:, 1],
                                   xerr=0.1*xy_binned[:, 0],
                                   *args, **kwargs))

    if ErrEst_kwargs is not None:
        plotObj.append(plot(ax, xy_binned[:, 0], xy_binned[:, 2],
                            **ErrEst_kwargs))

    if len(sel) > 0 and Ref_kwargs is not None:
        plotObj.append(ax.add_line(
            FunctionLine2D.ExpLine(data=xy_binned[sel[-4:], :2],
                                   **Ref_kwargs)))
        if ErrEst_kwargs is not None:
            plotObj.append(ax.add_line(
                FunctionLine2D.ExpLine(data=xy_binned[sel, :][:,
                                                              [0,2]], **Ref_ErrEst_kwargs)))

    return xy_binned[:, :2], plotObj



def plotProfits(ax, itr, *args, **kwargs):
    work_est = kwargs.pop('work_est', 'work')
    error = itr.parent.fn.Norm(itr.calcEl())
    if work_est == 'time':
        work = itr.calcTl()
    else:
        work = itr.calcWl()

    lvls = list(itr.lvls_itr(min_dim=2))
    assert(np.all([len(l) == 2 for l in lvls]))
    lvls = np.array(lvls)
    prof = setutil.calc_log_prof_from_EW(error, work)

    max_lvl = np.max(lvls, axis=0)

    X, Y = np.meshgrid(np.arange(0, max_lvl[0]+1), np.arange(0, max_lvl[1]+1))
    data = np.zeros((max_lvl[1]+1, max_lvl[0]+1))
    data.fill(np.nan)
    prof = setutil.calc_log_prof_from_EW(error, work)
    for i, l in enumerate(lvls):
        data[l[1], l[0]] = prof[i]
    ax.contourf(X, Y, data)
    ax.set_xlabel('$\\ell_1$')
    ax.set_ylabel('$\\ell_2$')


def plotSeeds(ax, runs, *args, **kwargs):
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Dim')
    ax.set_ylabel('Error')
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    iter_idx = kwargs.pop('iter_idx', None)
    fnNorm = kwargs.pop("fnNorm", np.abs)
    if iter_idx is None:
        itr = runs[0].last_itr
    else:
        itr = runs[0].iters[iter_idx]
    El = itr.calcEl()
    inds = []
    x = []
    for d in xrange(1, itr.lvls_max_dim()):
        ei = np.zeros(d)
        ei[-1] = 1
        # if len(ei) >= 2:
        #     ei[-2] = 1
        ii = itr.lvls_find(ei)
        if ii is not None:
            inds.append(ii)
            x.append(d)
    inds = np.array(inds)
    x = np.array(x)
    line = ax.plot(x, fnNorm(El[inds]), *args, **kwargs)
    if Ref_kwargs is not None and len(x) > 1:
        ax.add_line(miplot.FunctionLine2D.ExpLine(data=line[0].get_xydata(),
                                                  linewidth=1,
                                                  **Ref_kwargs))
    return line[0].get_xydata(), [line]


def plotBestNTerm(ax, runs, *args, **kwargs):
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylabel('Error')
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    iter_idx = kwargs.pop('iter_idx', None)
    ##### TEMP
    #itr = runs[0].last_itr
    if iter_idx is None:
        itr = runs[0].last_itr
    else:
        itr = runs[0].iters[iter_idx]
    sorted_coeff = np.sort(np.abs(itr.calcEg().coefficients))[::-1]

    error = np.cumsum(np.abs(sorted_coeff[::-1]))[::-1]
    N = 2 * np.arange(1, len(sorted_coeff)+1)
    N[1] = 4
    line = ax.plot(N, error, *args, **kwargs)
    if Ref_kwargs is not None:
        sel = np.zeros(len(N), dtype=np.bool)
        #sel[np.arange(int(0.01*len(N)), int(0.03*len(N)))] = True
        sel[50:500] = True
        sel = np.logical_and(sel, error > 1e-8)
        ax.add_line(miplot.FunctionLine2D.ExpLine(data=line[0].get_xydata()[sel, :],
                                                  linewidth=1,
                                                  **Ref_kwargs))
    return line[0].get_xydata(), [line]


def plotUserData(ax, runs, *args, **kwargs):
    which = kwargs.pop('which', 'cond').lower()
    def fnItrStats(run, i):
        itr = run.iters[i]
        max_cond = np.max([d.max_cond for d in itr.userdata])
        max_size = np.max([d.matrix_size for d in itr.userdata])
        return [i, max_size, max_cond]

    xy_binned = miplot.computeIterationStats(runs,
                                             work_bins=None,
                                             filteritr=miplot.filteritr_all,
                                             fnItrStats=fnItrStats,
                                             arr_fnAgg=[np.mean,
                                                        np.mean,
                                                        np.mean])

    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    if which == 'cond':
        ax.set_ylabel('Condition')
        line, = ax.plot(xy_binned[:, 0], xy_binned[:, 2],
                        *args, **kwargs)
    else:
        ax.set_ylabel('Matrix Size')
        line, = ax.plot(xy_binned[:, 0], xy_binned[:, 1], *args,
                        **kwargs)

    return line.get_xydata(), [line]

def plot_all(runs, **kwargs):
    runs = list(itertools.chain(*runs))
    filteritr = kwargs.pop("filteritr", miplot.filteritr_all)
    modifier = kwargs.pop("modifier", None)
    TOLs_count = len(np.unique([itr.TOL for _, itr
                                in miplot.enum_iter(runs, filteritr)]))
    convergent_count = len([itr.TOL for _, itr
                            in miplot.enum_iter(runs, miplot.filteritr_convergent)])
    iters_count = np.sum([len(r.iters) for r in runs])
    verbose = kwargs.pop('verbose', False)
    legend_outside = kwargs.pop("legend_outside", 8)
    PaperOnly = True   # If True, only produce the figures that are included in the paper


    if verbose:
        def print_msg(*args):
            print(*args)
    else:
        def print_msg(*args):
            return

    fnNorm = kwargs.pop("fnNorm", None)

    figures = []
    def add_fig(name):
        fig = plt.figure()
        fig.label = name
        fig.file_name = name
        figure.append(fig)
        return fig.gca()

    label_fmt = '{label}'
    Ref_kwargs = {'ls': '--', 'c':'k', 'label': label_fmt.format(label='{rate:.2g}')}
    ErrEst_kwargs = {'fmt': '--*','label': label_fmt.format(label='Error Estimate')}
    Ref_ErrEst_kwargs = {'ls': '-.', 'c':'k', 'label': label_fmt.format(label='{rate:.2g}')}

    # This command will produce the paper plots
    figures.extend(plotSingleLevel(runs,
                                   kwargs['input_args'],
                                   modifier=modifier,
                                   plot_individual=False,
                                   fnNorm=fnNorm,
                                   Ref_kwargs=Ref_kwargs))

    if not PaperOnly:
        print_msg("plotWorkVsMaxError")
        ax = add_fig('work-vs-max-error')
        try:
            plotWorkVsMaxError(ax, runs,
                               iter_stats_args=dict(work_spacing=np.log(np.sqrt(2)),
                                                    filteritr=filteritr),
                               fnWork=lambda run, i:
                               run.iters[i].calcTotalWork(),
                               modifier=modifier, fmt='-*',
                               Ref_kwargs=Ref_kwargs)
            ax.set_xlabel('Avg. Iteration Work')
        except:
            miplot.plot_failed(ax)
            raise

        print_msg("plotWorkVsMaxError")
        ax = add_fig('time-vs-max-error')
        try:
            plotWorkVsMaxError(ax, runs,
                               iter_stats_args=dict(work_spacing=np.log(np.sqrt(2)),
                                                    filteritr=filteritr),
                               fnWork=lambda run, i:
                               run.iters[i].calcTotalTime(),
                               modifier=modifier, fmt='-*',
                               Ref_kwargs=Ref_kwargs)
            ax.set_xlabel('Avg. Iteration Time')
        except:
            miplot.plot_failed(ax)

        print_msg("plotSeeds")
        try:
            ax = add_fig('error-vs-dim')
            plotSeeds(ax, runs, '-o', fnNorm=fnNorm,
                      label='Last iteration', Ref_kwargs=Ref_kwargs)
            plotSeeds(ax, runs, '-o', fnNorm=fnNorm,
                      Ref_kwargs=None,
                      iter_idx=int(len(runs[0].iters)/4))
        except:
            miplot.plot_failed(ax)

        print_msg("plotProfits")
        ax = add_fig('profits')
        plotProfits(ax, runs[0].last_itr)
        ax.set_title('Err/Work')

        ax = add_fig('profits')
        plotProfits(ax, runs[0].last_itr, work_est='time')
        ax.set_title('Err/Time')

        print_msg("plotUserData")
        ax = add_fig('cond-vs-iteration')
        try:
            plotUserData(ax, runs, '-o', which='cond')
        except:
            miplot.plot_failed(ax)

        ax = add_fig('size-vs-iteration')
        try:
            plotUserData(ax, runs, '-o', which='size')
        except:
            miplot.plot_failed(ax)

        print_msg("plotDirections")
        ax = add_fig('error-vs-lvl')
        #try:
        miplot.plotDirections(ax, runs, miplot.plotExpectVsLvls,
                              fnNorm=fnNorm,
                              dir_kwargs=[{'x_axis':'ell'}, {'x_axis':'ell'}])
        # except:
        #     miplot.plot_failed(ax)
        #     raise

        print_msg("plotDirections")
        ax = add_fig('work-vs-lvl')
        try:
            miplot.plotDirections(ax, runs, miplot.plotWorkVsLvls,
                                  fnNorm=fnNorm, dir_kwargs=[{'x_axis':'ell'}, {'x_axis':'ell'}])
        except:
            miplot.plot_failed(ax)
            raise

        print_msg("plotDirections")
        ax = add_fig('time-vs-lvl')
        try:
            miplot.plotDirections(ax, runs, miplot.plotTimeVsLvls,
                                  fnNorm=fnNorm, dir_kwargs=[{'x_axis':'ell'}, {'x_axis':'ell'}])
        except:
            miplot.plot_failed(ax)
            raise

        if runs[0].params.min_dim > 0 and runs[0].last_itr.lvls_max_dim() > 2:
            print("Max dim", runs[0].last_itr.lvls_max_dim())
            run = runs[0]
            from mimclib import setutil
            if run.params.qoi_example == 'sf-matern':
                profit_calc = setutil.MIProfCalculator([0.0] * run.params.min_dim,
                                                       run.params.miproj_set_xi,
                                                       run.params.miproj_set_sexp,
                                                       run.params.miproj_set_mul)
            else:
                qoi_N = run.params.miproj_max_vars
                miproj_set_dexp = run.params.miproj_set_dexp if run.params.min_dim > 0 else 0
                td_w = [miproj_set_dexp] * run.params.min_dim + [0.] * qoi_N
                hc_w = [0.] * run.params.min_dim +  [run.params.miproj_set_sexp] * qoi_N
                profit_calc = setutil.TDHCProfCalculator(td_w, hc_w)

            profits = run.last_itr._lvls.calc_log_prof(profit_calc)
            reduced_run = runs[0].reduceDims(np.arange(0, runs[0].params.min_dim),
                                             profits)    # Keep only the spatial dimensions
            print_msg("plotDirections")
            ax = add_fig('reduced-expect-vs-lvl')
            try:
                miplot.plotDirections(ax, [reduced_run],
                                      miplot.plotExpectVsLvls, fnNorm=fnNorm,
                                      dir_kwargs=[{'x_axis':'ell'}, {'x_axis':'ell'}])
            except:
                miplot.plot_failed(ax)
            print_msg("plotDirections")
            ax = add_fig('reduced-work-vs-lvl')
            try:
                miplot.plotDirections(ax, [reduced_run],
                                      miplot.plotWorkVsLvls, fnNorm=fnNorm,
                                      dir_kwargs=[{'x_axis':'ell'}, {'x_axis':'ell'}])
            except:
                miplot.plot_failed(ax)
        print_msg("plotBestNTerm")
        try:
            ax = add_fig('best-nterm')
            plotBestNTerm(ax, runs, '-o', Ref_kwargs=Ref_kwargs)
        except:
            miplot.plot_failed(ax)

        print_msg("plotWorkVsLvlStats")
        ax = add_fig('stats-vs-lvls')
        try:
            miplot.plotWorkVsLvlStats(ax, runs, '-ob',
                                      filteritr=filteritr,
                                      label=label_fmt.format(label='Total dim.'),
                                      active_kwargs={'fmt': '-*g', 'label':
                                                     label_fmt.format(label='Max active dim.')},
                                      maxrefine_kwargs={'fmt': '-sr', 'label':
                                                        label_fmt.format(label='Max refinement')})
        except:
            miplot.plot_failed(ax)

    for fig in figures:
        for ax in fig.axes:
            legend = miplot.add_legend(ax, outside=legend_outside,
                                       frameon=False, loc='lower left')
            if legend is not None:
                legend.get_frame().set_facecolor('none')
    return figures

def lower_envelope(x, cmp=lambda x,y: y<x):
    ret = [0]
    for i in range(len(x)):
        if cmp(x[ret[-1]], x[i]):
            ret.append(i)
    return ret

def to_single_run(runs, fnFilter=miplot.filteritr_all):
    all_itrs = np.array([[r, itr, itr.exact_error, itr.total_time,
                          itr.calcTotalWork(),
                          r.params.miproj_fix_lvl] for i, r, itr in
                         miplot.enum_iter_i(runs, fnFilter=fnFilter)])
    # Sort over error
    ind = np.argsort(all_itrs[:, 2])
    all_itrs = all_itrs[ind, :]

    # Lower envelope work
    all_itrs = all_itrs[lower_envelope(all_itrs[:, 4]), :]

    # Sort over work
    ind = np.argsort(all_itrs[:, 4])
    all_itrs = all_itrs[ind, :]

    # Lower envelope error
    eps = 0.1 * np.min(all_itrs[:, 2])
    all_itrs = all_itrs[lower_envelope(all_itrs[:, 2],
                                       lambda x,y: x-y > eps), :]

    all_itrs = all_itrs[::-1, :]

    # Lower envelope alpha
    all_itrs = all_itrs[lower_envelope(all_itrs[:, 5]), :]
    # Collapse iterations
    from mimclib import mimc
    new_run = mimc.MIMCRun(miproj_reuse_samples=False, confidence=0.95)

    curTime = 0
    all_itrs = all_itrs[::-1, :]
    last_run = None
    for i, itr in enumerate(all_itrs[:, 1]):  # Last iteration is not following a trend
        new_itr = itr.copy()
        new_itr.userdata = itr.userdata.copy()
        new_itr.parent = new_run

        new_itr.total_time = all_itrs[i, 3]

        # Consolidate user_data
        itr_i = itr.parent.iters.index(itr)
        while itr_i>0:
            new_itr.userdata[:0] = itr.parent.iters[itr_i-1].userdata
            itr_i -= 1

        new_run.iters.append(new_itr)

    print([r.params.miproj_fix_lvl for r in all_itrs[:, 0]])
    return new_run


def plotSingleLevel(runs, input_args, *args, **kwargs):
    # cmp_labels = ['SL', 'Adaptive ML', 'Time-Adapt ML',
    #               'TD fit ML', 'Full Adapt ML', 'TD Theory']
    # cmp_tags = [None, '-adapt', '-adapt-time',
    #             '-tdfit', '-full-adapt', '-td-theory']

    # cmp_labels = ['Single level', 'Multilevel', 'Adaptive Multilevel']
    # cmp_tags = ['SL', '-td-theory', '-adapt']

    # cmp_labels = ['SL', 'ML', 'Adaptive ML', 'Adaptive ML - Arcsine',
    #               'Adaptive ML - Discard']
    # cmp_tags = ['SL', '-td-theory', '-adapt', '-adapt-arcsine', '-adapt-discard']

    cmp_labels = ['SL', 'ML', 'Adaptive ML', 'Adaptive ML - Arcsine']
    cmp_tags = ['SL', '-theory-discard', '-adapt-discard', '-adapt-arcsine-discard']

    # cmp_labels = ['SL', 'ML']
    # cmp_tags = [None, '-td-theory']

    modifier = kwargs.pop('modifier', None)
    fnNorm = kwargs.pop('fnNorm', None)
    flip = kwargs.pop('flip', True)
    Ref_kwargs = kwargs.pop('Ref_kwargs', None)
    plot_individual = kwargs.pop('plot_individual', True)
    from mimclib import db as mimcdb
    db = mimcdb.MIMCDatabase(**input_args.db)
    print("Reading data")

    db_tag = input_args.db_tag[0]
    for t in cmp_tags:
        if t is not None and len(t) > 0 and db_tag.endswith(t):
            db_tag = db_tag[:-len(t)]
            break

    figures_dict = dict()
    axes = []
    def add_fig(name):
        if name not in figures_dict:
            fig = plt.figure()
            fig.label = name
            fig.file_name = name
            figures_dict[name] = fig
        return figures_dict[name].gca()

    plot_time_breakdown = True
    time_vars = [
        lambda b, v: v + b.sampling_time,
        lambda b, v: v + b.pt_sampling_time,
        lambda b, v: v + b.assembly_time_1,
        lambda b, v: v + b.assembly_time_2,
        lambda b, v: v + b.projection_time,
        # lambda b, v: v + b.sampling_time + b.pt_sampling_time + \
        #   b.assembly_time_1 + b.assembly_time_2 + b.projection_time,
        lambda b, v: np.maximum(v, b.matrix_size[0]),
        lambda b, v: np.maximum(v, b.matrix_size[1]),
        lambda b, v: np.maximum(v, b.gmres_counter)]

    time_vars_name = ["Sampling",
                      "PtSampling",
                      "Assembly1",
                      "Assembly2",
                      "Projection",
                      "Max Number of Points",
                      "Max Number of Basis",
                      "Max GMRES iterations"]

    def calcTime(run, i, time_var, all_itr=True):
        time_taken = 0
        if (time_var == time_vars[0] or time_var == time_vars[1]) \
           and run.params.miproj_reuse_samples:
            all_itr = True

        if run == fix_single_run[0]:
            all_itr = False   # Always consider the single run to be able to be a single iteration run

        itrs = run.iters[:(i+1)] if all_itr else run.iters[i:(i+1)]
        for i, itr in enumerate(itrs):
            for b in itr.userdata:
                time_taken = time_var(b, time_taken)
        return time_taken

    def calcTimes(run, i, time_vars, all_itr=True):
        time_taken = 0
        for v in time_vars:
            time_taken += calcTime(run, i, v, all_itr)
        return time_taken

    fnTimes = []
    fnTimes.append([
        "work-est-vs-error",
        lambda run, i: np.sum([run.iters[j].calcTotalWork() for j in range(i+1)])
               if run.params.miproj_reuse_samples else
               run.iters[i].calcTotalWork(),
        "-", None, True, True])

    fnTimes.append([
        "total-time-vs-error",
        lambda run, i: run.iter_total_times[i]
        if run.params.miproj_reuse_samples else
        run.iters[i].total_time,
        "-", None, True, True])

    if plot_time_breakdown:
        # fnTimes.append(["times-vs-error-A-alg",
        #                 lambda run, i, vv=time_vars[0]: calcTime(run, i, vv),
        #                 "-", "Sampling (solid) vs Other (dashed), Algorithm", True, True])

        # fnTimes.append(["times-vs-error-A-est",
        #                 lambda run, i, vv=time_vars[0]: calcTime(run, i, vv, False),
        #                 "-", "Sampling (solid) vs Other (dashed), Estimate", True, True])

        # fnTimes.append(["times-vs-error-A-alg",
        #                 lambda run, i, vv=time_vars[1:5]: calcTimes(run, i, vv),
        #                 "--", None, False, False])

        # fnTimes.append(["times-vs-error-A-est",
        #                 lambda run, i, vv=time_vars[1:5]: calcTimes(run, i, vv, False),
        #                 "--", None, False, False])

        for j, v in enumerate(time_vars):
            if j < 5:
                continue

            # fnTimes.append(['times-vs-error-%d-alg' % j,
            #                 lambda run, i, vv=v: calcTime(run, i, vv),
            #                 "-", "%s - Algorithm" %  time_vars_name[j], True, False])
            fnTimes.append(['times-vs-error-%d-est' % j,
                            lambda run, i, vv=v: calcTime(run, i, vv, False),
                            "-", "%s - Estimate" %  time_vars_name[j], True, False])


    for i in range(0, len(fnTimes)):
        ax = add_fig(fnTimes[i][0])
        if fnTimes[i][3] is not None:
            ax.set_title(fnTimes[i][3])

    fix_runs = []
    while True:
        fix_tag = db_tag + "-fix-" + str(len(fix_runs))
        run_data = db.readRuns(tag=fix_tag,
                               done_flag=input_args.get("done_flag", None))
        if len(run_data) == 0:
            print("Couldn't get", fix_tag)
            break
        print("Got", fix_tag)
        assert(len(run_data) == 1)
        fix_runs.append(run_data[0])

    fix_single_run = [to_single_run(fix_runs)]

    cmp_runs = [None] * len(cmp_tags)
    for i, subtag in enumerate(cmp_tags):
        if i == 0:
            cmp_runs[i] = fix_single_run
        elif db_tag + subtag == input_args.db_tag[0]:
            cmp_runs[i] = runs
        else:
            cmp_runs[i] = db.readRuns(tag=db_tag + subtag,
                                      done_flag=input_args.get("done_flag", None))
            if len(cmp_runs[i]) == 0:
                print("Couldn't get", db_tag + subtag)
            else:
                print("Got", db_tag + subtag)

    if hasattr(input_args, "qoi_exact"):
        print("Setting errors")
        fnExactErr = lambda itrs, e=input_args.qoi_exact: \
                     fnNorm([v.calcEg() + e*-1 for v in itrs])
        miplot.set_exact_errors(sum(cmp_runs, []), fnExactErr)

    iter_stats_args = OrderedDict(work_bins=None)
    if plot_individual:
        for j in range(0, 4):
            fig_T = add_fig(fnTimes[j][0])
            for rr in fix_runs:
                miplot.plotWorkVsMaxError(fig_T, [rr],
                                          flip=flip,
                                          iter_stats_args=iter_stats_args,
                                          fnWork=fnTimes[j][1],
                                          modifier=modifier, fmt=':xk',
                                          fnAggError=np.max,
                                          linewidth=2, markersize=4,
                                          #label='\\ell={}'.format(i),
                                          alpha=0.4)

    rates_ML, rates_SL = None, None
    if runs[0].params.qoi_example == 'sf-kink':
        t = 3.
        N = runs[0].params.miproj_max_vars
        alpha_r = [t, N]
        alpha = alpha_r[0]/alpha_r[1]
        gamma = 1
        beta = 1

        if gamma/beta <= 1/alpha:
            rates_ML = [alpha_r[1], alpha_r[0], 0]
        else:
            rates_ML = [gamma, beta, 0]

        if gamma/beta < 1/alpha:
            rates_ML[-1] = 2
        elif gamma/beta == 1/alpha:
            rates_ML[-1] = 3 + 1/alpha
        else:
            rates_ML[-1] = 1
        rates_SL = [alpha_r[1]*beta + alpha_r[0]*gamma, alpha_r[0]*beta, 1.]

    from math import gcd
    g = gcd(int(rates_ML[0]), int(rates_ML[1]))
    rates_ML[0] /= g
    rates_ML[1] /= g
    g = gcd(int(rates_SL[0]), int(rates_SL[1]))
    rates_SL[0] /= g
    rates_SL[1] /= g

    cycl = plt.rcParams['axes.prop_cycle']()
    for i in range(0, len(cmp_runs)):
        props = next(cycl)
        rr = cmp_runs[i]
        if rr is None or len(rr) == 0:
            continue
        label = cmp_labels[i]
        rates = rates_SL if rr == fix_single_run else (rates_ML if rr == runs else None)
        ref_ls = '-.' if rr == fix_single_run else '--'
        zorder = 10+i
        iter_stats_args = dict(work_bins=None,
                               #work_spacing=None,
                               work_spacing=np.sqrt(2)/2,
                               fnFilterData=None)

        if rates is not None:
            if rates[1] == 1:  # Denominator
                if rates[0] == 1:  # Numerator
                    base = r'\epsilon^{-1}'
                else:
                    base = r'\epsilon^{{-{:.2g}}}'.format(rates[0])
            else:
                base = r'\epsilon^{{-\frac{{ {:.2g} }}{{ {:.2g} }}}}'.format(rates[0], rates[1])

            if rates[2] == 0:
                log_factor = r''
            elif rates[2] == 1:
                log_factor = r'\log(\epsilon^{-1})'
            else:
                log_factor = r'\log(\epsilon^{{-1}})^{{{:.2g}}}'.format(rates[2])

            Ref_kwargs['label'] = '${}{}$'.format(base, log_factor)
            Ref_kwargs['ls'] = ref_ls

        for i in range(0, len(fnTimes)):
            fig_T = add_fig(fnTimes[i][0])
            data, lns = plotWorkVsMaxError(fig_T, rr, flip=True,
                                           modifier=modifier,
                                           fnWork=fnTimes[i][1],
                                           fnAggError=np.max,
                                           fmt=fnTimes[i][2],
                                           iter_stats_args=iter_stats_args,
                                           Ref_kwargs=Ref_kwargs
                                           if rates is None and
                                           rr == runs else None,
                                           zorder=zorder,
                                           label=label if fnTimes[i][4] else None,
                                           **props)
            if data is None:
                continue;
            if rates is not None and fnTimes[i][5]:
                data = data[np.argsort(data[:, 0]), :]
                def fnRate(x, rr=rates):
                    return (x)**(-rr[0]/rr[1])*np.abs(np.log(x)**rr[2])
                fig_T.add_line(miplot.FunctionLine2D(fn=fnRate,
                                                     linewidth=1,
                                                     zorder=5,
                                                     data=data[:-3, :],
                                                     **Ref_kwargs))

    # Plot time break-down -- a figure for every run
    iter_stats_args = dict(work_bins=None,
                           #work_spacing=None,
                           work_spacing=np.sqrt(2)/2,
                           fnFilterData=None)

    if plot_time_breakdown:
        for i in range(0, len(cmp_runs)):
            for j in range(0, 5):
                rr = cmp_runs[i]

                #fig_T = add_fig("time-breakdown-%d" % i)
                # plotWorkVsMaxError(fig_T, rr, flip=True,
                #                    fnWork=lambda run, i, vv=time_vars[j]: calcTime(run, i, vv),
                #                    fnAggError=np.max,
                #                    fmt='-',
                #                    iter_stats_args=iter_stats_args,
                #                    Ref_kwargs=None,
                #                    label=time_vars_name[j])
                # fig_T.set_title(cmp_tags[i] + " - Alg")

                fig_T = add_fig("time-breakdown-%d-est" % i)
                plotWorkVsMaxError(fig_T, rr, flip=True,
                                   fnWork=lambda run, i, vv=time_vars[j]: calcTime(run, i, vv, False),
                                   fnAggError=np.max,
                                   fmt='-',
                                   iter_stats_args=iter_stats_args,
                                   Ref_kwargs=None,
                                   label=time_vars_name[j])
                fig_T.set_title(cmp_tags[i] + " - Est")

    axes = [f.gca() for f in figures_dict.values()]
    if flip:
        for ax in axes:
            ax.set_xlabel('$L^2$ Error')

        axes[0].set_ylabel('Work Estimate')
        axes[1].set_ylabel('Time [s.]')
        if plot_time_breakdown:
            axes[2].set_ylabel('Time [s.]')
            for i in range(3, len(axes)):
                axes[i].set_ylabel('Time [s.]')
    else:
        for ax in axes:
            ax.set_ylabel('$L^2$ Error')

        axes[0].set_xlabel('Work Estimate')
        axes[1].set_xlabel('Time [s.]')
        if plot_time_breakdown:
            axes[2].set_xlabel('Time [s.]')
            for i in range(3, len(axes)):
                axes[i].set_xlabel('Time [s.]')
    return figures_dict.values()

if __name__ == "__main__":
    from mimclib import ipdb
    ipdb.set_excepthook()
    from mimclib.plot import run_plot_program
    run_plot_program(plot_all)
