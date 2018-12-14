from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os.path
import numpy as np
warnings.filterwarnings("error")
from matern import SField_Matern
from mimclib import ipdb

class MCMC():
    def __init__(self, args):
        self.args = args
        self.L0 = int(np.sqrt(self.args.data_size))   # Minimum for data
        self.sf = SField_Matern(args, nested=True)
        self.genData()
        self.exact = self.qoi(self.data_y)
        self.reset()

    def qoi(self, theta):
        return np.cos(np.sum(theta))

    def genData(self):
        self.sf.BeginRuns(np.array([self.L0+self.args.data_lvl]), args.qoi_N)
        x, y = self.sf.GetSolution(np.random.random(args.qoi_N))
        self.sf.EndRuns()
        # Want 16 data points
        D = self.args.data_size
        assert(x.shape[1] == len(y) and (len(y)+1) % D == 0)
        every = int((len(y)+1)/D)
        self.data_x = x[:, ::every]
        self.data_y = y[::every]

    def likelihood(self, theta, ell):
        assert(len(theta) == args.qoi_N)
        self.sf.BeginRuns(np.array([self.L0 + ell]), args.qoi_N)
        x, y = self.sf.GetSolution(theta)
        self.sf.EndRuns()
        self.total_solves[ell] += 1
        D = len(self.data_y)
        assert(x.shape[1] == len(y) and (len(y)+1) % D == 0)
        every = int((len(y)+1)/D)
        assert(np.all(x[0, ::every] == self.data_x[0, :]))
        val = np.exp(-0.5*np.sum((y[::every] - self.data_y)**2) / self.args.data_noise)
        return val

    def calcQoI_MC(self, L, stat_tol, Ca=2.,
                   discard=1000, M=1000, ML=0):
        assert(L <= self.args.data_lvl)

        qoi = 0
        qoi2 = 0
        totalM = 0
        sample_funcs = [self.mcmc_sample, self.mlmcmc_sample1,
                        self.mlmcmc_sample2, self.mlmcmc_sample3]
        ML = int(ML)

        for m in xrange(0, discard):
            sample_funcs[ML](L)

        while True:
            for m in xrange(0, M):
                val = self.qoi(sample_funcs[ML](L))
                qoi += val
                qoi2 += val**2
            totalM = m
            stat_err = Ca * np.sqrt(((qoi2/totalM - qoi/totalM)**2)/totalM)
            if stat_err < stat_tol:
                print("""
Value is:             {:.12f}
Stat Error is:        {:.12f}
Rejection percentage: [{}]
Rejected:             [{}]
Total:                [{}]""".format(qoi/totalM, stat_err,
                      ','.join(map(lambda x: "{:.3}".format(x), 100*self.rejection_ratio())),
                      ','.join(map(lambda x: "{:}".format(int(x)), self.rejected)),
                      ','.join(map(lambda x: "{:}".format(int(x)), self.total_solves))))
                return qoi/totalM
            M *= 2
            #print("Error is: {:.12f}, Computing an extra {} samples".format(stat_err, M))
        return qoi/totalM

    def mcmc_sample(self, L):
        theta_p = np.random.multivariate_normal(self.theta,
                                                self.args.proposal_var*np.eye(self.args.qoi_N))

        # Symmetric kernel
        alpha = np.minimum(1, self.likelihood(theta_p, L) /
                           self.likelihood(self.theta, L))
        # We don't need L+1, but this is to align the random
        # number generator with MLMCMC
        U = np.random.random(L+1)
        if U[L] <= alpha:
            self.theta = theta_p    # Accept
        else:
            # Reject
            self.rejected[L] += 1
        self.n += 1
        return self.theta

    def mlmcmc_sample1(self, L):
        theta_p = np.random.multivariate_normal(self.theta,
                                                self.args.proposal_var*np.eye(self.args.qoi_N))

        prob = []
        accept = None
        U = np.random.random(L+1)
        for ell in xrange(0, L+1):
            # Symmetric kernel
            new = np.minimum(1, self.likelihood(theta_p, ell) /
                             self.likelihood(self.theta, ell))
            prob.append(new)
            beta = (prob[-1] / prob[-2]) if len(prob) > 1 else prob[-1]
            if U[ell] > beta:
                # Reject
                self.rejected[ell] += 1
                accept = False
                break
            else:
                accept = True

        # Accept
        if accept:
            self.theta = theta_p # Accept
        self.n += 1
        return self.theta

    def mlmcmc_sample2(self, L):
        theta_p = np.random.multivariate_normal(self.theta,
                                                self.args.proposal_var*np.eye(self.args.qoi_N))

        prob = []
        accept = None
        # This is not needed, but it's to align the random number generator with MLMCMC
        U = 1-np.random.random(L+1)
        for ell in xrange(0, L+1):
            # Symmetric kernel
            new = np.minimum(1, self.likelihood(theta_p, ell) /
                             self.likelihood(self.theta, ell))
            prob.append(1-new)
            beta = (prob[-1] / prob[-2]) if len(prob) > 1 else prob[-1]
            if U[ell] > beta:
                # Accept
                accept = True
                break
            else:
                accept = False
                self.rejected[ell] += 1
        # Accept
        if accept:
            self.theta = theta_p # Accept
        self.n += 1
        return self.theta


    def mlmcmc_sample3(self, L):
        theta_p = np.random.multivariate_normal(self.theta,
                                                self.args.proposal_var*np.eye(self.args.qoi_N))

        prob = []
        accept = False
        U = np.random.random(L+1)
        for ell in xrange(0, L+1):
            # Symmetric kernel
            new = np.minimum(1, self.likelihood(theta_p, ell) /
                             self.likelihood(self.theta, ell))
            mod = prob[-1] if len(prob) > 1 else 1
            beta_1 = new / mod
            beta_2 = (1-new) / mod
            if beta_1 > 1:
                beta_1 = new
            if beta_2 > 1:
                beta_2 = new
            if beta_1 < beta_2:
                prob.append(beta_1)
                if U[ell] <= prob[-1]:
                    # Accept, check next ell
                    accept = True
                    pass
                else:
                    # Reject, break
                    accept = False
                    self.rejected[ell] += 1
                    break
            else:
                prob.append(beta_2)
                U[ell] = 1-U[ell]   # NOT NEEDED. ALIGNMENT PURPOSES. SHOULD DELETE
                if U[ell] <= prob[-1]:
                    # Reject, check next ell
                    accept = False
                    self.rejected[ell] += 1
                    pass
                else:
                    accept = True
                    break

        # Accept
        if accept:
            self.theta = theta_p # Accept
        self.n += 1
        return self.theta

    def rejection_ratio(self):
        ratio = np.empty(len(self.rejected))
        ratio.fill(np.nan)
        s = self.total_solves > 0
        ratio[s] = 2 * self.rejected[s] / (self.total_solves[s])
        # We multiply by two because we solve twice for every
        # acceptance-rejection decision
        return ratio

    def reset(self):
        self.rejected = np.zeros(1+self.args.data_lvl)
        self.total_solves = np.zeros(1+self.args.data_lvl)
        self.theta = np.zeros(self.args.qoi_N)
        self.n = 0


def addExtraArguments(parser):
    class store_as_array(argparse._StoreAction):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, np.array(values))

    parser.add_argument("-qoi_dim", type=int, default=1, action="store")
    parser.add_argument("-qoi_a0", type=float, default=0., action="store")
    parser.add_argument("-qoi_f0", type=float, default=1., action="store")
    parser.add_argument("-qoi_df_nu", type=float, default=4.5, action="store")
    parser.add_argument("-qoi_df_L", type=float, default=1., action="store")
    parser.add_argument("-qoi_df_sig", type=float, default=1., action="store")
    parser.add_argument("-qoi_N", type=int, default=10, action="store")
    parser.add_argument("-h0inv", type=int, default=2, action="store")
    parser.add_argument("-beta", type=int, default=2, action="store")
    parser.add_argument("-qoi_seed", type=int, default=1, action="store")

    parser.add_argument("-data_seed", type=int, default=0, action="store")
    parser.add_argument("-data_noise", type=float, default=0.2, action="store")
    parser.add_argument("-data_lvl", type=int, default=10, action="store")
    parser.add_argument("-data_size", type=int, default=16, action="store")
    parser.add_argument("-proposal_var", type=float, default=1e-2, action="store")
    parser.add_argument("-L", type=int, default=0, action="store")

    # NOT USED, JUST FOR COMPATIBILITY
    parser.add_argument("-qoi_problem", type=int, default=0, action="store")
    parser.add_argument("-qoi_scale", type=float, default=1., action="store")
    parser.add_argument("-qoi_sigma", type=float, default=1., action="store")
    parser.add_argument("-qoi_x0", type=float, nargs='+',
                        default=np.array([0.4,0.2,0.6]),
                        action=store_as_array)

if __name__ == "__main__":
    ipdb.set_excepthook()

    import argparse
    parser = argparse.ArgumentParser(add_help=True)
    addExtraArguments(parser)

    args, unknowns = parser.parse_known_args()

    SField_Matern.Init()
    np.random.seed(args.data_seed)   # For data generation
    mcmc = MCMC(args)
    import time
    for ML in [3]:
        mcmc.reset()
        np.random.seed(args.qoi_seed)   # For data generation
        print("Doing ML = {}".format(ML))
        timeStart = time.time()
        mcmc.calcQoI_MC(L=args.L, stat_tol=0.1, ML=ML)
        print("Took: {:.2f} seconds".format(time.time()-timeStart))
        print("-------------------------------------------------")
    SField_Matern.Final()
