from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from . import setutil
import itertools
import warnings
import time
from . import mimc
from scipy.linalg import solve
from scipy.sparse.linalg import lsmr, gmres, LinearOperator
from itertools import count
from collections import defaultdict
from .mimc import Bunch
from scipy.linalg import solve

__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym

import os
import ctypes as ct
import numpy.ctypeslib as npct
__lib__ = setutil.__lib__ #npct.load_library("libset_util", __file__)
__lib__.sample_optimal_leg_pts.restype = np.uint32
__lib__.sample_optimal_leg_pts.argtypes = [npct.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS'),
                                           ct.c_uint32,
                                           ct.c_voidp,
                                           npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS'),
                                           ct.c_double, ct.c_double]
__lib__.sample_optimal_random_leg_pts.restype = np.uint32
__lib__.sample_optimal_random_leg_pts.argtypes = [ct.c_uint32,
                                                  npct.ndpointer(dtype=np.uint32, ndim=1, flags='CONTIGUOUS'),
                                                  ct.c_uint32, ct.c_voidp,
                                                  npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS'),
                                                  ct.c_double, ct.c_double]

__lib__.evaluate_legendre_basis.restype = None
__lib__.evaluate_legendre_basis.argtypes = [ct.c_voidp,
                                            ct.c_uint32,
                                            ct.c_uint32,
                                            npct.ndpointer(dtype=np.double,
                                                           ndim=1,
                                                           flags='CONTIGUOUS'),
                                            ct.c_uint32,
                                            ct.c_uint32,
                                            npct.ndpointer(dtype=np.double,
                                                           ndim=1,
                                                           flags='CONTIGUOUS')]

__lib__.max_deg.restype = None
__lib__.max_deg.argtypes = [ct.c_voidp, ct.c_uint32,
                            npct.ndpointer(dtype=setutil.ind_t,
                                           ndim=1, flags='CONTIGUOUS')]

__lib__.init_matvec.restype = ct.c_voidp
__lib__.init_matvec.argtypes = [ct.c_voidp,
                                npct.ndpointer(dtype=np.double,
                                               ndim=1, flags='CONTIGUOUS'),
                                ct.c_uint32, ct.c_uint32, ct.c_bool]


__lib__.matvec_legendre_basis.restype = None
__lib__.matvec_legendre_basis.argtypes = [ct.c_voidp,
                                          npct.ndpointer(dtype=np.double,
                                                         ndim=1, flags='CONTIGUOUS'),
                                          ct.c_bool,
                                          ct.c_bool,
                                          npct.ndpointer(dtype=np.double,
                                                         ndim=1, flags='CONTIGUOUS')]


__lib__.assemble_projection_matrix.restype = None
__lib__.assemble_projection_matrix.argtypes = [ct.c_voidp,
                                               npct.ndpointer(dtype=np.double,
                                                              ndim=1,
                                                              flags='CONTIGUOUS'),
                                               npct.ndpointer(dtype=np.double,
                                                              ndim=1,
                                                              flags='CONTIGUOUS')]



class matvec:
    def __init__(self, basis, X, matfill=True):
        X = np.array(X)
        self._handle = __lib__.init_matvec(basis._handle, X.reshape(-1),
                                           X.shape[1], len(X), not matfill)
        self.basis_count = len(basis)
        self.pt_count = len(X)
        self.matfill = matfill
        if self.matfill:
            self.B = np.empty(self.basis_count*self.pt_count)
            __lib__.assemble_projection_matrix(self._handle, X.reshape(-1), self.B)
            self.B = self.B.reshape((self.pt_count, self.basis_count))
            self.eval = self.eval_matfill
        else:
            self.eval = self.eval_matvec

    def eval_matvec(self, v, square=False, transpose=False):
        result = np.empty(self.basis_count if transpose else self.pt_count)
        assert(len(v) == (self.pt_count if transpose else self.basis_count))
        __lib__.matvec_legendre_basis(self._handle, np.array(v),
                                      square, transpose, result)
        return result

    def eval_matfill(self, v, square=False, transpose=False):
        B = self.B.transpose() if transpose else self.B
        return np.dot(B**2 if square else B, v)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle is not None:
            __lib__.free_matvec(self._handle)
            self._handle = None


@public
def evaluate_legendre_basis(basis_indices, X, basis_start=0,
                            basis_count=None):
    if basis_count is None:
        basis_count = len(basis_indices)-basis_start
    val = np.empty(basis_count * len(X))
    X = np.array(X)
    __lib__.evaluate_legendre_basis(basis_indices._handle, basis_start,
                                    basis_count,
                                    X.reshape(-1),
                                    X.shape[1], len(X), val)
    return val.reshape(len(X), basis_count)

@public
def matvec_legendre_basis(basis_indices, X, v, exponent=1.,
                          transpose=False,
                          max_deg=None):
    result = np.empty(len(basis_indices) if transpose else len(X))
    assert(len(v) == (len(X) if transpose else len(basis_indices)))
    X = np.array(X)
    if max_deg is None:
        max_deg = get_basis_max_deg(basis_indices)
    __lib__.matvec_legendre_basis(basis_indices._handle, max_deg,
                                  X.reshape(-1), np.array(v),
                                  ct.c_uint32(X.shape[1]),
                                  ct.c_uint32(len(X)), exponent,
                                  transpose, result)
    return result



@public
def get_basis_max_deg(basis_indices, dim):
    max_deg = np.empty(dim, dtype=setutil.ind_t)
    __lib__.max_deg(basis_indices._handle, dim, max_deg)
    return max_deg

@public
def sample_optimal_leg_pts(N_per_basis, bases_indices, min_dim,
                           random=False, interval=(-1, 1)):
    if random:
        totalN = np.ceil(np.sum(N_per_basis))
    else:
        N_per_basis = np.ceil(N_per_basis).astype(np.uint32)
        totalN = int(np.sum(N_per_basis))

    max_dim = int(np.maximum(min_dim, bases_indices.max_dim()))
    X = np.empty(max_dim*totalN)
    assert(len(N_per_basis) == len(bases_indices))
    if random:
        N_per_basis = np.zeros(len(N_per_basis), dtype=np.uint32)
        count = __lib__.sample_optimal_random_leg_pts(np.uint32(totalN),
                                                      N_per_basis, max_dim,
                                                      bases_indices._handle,
                                                      X, interval[0],
                                                      interval[1])
    else:
        count = __lib__.sample_optimal_leg_pts(N_per_basis.astype(np.uint32),
                                               max_dim,
                                               bases_indices._handle,
                                               X,
                                               interval[0], interval[1])
    assert(count == totalN*max_dim)
    X = X.reshape((totalN, max_dim))
    return X, N_per_basis

"""
TensorExpansion is a simple object representing a basis function and a list of
coefficients. It assumes that the basis is orthonormal
"""
@public
class TensorExpansion(object):
    def __init__(self, fnEvalBasis, base_indices, coefficients):
        self.fnEvalBasis = fnEvalBasis
        self.base_indices = base_indices
        self.coefficients = coefficients

    def __call__(self, X):
        '''
        Return approximation at specified locations.

        :param X: Locations of evaluations
        :return: Values of approximation at specified locations
        '''
        X = np.array(X)
        if len(X.shape) == 0:
            X = X[None, None] # Scalar
        elif len(X.shape) == 1:
            X = X[:, None] # vector
        return self.fnEvalBasis(self.base_indices, X).dot(self.coefficients)

    @staticmethod
    def evaluate_basis(fnBasis, base_indices, X):
        '''
        Evaluates basis polynomials at given sample locations.
        Consistency condition is: fnBasis(X, 0) = 1 for any X

        :param X: Sample locations (M, dim)
        :param base_indices: indices of basis to return (N, up_to_dim)
        :return: Basis polynomials evaluated at X, (M, N)
        :rtype: `len(X) x len(mis)` np.array
        '''
        # M Number of samples, dim is dimensions
        X = np.array(X)
        if len(X.shape) == 1:
            X.reshape((-1, 1))
        # TODO: Find a way to do this
        max_deg = base_indices.to_sparse_matrix().max(axis=0).toarray()[0]

        rdim = np.minimum(X.shape[1], base_indices.max_dim())
        values = np.ones((len(X), len(base_indices)))
        basis_values = np.empty(rdim, dtype=object)

        for d in range(0, rdim):
            basis_values[d] = fnBasis(X[:, d], max_deg[d]+1)

        for i, mi in enumerate(base_indices):
            for d, j in enumerate(mi):
                values[..., i] *= basis_values[d][:, j]
        return values

    def norm(self):
        '''
        Return L^2 norm of expansion. Assumes basis is orthonormal
        '''
        return np.sqrt(np.sum(self.coefficients**2))

    def __add__(self, other):
        if not isinstance(other, TensorExpansion):
            raise NotImplementedError();
        res_base = self.base_indices.copy()
        ind_other = res_base.set_union(other.base_indices)
        coeffs = np.zeros(len(res_base))
        coeffs[:len(self.base_indices)] = self.coefficients
        coeffs[ind_other] += other.coefficients
        return TensorExpansion(self.fnEvalBasis, res_base, coeffs)

    def __mul__(self, scale):
        return TensorExpansion(self.fnEvalBasis,
                               self.base_indices.copy(),
                               self.coefficients*scale)

    def __str__(self):
        return "<Polynomial expansion, norm={}>".format(self.norm())

"""
Maintains polynomial approximation of given function on :math:`[a,b]^d`.
Supposed to take function and maintain polynomial coefficients
"""
@public
class MIWProjSampler(object):
    class SamplesCollection(object):
        def __init__(self, min_dim=1):
            self.min_dim = min_dim
            self.beta_count = 0
            self.pols_to_beta = np.empty(0)
            self.basis = setutil.VarSizeList()
            self.pt_sampling_time = 0
            self.sampling_time = 0
            self.clear_samples()

        def max_dim(self):
            return np.max([len(x) for x in self.X]) if len(self) > 0 else 0

        def clear_samples(self):
            self.X = []
            self.W = np.empty(0)
            self.Y = None
            self.N_per_basis = np.empty(0, dtype=np.uint32)
            self.sampling_time = 0
            self.pt_sampling_time = 0
            self.basis_values = None

        def update_basis_values(self, fnEvalBasis):
            if self.basis_values is None:
                self.basis_values = fnEvalBasis(self.basis, self.X)
                return self.basis_values

            prev_basis = self.basis_values.shape[1]
            prev_pts = self.basis_values.shape[0]
            if prev_basis == len(self.basis) and prev_pts == len(self.X):
                return self.basis_values # No change

            X = np.array(self.X)

            A = fnEvalBasis(self.basis, X[prev_pts:, :], 0,
                            prev_basis) if len(X) > prev_pts else None
            B = self.basis_values
            C = fnEvalBasis(self.basis, X, prev_basis) if len(self.basis) > prev_basis else None

            if A is None:
                self.basis_values = np.block([B, C])
            elif C is None:
                self.basis_values = np.block([[B], [A]])
            else:
                self.basis_values = np.block([[B, C[:prev_pts, :]],
                                              [A, C[prev_pts:, :]]])
            return self.basis_values


        def add_points(self, fnSample, alphas, X):
            self.X.extend(X.tolist())
            if self.Y is None:
                self.Y = [np.zeros(0) for i in range(len(alphas))]
            assert(len(self.Y) == len(alphas))
            for i in range(0, len(alphas)):
                self.Y[i] = np.hstack((self.Y[i], fnSample(alphas[i], X)))

        def __len__(self):
            return len(self.X)

    def __init__(self, d=0,  # d is the spatial dimension
                 min_dim=1, # Minimum stochastic dimensions
                 max_dim=None,
                 fnEvalBasis=None,
                 fnSamplesCount=None,
                 fnSamplePoints=None,
                 fnBasisFromLvl=None,
                 fnWeightPoints=None,
                 fnWorkModel=None,
                 fnGetProjector=None,
                 reuse_samples=False, proj_sample_ratio=0):
        self.fnEvalBasis = fnEvalBasis
        # Returns samples count of a projection index to ensure stability
        self.fnSamplesCount = fnSamplesCount if fnSamplesCount is not None else default_samples_count
        # Returns point sample and their weights
        self.fnSamplePoints = fnSamplePoints
        self.fnGetProjector = fnGetProjector
        self.fnWeightPoints = fnWeightPoints
        self.fnBasisFromLvl = fnBasisFromLvl if fnBasisFromLvl is not None else default_basis_from_level
        self.d = d   # Spatial dimension
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.alpha_ind = np.zeros(0)
        self.fnWorkModel = fnWorkModel

        self.alpha_dict = defaultdict(lambda c=count(0): next(c))
        self.lvls = None

        self.prev_samples = defaultdict(lambda: MIWProjSampler.SamplesCollection(self.min_dim))
        self.reuse_samples = reuse_samples
        self.proj_sample_ratio = proj_sample_ratio
        self.method = 'gmres'
        self.user_data = []

    def init_mimc_run(self, run):
        run.params.M0 = np.array([0])
        run.params.reuse_samples = False
        run.params.lsq_est = False
        run.params.moments = 1

    def update_index_set(self, lvls):
        if self.lvls is None:
            self.lvls = lvls
        assert(self.lvls == lvls)
        new_items = len(lvls) - len(self.alpha_ind)
        assert(new_items >= 0)
        new_alpha = lvls.sublist(np.arange(0, new_items) +
                                 len(self.alpha_ind)).to_dense_matrix(d_start=0, d_end=self.d)
        self.alpha_ind = np.hstack((self.alpha_ind,
                                    np.array([self.alpha_dict[tuple(k)] for
                                              k in new_alpha])))

    def estimateWork(self):
        total_work = 0
        for alpha, ind in self.alpha_dict.items():
            work_per_sample = self.fnWorkModel(setutil.VarSizeList([alpha]))[0]
            sel_lvls = np.nonzero(self.alpha_ind == ind)[0]
            beta_indset = self.lvls.sublist(sel_lvls, d_start=self.d, min_dim=0)
            basis = setutil.VarSizeList()
            for i, beta in enumerate(beta_indset):
                new_b = self.fnBasisFromLvl(beta)
                if isinstance(new_b, setutil.VarSizeList):
                    basis.set_union(new_b)
                else:
                    basis.add_from_list(new_b)
            total_samples = np.sum(self.fnSamplesCount(basis))
            total_work += work_per_sample * total_samples + self.proj_sample_ratio * len(basis) * total_samples
        return total_work

    #@profile
    def sample_all(self, run, lvls, M, moments, fnSample):
        assert np.all(moments == 1), "miproj only support first moments"
        assert np.all(M == 1), "miproj only supports M=1 exactly"
        assert(self.lvls == lvls) # Assume the levels are the same
        assert(len(self.alpha_ind) == len(lvls))
        psums_delta = np.empty((len(lvls), 1), dtype=TensorExpansion)
        psums_fine = np.empty((len(lvls), 1), dtype=TensorExpansion)
        total_time = np.zeros(len(lvls))
        total_work = np.zeros(len(lvls))
        tol = 1e-7
        max_cond = np.nan

        # Add previous times and works from previous iteration
        timer = mimc.Timer(clock=time.time)
        for alpha, ind in self.alpha_dict.items():
            sampling_time = 0
            pt_sampling_time = 0
            assembly_time_1 = 0
            assembly_time_2 = 0
            projection_time = 0

            timer.tic()
            sam_col = self.prev_samples[ind]
            sel_lvls = np.nonzero(self.alpha_ind == ind)[0]
            work_per_sample = self.fnWorkModel(setutil.VarSizeList([alpha]))[0]

            # self.fnBasisFromLvls(sel_lvls,
            #                      sam_col.basis,
            #                      sam_col.pols_per_beta)

            beta_indset = lvls.sublist(sel_lvls[sam_col.beta_count:],
                                       d_start=self.d, min_dim=0)
            for i, beta in enumerate(beta_indset):
                new_b = self.fnBasisFromLvl(beta)
                if isinstance(new_b, setutil.VarSizeList):
                    sam_col.basis.set_union(new_b)
                else:
                    sam_col.basis.add_from_list(new_b)
                sam_col.pols_to_beta = np.concatenate((sam_col.pols_to_beta,
                                                      [sam_col.beta_count]*len(new_b)))
                sam_col.beta_count += 1

            if len(sam_col.basis) > 30000:
                raise MemoryError("Too many basis functions {}".format(len(sam_col.basis)))

            if not self.reuse_samples:
                sam_col.clear_samples()

            if sam_col.min_dim < sam_col.basis.max_dim():
                sam_col.min_dim *= 2**int(np.ceil(np.log2(sam_col.basis.max_dim() / sam_col.min_dim)))
                if self.max_dim is not None:
                    sam_col.min_dim = np.minimum(sam_col.min_dim, self.max_dim)
                assert(len(sam_col.X) == 0)
                sam_col.clear_samples()

            N_per_basis = self.fnSamplesCount(sam_col.basis)
            mods, sub_alphas = mimc.expand_delta(alpha)
            N_todo = N_per_basis.copy()
            if len(sam_col.N_per_basis) > 0:
                N_todo[:len(sam_col.N_per_basis)] -= sam_col.N_per_basis
                N_todo = np.maximum(0, N_todo)

            todoN_per_beta = np.zeros(sam_col.beta_count)
            totalN_per_beta = np.zeros(sam_col.beta_count)
            totalBasis_per_beta = np.zeros(sam_col.beta_count)
            for i in range(0, sam_col.beta_count):
                todoN_per_beta[i] = np.sum(N_todo[sam_col.pols_to_beta == i])
                totalN_per_beta[i] = np.sum(N_per_basis[sam_col.pols_to_beta == i])
                totalBasis_per_beta[i] = np.sum(sam_col.pols_to_beta == i)

            if np.sum(N_todo) > 0:
                #print("Sampling points", np.sum(N_todo))
                X, N_done = self.fnSamplePoints(N_todo, sam_col.basis, sam_col.min_dim)
                N_done[:len(sam_col.N_per_basis)] += sam_col.N_per_basis
                sam_col.N_per_basis = N_done
                pt_sampling_time = timer.toc()
                #print("Sampling points, took", pt_sampling_time)

                #print("computing samples", len(X))
                timer.tic()
                sam_col.add_points(fnSample, sub_alphas, X)
                sampling_time = timer.toc()
                #print("computing samples, took", sampling_time)
            else:
                pt_sampling_time = timer.toc()

            #print("Assembling: ", len(sam_col.basis), len(sam_col.X))
            #print("Took: ", assembly_time_1)

            timer.tic()
            B_matvec, B_rmatvec, W = self.fnGetProjector(sam_col.basis, sam_col.X)
            sqrtW = np.sqrt(W)
            assembly_time_1 = timer.toc()
            timer.tic()

            # timer.tic()
            # B = sam_col.update_basis_values(self.fnEvalBasis)
            # old_assembly_time_1 = timer.toc()
            # print("Old", old_assembly_time_1, "New", assembly_time_1)
            # W = self.fnWeightPoints(sam_col.X, B)
            # sqrtW = np.sqrt(W)
            # def B_matvec(v, B=B):
            #     return np.dot(B, v)
            # def B_rmatvec(v, BT=B.tranpose()):
            #     return np.dot(BT, v)

            if self.method == 'lsmr':
                def matvec(v, fn=B_matvec, sW=sqrtW):
                    return sW*fn(v)
                def rmatvec(v, fn=B_rmatvec, sW=sqrtW):
                    return fn(sW*v)
                G = LinearOperator((len(sam_col.X), len(sam_col.basis)),
                                   matvec=matvec, rmatvec=rmatvec)
            elif self.method == 'gmres':
                def matvec(v, fn=B_matvec, fnr=B_rmatvec, WW=W):
                    return fnr(WW*fn(v))
                def rmatvec(v, fn=B_rmatvec, fnr=B_rmatvec, sW=sqrtW):
                    return sW*fn(fnr(sW*v))
                G = LinearOperator((len(sam_col.basis), len(sam_col.basis)),
                                   matvec=matvec, rmatvec=rmatvec)

            assembly_time_2 = timer.toc()
            max_cond = np.nan
            # # This following operation is only needed for diagnosis purposes
            # try:
            #     GFull = G if self.direct else BW.transpose().dot(BW)
            #     max_cond = np.linalg.cond(GFull)
            # except:
            #     pass

            projection_time = 0
            for i in range(0, len(sub_alphas)):
                timer.tic()
                # Add each element separately
                if self.method == 'lsmr':
                    coeffs, *info = lsmr(G, sqrtW * sam_col.Y[i], atol=tol, btol=tol)
                    solver_itr_count = info[1]
                elif self.method == 'gmres':
                    class gmres_counter(object):
                        def __init__(self):
                            self.niter = 0
                        def __call__(self, rk=None):
                            self.niter += 1

                    gcounter = gmres_counter()
                    coeffs, info = gmres(G, B_rmatvec(sam_col.Y[i] * W),
                                         tol=tol, atol=tol,
                                         callback=gcounter)
                    solver_itr_count = gcounter.niter
                    assert(info == 0)
                else:
                    coeffs = solve(G, B_rmatvec(sam_col.Y[i] * W),
                                   sym_pos=True)
                projection_time += timer.toc()

                # print("Done Projecting")
                projections = np.empty(sam_col.beta_count, dtype=TensorExpansion)
                for j in range(0, sam_col.beta_count):
                    # if len(beta_indset[j]) == 0:
                    #     sel_coeff = np.ones(len(coeffs), dtype=np.Boole)
                    # else:
                    sel_coeff = sam_col.pols_to_beta == j
                    projections[j] = TensorExpansion(fnEvalBasis=self.fnEvalBasis,
                                                     base_indices=sam_col.basis.sublist(sel_coeff),
                                                     coefficients=coeffs[sel_coeff])
                # assert(np.all(np.sum(projections).coefficients == coeffs))
                # assert(len(sam_col.basis.set_diff(np.sum(projections).base_indices)) == 0)
                if i == 0:
                    psums_delta[sel_lvls, 0] = projections*mods[i]
                    psums_fine[sel_lvls, 0] = projections
                else:
                    psums_delta[sel_lvls, 0] += projections*mods[i]

            # For now, only compute sampling time
            sam_col.sampling_time += sampling_time
            sam_col.pt_sampling_time = pt_sampling_time
            time_taken = sam_col.sampling_time + sam_col.pt_sampling_time
            #time_taken += assembly_time_1 + assembly_time_2 + projection_time
            total_time[sel_lvls] = time_taken * totalN_per_beta / np.sum(totalN_per_beta)
            total_time[sel_lvls] += assembly_time_1 + assembly_time_2 + projection_time
            total_work[sel_lvls] = work_per_sample * totalN_per_beta + \
                                   self.proj_sample_ratio * np.cumsum(totalN_per_beta) * np.cumsum(totalBasis_per_beta)

            # print("Sampling: ", sampling_time,
            #       "Assembly: ", assembly_time_1+assembly_time_2,
            #       "Projection: ", projection_time,
            #       "iteration count:", solver_itr_count,
            #       "mat_vec", matvec_timer.val_time, "/" , matvec_timer.val_count)
            # if projection_time > 12:
            #     raise Exception("Hello")
            self.user_data.append(Bunch(alpha=alpha,
                                        gmres_counter=solver_itr_count,
                                        max_cond=max_cond,
                                        work_per_sample=work_per_sample,
                                        matrix_size=(len(sam_col.basis),
                                                     len(sam_col.X)),
                                        todoN_per_beta=todoN_per_beta,
                                        sampling_time=sampling_time,
                                        pt_sampling_time=pt_sampling_time,
                                        assembly_time_1=assembly_time_1,
                                        assembly_time_2=assembly_time_2,
                                        projection_time=projection_time))
        return M, psums_delta, psums_fine, total_time, total_work

    @staticmethod
    def weighted_least_squares(Y, W, basisvalues):
        '''
        Solve least-squares system.
        :param Y: sample values
        :param W: weights
        :param basisvalues: polynomial basis values
        :return: coefficients
        '''
        R = basisvalues.transpose().dot(Y * W)
        G = basisvalues.transpose().dot(basisvalues * W[:, None])
        cond = np.linalg.cond(G)
        if cond > 100:
            warnings.warn('Ill conditioned Gramian matrix encountered, cond={}'.format(np.linalg.cond(G)))
        # Solving normal equations is faster than QR, because of good condition
        coefficients = solve(G, R, sym_pos=True)
        if not np.isfinite(coefficients).all():
            warnings.warn('Numerical instability encountered')
        return coefficients, cond

# @public
# def sample_optimal_pts(fnBasis, N_per_basis,
#                        bases_indices, min_dim,
#                        interval=(0, 1)):
#     assert(len(N_per_basis) == len(bases_indices))
#     N = np.ceil(np.sum(N_per_basis))
#     max_dim = np.maximum(min_dim, bases_indices.max_dim())
#     acceptanceratio = 1./(4*np.exp(1))
#     X = np.zeros((N, max_dim))
#     with np.errstate(divide='ignore', invalid='ignore'):
#         pol = np.random.randint(0, len(bases_indices))
#         # for pol in range(0, len(bases_indices)):
#         #     for i in range(N_per_basis[pol]):
#         for i in range(N):
#             base_pol = bases_indices.get_item(pol, max_dim)
#             for dim in range(max_dim):
#                 accept=False
#                 while not accept:
#                     Xnext = (np.cos(np.pi * np.random.rand()) + 1) / 2
#                     dens_prop_Xnext = 1 / (np.pi * np.sqrt(Xnext*(1 - Xnext)))   # TODO: What happens if Xnext is 0
#                     Xreal = interval[0] + Xnext *(interval[1] - interval[0])
#                     dens_goal_Xnext = fnBasis(np.array([Xreal]), 1+base_pol[dim])[0,-1] ** 2
#                     alpha = acceptanceratio * dens_goal_Xnext / dens_prop_Xnext
#                     U = np.random.rand()
#                     accept = (U < alpha)
#                     if accept:
#                         X[i,dim] = Xreal
#     return X

@public
def sample_arcsine_pts(N_per_basis, bases_indices, min_dim, interval=(-1, 1)):
    N_done = np.ceil(N_per_basis).astype(int)
    N = np.sum(N_done)
    max_dim = int(np.maximum(min_dim, bases_indices.max_dim()))
    X_temp = (np.cos(np.pi * np.random.rand(N, max_dim)) + 1) / 2
    return interval[0]+X_temp*(interval[1]-interval[0]), N_done

@public
def arcsine_weights(X, interval=(-1, 1)):
    X = np.array(X)
    return np.prod(np.pi * np.sqrt((X-interval[0])*(interval[1]-X)), axis=1)

@public
def optimal_weights(basis_values):
    if basis_values.shape[0] == 0:
        return np.zeros(0)
    # TODO: Check sizes
    return basis_values.shape[1] / np.sum(basis_values**2, axis=1)


@public
def prod_basis_from_level(beta):
    # beta is zero indexed
    # max_deg = 2 ** beta + 1
    # prev_deg = max_deg
    # prev_deg = 2 ** np.maximum((beta-1), 0) + 1
    # prev_deg[beta == 0] = 0
    max_deg = 2 ** (beta+1)-1
    prev_deg = 2 ** (beta)-1
    # prev_deg = 2 ** np.maximum((beta-1), 0) + 1
    # prev_deg[beta == 0] = 0
    return list(itertools.product(*[np.arange(prev_deg[i], max_deg[i])
                                    for i in range(0, len(beta))]))

@public
def td_basis_from_level(d, beta):
    # Return basis functions from a TD set of d dimensions
    td_prof = setutil.TDFTProfCalculator(np.ones(d))
    assert(len(beta) <= 1)
    b = 0 if len(beta) == 0 else beta[0]
    basis = setutil.VarSizeList()
    basis.expand_set(td_prof, d, max_prof=2**(b+1)-1)
    profits = basis.calc_log_prof(td_prof)
    return basis.sublist(np.logical_and(profits >= 2**b-1, profits < 2**(b+1)-1))


@public
def pair_basis_from_level(beta):
    # beta is zero indexed
    max_deg = 1 + 2*np.array(beta, dtype=int)
    prev_deg = np.maximum(0, max_deg - 2)
    l = len(beta)
    return list(itertools.product(*[np.arange(prev_deg[i], max_deg[i])
                                     for i in range(0, l)]))

@public
def default_samples_count(basis, C=2):
    m = len(basis)
    if m == 1:
        return np.array([int(2*C)])
    else:
        return np.ones(len(basis), dtype=float) * C * np.log2(m+1)

@public
def chebyshev_polynomials(Xtilde, N, interval=(-1,1)):
    r'''
    Compute values of the orthonormal Chebyshev polynomials on
    :math:`([-1,1],dx/2)` in :math:`X\subset [-1,1]`

    :param X: Locations of desired evaluations
    :param N: Number of polynomials
    :rtype: numpy.array of size :code:`X.shape[0]xN`
    '''
    X= (Xtilde-(interval[1]+interval[0])/2.)/((interval[1]-interval[0])/2.)

    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = np.concatenate((np.array([1]).reshape(1,1),np.sqrt(2)*np.ones((1,deg))),axis=1)
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = 2*X * out[:, n] - out[:, n - 1]
    return out * orthonormalizer

@public
def legendre_polynomials(Xtilde, N, interval=(-1, 1)):
    r'''
    Compute values of the orthonormal Legendre polynomials on
    :math:`([-1,1],dx/2)` in :math:`X\subset [-1,1]`

    :param X: Locations of desired evaluations
    :param N: Number of polynomials
    :rtype: numpy.array of size :code:`X.shape[0]xN`
    '''
    X = (Xtilde-(interval[1]+interval[0])/2.)/((interval[1]-interval[0])/2.)
    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = np.reshape(np.sqrt(2 * (np.array(range(deg + 1))) + 1), (1, N))
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = 1. / (n + 1) * ((2 * n + 1) * X * out[:, n] - n * out[:, n - 1])
    return out * orthonormalizer

@public
def hermite_polynomials(X, N):
    r'''
    Compute values of the orthonormal Hermite polynomials on
    :math:`(\mathbb{R},\frac{1}{\sqrt{2pi}}\exp(-x^2/2)dx)` in :math:`X\subset\mathbb{R}`


    :param X: Locations of desired evaluations
    :param N: Number of polynomials
    :rtype: numpy.array of size :code:`X.shape[0]xN`
    '''
    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = 1/np.reshape([math.sqrt(math.factorial(n)) for n in range(N)], (1, N))
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = X * out[:, n] - n * out[:, n - 1]
    return out * orthonormalizer
