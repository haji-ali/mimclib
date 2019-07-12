from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#warnings.filterwarnings('error')
import numpy as np
import warnings
import warnings
import os.path
import mimclib.mimc as mimc
import mimclib.db as mimcdb
import argparse
import time

__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym


@public
class ArgumentWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def parse_known_args(parser, return_unknown=False):
    knowns, unknowns = parser.parse_known_args(namespace=mimc.Nestedspace())
    for a in unknowns:
        if a.startswith('-'):
            warnings.warn(ArgumentWarning("Argument {} was not used!".format(a)))

    if return_unknown:
        return knowns, unknowns
    return knowns


def CreateStandardTest(fnSampleLvl=None, fnSampleAll=None,
                       fnAddExtraArgs=None, fnInit=None, fnItrDone=None,
                       fnSeed=np.random.seed):
    warnings.formatwarning = lambda msg, cat, filename, lineno, line: \
                             "{}:{}: ({}) {}\n".format(os.path.basename(filename),
                                                       lineno, cat.__name__, msg)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-db_user", type=str,
                        action="store", help="Database User", dest="db.user")
    parser.add_argument("-db_password", type=str,
                        action="store", help="Database password", dest="db.password")
    parser.add_argument("-db_host", type=str,
                        action="store", help="Database Host", dest="db.host")
    parser.add_argument("-db_engine", type=str, default='mysql',
                        action="store", help="Database Host", dest="db.engine")
    parser.add_argument("-db_tag", type=str,
                        action="store", help="Database Tag")
    parser.add_argument("-qoi_seed", type=int,
                        action="store", help="Seed for random generator")
    parser.add_argument("-db_name", type=str, action="store", help="", dest="db.db")

    if fnAddExtraArgs is not None:
        fnAddExtraArgs(parser)
    mimc.MIMCRun.addOptionsToParser(parser)
    mimcRun = mimc.MIMCRun(**vars(parse_known_args(parser)))

    if fnSampleLvl is not None:
        fnSampleLvl = lambda inds, M, fn=fnSampleLvl: fn(mimcRun, inds, M)
        mimcRun.setFunctions(fnSampleLvl=fnSampleLvl)
    else:
        fnSampleAll = lambda lvls, M, moments, fn=fnSampleAll: \
                      fn(mimcRun, lvls, M, moments)
        mimcRun.setFunctions(fnSampleAll=fnSampleAll)

    if not hasattr(mimcRun.params, 'qoi_seed'):
        mimcRun.params.qoi_seed = np.random.randint(2**32-1)

    if fnInit is not None:
        res = fnInit(mimcRun)
        if res is not None and res < 0:
            return res

    if fnSeed is not None:
        fnSeed(mimcRun.params.qoi_seed)

    if hasattr(mimcRun.params, "db"):
        db = mimcdb.MIMCDatabase(**mimcRun.params.db)
        mimcRun.db_data = mimc.Nestedspace()
        mimcRun.db_data.run_id = db.createRun(mimc_run=mimcRun,
                                              tag=mimcRun.params.db_tag)
        if fnItrDone is None:
            def ItrDone(db=db, r_id=mimcRun.db_data.run_id, r=mimcRun):
                if r.is_itr_tol_satisfied():   # Only save iterations that have tol satisifed
                    db.writeRunData(r_id, r, iteration_idx=len(r.iters)-1)
            fnItrDone = ItrDone
        else:
            fnItrDone = lambda db=db, r_id=mimcRun.db_data.run_id, r=mimcRun, fn=fnItrDone: \
                        fn(db, r_id, r)
    elif fnItrDone is not None:
        fnItrDone = lambda r=mimcRun, fn=fnItrDone: \
                        fn(None, None, r)
    mimcRun.setFunctions(fnItrDone=fnItrDone)
    return mimcRun

def RunTest(mimcRun):
    tStart = time.clock()
    try:
        mimcRun.doRun()
    except:
        if hasattr(mimcRun.params.db, "db"):
            db = mimcdb.MIMCDatabase(**mimcRun.params.db)
            db.markRunFailed(mimcRun.db_data.run_id,
                             total_time=time.clock()-tStart)
        raise   # If you don't want to raise, make sure the following code is not executed

    if hasattr(mimcRun.params.db, "db"):
        db = mimcdb.MIMCDatabase(**mimcRun.params.db)
        db.markRunSuccessful(mimcRun.db_data.run_id,
                             total_time=time.clock()-tStart)
    return mimcRun


def RunStandardTest(fnSampleLvl=None,
                    fnSampleAll=None,
                    fnAddExtraArgs=None,
                    fnInit=None,
                    fnItrDone=None,
                    fnSeed=np.random.seed):
    mimcRun = CreateStandardTest(fnSampleLvl=fnSampleLvl,
                                 fnSampleAll=fnSampleAll,
                                 fnAddExtraArgs=fnAddExtraArgs,
                                 fnInit=fnInit, fnItrDone=fnItrDone,
                                 fnSeed=fnSeed)
    return RunTest(mimcRun)

def init_post_program():
    from . import db as mimcdb
    import argparse
    import warnings
    import os
    warnings.formatwarning = lambda msg, cat, filename, lineno, line: \
                             "{}:{}: ({}) {}\n".format(os.path.basename(filename),
                                                       lineno, cat.__name__, msg)
    try:
        from matplotlib.cbook import MatplotlibDeprecationWarning
        warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
    except:
        pass   # Ignore
      
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-db_name", type=str, action="store",
                        help="Database Name", dest="db.db")
    parser.add_argument("-db_engine", type=str, action="store",
                        help="Database Name", dest="db.engine")
    parser.add_argument("-db_user", type=str, action="store",
                        help="Database User", dest="db.user")
    parser.add_argument("-db_host", type=str, action="store",
                        help="Database Host", dest="db.host")
    parser.add_argument("-db_password", type=str, action="store",
                        help="Database Password", dest="db.password")
    parser.add_argument("-db_tag", type=str, action="store",
                        help="Database Tags")
    # parser.add_argument("-qoi_exact_tag", type=str, action="store")
    # parser.add_argument("-qoi_exact", type=float, action="store", help="Exact value")
    
    args = parse_known_args(parser)
    db = mimcdb.MIMCDatabase(**args.db)
    if not hasattr(args, "db_tag"):
        warnings.warn("You did not select a database tag!!")
    print("Reading data")

    run_data = db.readRuns(tag=args.db_tag)
    if len(run_data) == 0:
        raise Exception("No runs!!!")
    
    # fnNorm = run_data[0].fn.Norm
    # if hasattr(args, "qoi_exact_tag"):
    #     assert args.qoi_exact is None, "Multiple exact values given"
    #     exact_runs = db.readRuns(tag=args.qoi_exact_tag)
    #     from . import plot
    #     args.qoi_exact, _ = plot.estimate_exact(exact_runs)
    #     print("Estimated exact value is {}".format(args.qoi_exact))
    #     if fnExactErr is not None:
    #         fnExactErr = lambda itrs, r=exact_runs[0], fn=fnExactErr: fn(r, itrs)
    # elif fnExactErr is not None:
    #     fnExactErr = lambda itrs, fn=fnExactErr: fn(itrs[0].parent, itrs)

    # if hasattr(args, "qoi_exact") and fnExactErr is None:
    #     fnExactErr = lambda itrs, e=args.qoi_exact: \
    #                  fnNorm([v.calcEg() + e*-1 for v in itrs])

    #print("Updating errors")
    
    return db, run_data
