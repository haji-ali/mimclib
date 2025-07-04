from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
from . import setutil
import sys
import io
import dill

import hashlib
__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym

def _md5(string):
    return hashlib.md5(string.encode()).hexdigest()

def _pickle(obj, use_dill=False):
    with io.BytesIO() as f:
        if use_dill:
            dill.dump(obj, f, protocol=3)
        else:
            pickle.dump(obj, f, protocol=3)
        f.seek(0)
        return f.read()


def _unpickle(obj, use_dill=False):
    try:
        with io.BytesIO(obj) as f:
            return dill.load(f) if use_dill else pickle.load(f)
    except UnicodeDecodeError:
        # This probably means that the data was saved using the
        # old python2 pickle. Try a different encoding.
        assert(not use_dill)   # dill is not backward compatible
        with io.BytesIO(obj) as f:
            return pickle.load(f, encoding='latin1')

def _nan2none(arr):
    return [None if not np.isfinite(x) else x for x in arr]

def _none2nan(x):
    return np.nan if x is None else x

class MySQLDBConn(object):
    def __init__(self, **kwargs):
        self.connArgs = kwargs

    def __enter__(self):
        import MySQLdb
        self.conn = MySQLdb.connect(compress=True, **self.connArgs)
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, type, value, traceback):
        self.Commit()
        self.conn.close()

    def execute(self, query, params=[]):
        query = query.replace("datetime()", "now()")
        query = query.replace("?", "%s")
        self.cur.execute(query, tuple(params))
        return self.cur

    def getLastRowID(self):
        return self.cur.lastrowid

    def getRowCount(self):
        return self.cur.rowcount

    def Commit(self):
        self.conn.commit()

    @staticmethod
    def DBCreationScript(drop_db=False, db="mimc"):
        script = ""
        if drop_db:
            script += "DROP DATABASE IF EXISTS {DBName};".format(DBName=db)
        script += '''
CREATE DATABASE IF NOT EXISTS {DBName};
USE {DBName};
CREATE TABLE IF NOT EXISTS tbl_runs (
    run_id                INTEGER PRIMARY KEY AUTO_INCREMENT NOT NULL,
    creation_date           DATETIME NOT NULL,
    TOL                   REAL NOT NULL,
    done_flag            INTEGER NOT NULL,
    totalTime               REAL,
    tag                   VARCHAR(128) NOT NULL,
    params                mediumblob,
    fn                    mediumblob,
    comment               TEXT
);
CREATE VIEW vw_runs AS SELECT run_id, creation_date, TOL, done_flag, tag, totalTime, comment FROM tbl_runs;

CREATE TABLE IF NOT EXISTS tbl_iters (
    iter_id                 INTEGER PRIMARY KEY AUTO_INCREMENT NOT NULL,
    run_id                  INTEGER NOT NULL,
    TOL                     REAL,
    bias                    REAL,
    stat_error              REAL,
    exact_error              REAL,
    creation_date           DATETIME NOT NULL,
    totalTime               REAL,
    Qparams                 mediumblob,
    userdata                mediumblob,
    iteration_idx           INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES tbl_runs(run_id) ON DELETE CASCADE,
    UNIQUE KEY idx_itr_idx (run_id, iteration_idx)
);
CREATE VIEW vw_iters AS SELECT iter_id, run_id, TOL,
creation_date, bias, stat_error, exact_error, totalTime, iteration_idx FROM tbl_iters;

CREATE TABLE IF NOT EXISTS tbl_lvls (
    iter_id       INTEGER NOT NULL,
    lvl           text NOT NULL,
    lvl_hash      varchar(35) NOT NULL,
    active        INT,
    El            REAL,
    Vl            REAL,
    tT            REAL,
    tW            REAL,
    Ml            BIGINT,
    weight        REAL,
    psums_delta   mediumblob,
    psums_fine    mediumblob,
    FOREIGN KEY (iter_id) REFERENCES tbl_iters(iter_id) ON DELETE CASCADE,
    UNIQUE KEY idx_run_lvl (iter_id, lvl_hash)
);

CREATE VIEW vw_lvls AS SELECT iter_id, lvl, active, weight, El, Vl, tT, tW, Ml FROM tbl_lvls;

CREATE VIEW vw_run_sum AS SELECT vw_runs.run_id, tag,
TRUNCATE(TIMESTAMPDIFF(SECOND, vw_runs.creation_date,
        max(vw_iters.creation_date))/3600., 4) as "wall time (hours)",
TRUNCATE(sum(vw_iters.totalTime) / 3600., 4) as "totTime (hours)",
min(vw_iters.TOL) as minTOL, vw_run.TOL as target_TOL
        from vw_runs LEFT JOIN vw_iters on
vw_iters.run_id=vw_runs.run_id GROUP BY vw_runs.run_id, tag;

CREATE VIEW vw_run_sum AS SELECT run_id, tag, wall_time "wall time (hours)",
        IF(minTOL>targetTOL, since_creation, NULL) "since creation (hours)",
        totTime as "totTime (hours)", minTOL, targetTOL,
        wall_time * POWER(minTOL / targetTOL, 2) as least_time
        from vw_run_sum_internal;

CREATE VIEW vw_run_sum_internal AS SELECT vw_runs.run_id as run_id, tag,
        TRUNCATE(TIMESTAMPDIFF(SECOND, vw_runs.creation_date, max(vw_iters.creation_date))/3600., 4)
        as wall_time,
        TRUNCATE(TIMESTAMPDIFF(SECOND, vw_runs.creation_date, CURRENT_TIME())/3600., 4)
        as since_creation, TRUNCATE(sum(vw_iters.totalTime) / 3600., 4) as totTime, min(vw_iters.TOL) as
        minTOL, vw_runs.TOL as targetTOL from vw_runs LEFT JOIN vw_iters on
        vw_iters.run_id=vw_runs.run_id GROUP BY vw_runs.run_id, tag;

-- CREATE USER 'USER'@'%';
-- GRANT ALL PRIVILEGES ON *.* TO 'USER'@'%' WITH GRANT OPTION;
'''.format(DBName=db)
        return script


class SQLiteDBConn(object):
    def __init__(self, **kwargs):
        if "db" in kwargs:
            kwargs["database"] = kwargs.pop("db")
        self.connArgs = kwargs
        if "database" in kwargs:
            import os.path
            if not os.path.isfile(kwargs.get("database")):
                with self:
                    self.execute(SQLiteDBConn.DBCreationScript())
        with self:
            self.execute("PRAGMA foreign_keys = ON;")

    def __enter__(self):
        import sqlite3
        self.conn = sqlite3.connect(**self.connArgs)
        self.conn.text_factory = str
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, type, value, traceback):
        self.Commit()
        self.conn.close()

    def execute(self, query, params=[]):
        if len(params) > 0 and len(query.split(';')) > 1:
            raise Exception("Multiple queries with parameters is unsupported")

        # Expand lists in paramters
        prev = -1
        new_params = []
        for p in params:
            prev = query.find('?', prev+1)
            if type(p) in [np.uint16, np.uint32, np.uint64]:
                new_params.append(np.int64(p))  # sqlite is really fussy about this
            elif type(p) in [list, tuple]:
                rep = "(" + ",".join("?"*len(p)) + ")"
                query = query[:prev] + rep + query[prev+1:]
                prev += len(rep)
                new_params.extend(p)
            else:
                new_params.append(p)

        for q in query.split(';'):
            self.cur.execute(q, tuple(new_params))
        return self.cur

    def getLastRowID(self):
        return self.cur.lastrowid

    def getRowCount(self):
        return self.cur.rowcount

    def Commit(self):
        self.conn.commit()

    @staticmethod
    def DBCreationScript():
        script = '''
CREATE TABLE IF NOT EXISTS tbl_runs (
    run_id                INTEGER PRIMARY KEY NOT NULL,
    creation_date           DATETIME NOT NULL,
    TOL                   REAL NOT NULL,
    done_flag            INTEGER NOT NULL,
    totalTime               REAL,
    tag                   VARCHAR(128) NOT NULL,
    params                blob,
    fn                    blob,
    comment               TEXT
);
CREATE VIEW vw_runs AS SELECT run_id, creation_date, TOL, done_flag, tag, totalTime, comment FROM tbl_runs;

CREATE TABLE IF NOT EXISTS tbl_iters (
    iter_id                 INTEGER PRIMARY KEY NOT NULL,
    run_id                  INTEGER NOT NULL,
    TOL                     REAL,
    bias                    REAL,
    stat_error              REAL,
    exact_error              REAL,
    creation_date           DATETIME NOT NULL,
    totalTime               REAL,
    Qparams                 blob,
    userdata                blob,
    iteration_idx           INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES tbl_runs(run_id) ON DELETE CASCADE,
    CONSTRAINT idx_itr_idx UNIQUE (run_id, iteration_idx)
);
CREATE VIEW vw_iters AS SELECT iter_id, run_id, TOL,
creation_date, bias, stat_error, exact_error, totalTime, iteration_idx FROM tbl_iters;

CREATE TABLE IF NOT EXISTS tbl_lvls (
    iter_id       INTEGER NOT NULL,
    lvl           text NOT NULL,
    lvl_hash      varchar(35) NOT NULL,
    active        INTEGER,
    El            REAL,
    Vl            REAL,
    tT            REAL,
    tW            REAL,
    Ml            INTEGER,
    weight        REAL,
    psums_delta   BLOB,
    psums_fine    BLOB,
    FOREIGN KEY (iter_id) REFERENCES tbl_iters(iter_id) ON DELETE CASCADE,
    CONSTRAINT idx_run_lvl UNIQUE (iter_id, lvl_hash)
);

CREATE VIEW vw_lvls AS SELECT iter_id, lvl, active, weight, El, Vl, tT, tW, Ml FROM tbl_lvls;

CREATE VIEW vw_run_sum AS SELECT vw_runs.run_id, tag,
(julianday(max(vw_iters.creation_date))-
julianday(vw_runs.creation_date))*24.0 as "wall time (hours)",
sum(vw_iters.totalTime) / 3600 as "totTime (hours)",
min(vw_iters.TOL) as minTOL from vw_runs INNER JOIN vw_iters on
vw_iters.run_id=vw_runs.run_id GROUP BY vw_runs.run_id, tag;

'''
        return script

@public
class MIMCDatabase(object):
    def __init__(self, engine='mysql', **kwargs):
        self.DBName = kwargs.pop("db", 'mimc')
        kwargs["db"] = self.DBName
        self.engine = engine
        if self.engine == "mysql":
            self.DBConn = MySQLDBConn
        elif self.engine == 'sqlite':
            self.DBConn = SQLiteDBConn
        else:
            raise Exception("Unrecognized DB engine")

        self.connArgs = kwargs.copy()

    def createRun(self, tag, TOL=None, params=None, fn=None,
                  mimc_run=None, comment=""):
        TOL = TOL or mimc_run.params.TOL
        params = params or mimc_run.params
        if fn is None and hasattr(mimc_run.fn, "Norm"):
            fn = {"Norm": mimc_run.fn.Norm}  # Only save the Norm function
        else:
            fn = dict()
        import dill
        with self.connect() as cur:
            cur.execute('''
            INSERT INTO tbl_runs(creation_date, TOL, tag, params, fn, done_flag, comment)
            VALUES(datetime(), ?, ?, ?, ?, -1, ?)''',
                        [TOL, tag, _pickle(params), _pickle(fn, use_dill=True), comment])
            return cur.getLastRowID()

    def markRunDone(self, run_id, flag, total_time=None, comment=''):
        with self.connect() as cur:
            cur.execute('''UPDATE tbl_runs SET done_flag=?, totalTime=?,
            comment = {}
            WHERE run_id=?'''.format('CONCAT(comment,  ?)' if self.engine=='mysql' else
            'comment || ?'), [flag, total_time, comment, run_id])

    def markRunSuccessful(self, run_id, total_time=None, comment=''):
        self.markRunDone(run_id, flag=1, comment=comment, total_time=total_time)

    def markRunFailed(self, run_id, total_time=None, comment='', add_exception=True):
        if add_exception:
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            if exc_obj is not None:
                comment += "{}: {}".format(exc_type.__name__, exc_obj)
        self.markRunDone(run_id, flag=0, comment=comment, total_time=total_time)

    def writeRunData(self, run_id, mimc_run, iteration_idx):
        base = 0
        iteration = mimc_run.iters[iteration_idx]
        El = mimc_run.fn.Norm(iteration.calcEl())
        Vl = iteration.Vl_estimate
        tT = iteration.tT
        tW = iteration.tW
        Ml = iteration.M

        prev_iter = mimc_run.iters[iteration_idx-1] if iteration_idx >= 1 else None
        if prev_iter is not None:
            prev_Vl = iteration.Vl_estimate
            prev_tT = iteration.tT
            prev_tW = iteration.tW
            prev_Ml = iteration.M

        with self.connect() as cur:
            cur.execute('''
INSERT INTO tbl_iters(creation_date, totalTime, TOL, bias, stat_error,
exact_error, Qparams, userdata, iteration_idx, run_id)
VALUES(datetime(), ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        _nan2none([float(iteration.total_time), float(iteration.TOL),
                                   float(iteration.bias), float(iteration.stat_error),
                                   float(iteration.exact_error)])
                        +[_pickle(iteration.Q), _pickle(iteration.userdata),
                          iteration_idx, run_id])
            iter_id = cur.getLastRowID()

            # Only add levels that are different from the
            #       previous iteration
            for k in range(0, iteration.lvls_count):
                lvl_data = _nan2none([float(El[k]), float(Vl[k]),
                                      float(tT[k]), float(tW[k]),
                                      int(Ml[k])])
                if prev_iter is not None:
                    if k < prev_iter.lvls_count:
                        if prev_iter.active_lvls[k] == iteration.active_lvls[k] and \
                           prev_iter.weights[k] == iteration.weights[k] and \
                           np.all(prev_iter.psums_delta[k, :] == iteration.psums_delta[k, :]) and \
                           np.all(prev_iter.psums_fine[k, :] == iteration.psums_fine[k, :]) and \
                           np.all(np.array(lvl_data[1:]) ==
                                  _nan2none([prev_Vl[k],
                                             prev_tT[k], prev_tW[k], prev_Ml[k]])):
                            continue         # Index is repeated as is in this iteration

                lvl = ",".join(["%d|%d" % (i, j) for i, j in
                                enumerate(iteration.lvls_get(k)) if j > base])
                cur.execute('''
INSERT INTO tbl_lvls(active, lvl, lvl_hash, psums_delta, psums_fine, iter_id, weight,  El, Vl, tT, tW, Ml)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            [int(iteration.active_lvls[k]),
                             lvl, _md5(lvl),
                             _pickle(iteration.psums_delta[k, :]),
                             _pickle(iteration.psums_fine[k, :]),
                             int(iter_id),
                             float(iteration.weights[k])] +
                            lvl_data)

    def readRunsByID(self, run_ids):
        from . import mimc
        import re
        lstruns = []
        run_ids = np.array(run_ids).astype(int).reshape(-1).tolist()
        if len(run_ids) == 0:
            return lstruns

        with self.connect() as cur:
            runAll = cur.execute(
                        '''SELECT r.run_id, r.params, r.TOL, r.comment, r.fn, r.tag, r.totalTime,
r.creation_date
                        FROM tbl_runs r WHERE r.run_id in ?''', [run_ids]).fetchall()
            iterAll = cur.execute('''
SELECT dr.run_id, dr.iter_id, dr.TOL, dr.creation_date,
        dr.totalTime, dr.bias, dr.stat_error, dr.Qparams, dr.userdata,
        dr.iteration_idx, dr.exact_error FROM tbl_iters dr WHERE dr.run_id in ?
ORDER BY dr.run_id, dr.iteration_idx
''', [run_ids]).fetchall()

            lvlsAll = cur.execute('''
            SELECT dr.iter_id, l.lvl, l.psums_delta, l.psums_fine, l.Ml,
                     l.tT, l.tW, l.Vl, l.active, l.weight
            FROM
            tbl_lvls l INNER JOIN tbl_iters dr ON
            dr.iter_id=l.iter_id INNER JOIN tbl_runs r on r.run_id=dr.run_id
            WHERE dr.run_id in ? ORDER BY dr.iter_id, l.lvl''',
                                  [run_ids]).fetchall()

        dictRuns = dict()
        import dill
        dictLvls = dict()
        dictIters = dict()
        import itertools
        for iter_id, itr in itertools.groupby(lvlsAll, key=lambda x:x[0]):
            dictLvls[iter_id] = list(itr)
        for run_id, itr in itertools.groupby(iterAll, key=lambda x: x[0]):
            dictIters[run_id] = list(itr)
        for run_data in runAll:
            run = mimc.MIMCRun(**_unpickle(run_data[1]))
            run.db_data = mimc.Bunch()
            run.db_data.finalTOL = run_data[2]
            run.db_data.comment = run_data[3]
            run.db_data.tag = run_data[5]
            run.db_data.total_time = run_data[6]
            run.db_data.creation_date = run_data[7]
            run.db_data.run_id = run_data[0]
            #run.db_data.fnNorm = run_data[4]  # Why?

            run.setFunctions(**_unpickle(run_data[4], use_dill=True))
            lstruns.append(run)
            if run.db_data.run_id not in dictIters:
                continue
            for i, data in enumerate(dictIters[run.db_data.run_id]):
                iter_id = data[1]
                if run.last_itr is not None:
                    iteration = run.last_itr.next_itr()
                else:
                    iteration = mimc.MIMCItrData(parent=run,
                                                 min_dim=run.params.min_dim,
                                                 moments=run.params.moments)
                iteration.TOL = data[2]
                iteration.db_data = mimc.Bunch()
                iteration.db_data.iter_id = iter_id
                iteration.userdata = _unpickle(data[8])
                iteration.db_data.creation_date = data[3]
                iteration.db_data.iter_idx = data[9]
                iteration.total_time = data[4]
                iteration.bias = _none2nan(data[5])
                iteration.stat_error = _none2nan(data[6])
                iteration.exact_error = _none2nan(data[10])
                iteration.Q = _unpickle(data[7])
                run.iters.append(iteration)
                if iter_id not in dictLvls:
                    continue
                for l in dictLvls[iter_id]:
                    t = np.array(list(map(int, [p for p in re.split(r"[,\|]", l[1]) if p])),
                                 dtype=setutil.ind_t)
                    k = iteration.lvls_find(ind=t[1::2], j=t[::2])
                    if k is None:
                        iteration.lvls_add_from_list(inds=[t[1::2]], j=[t[::2]])
                        k = iteration.lvls_count-1
                    iteration.active_lvls[k] = l[8]
                    iteration.zero_samples(k)
                    iteration.addSamples(k, M=_none2nan(l[4]),
                                         tT=_none2nan(l[5]),
                                         tW=_none2nan(l[6]),
                                         psums_delta=_unpickle(l[2]),
                                         psums_fine=_unpickle(l[3]))
                    iteration.Vl_estimate[k] = _none2nan(l[7])
                    iteration.weights[k] = l[9]
        return lstruns

    def connect(self):
        return self.DBConn(**self.connArgs)

    def _fetchArray(self, query, params=None):
        with self.connect() as cur:
            return np.array(cur.execute(query, params if params else []).fetchall())

    def getRunsIDs(self, minTOL=None, maxTOL=None, tag=None,
                   TOL=None, from_date=None, to_date=None,
                   done_flag=None):
        qs = []
        params = []
        if done_flag is not None:
            qs.append('done_flag in ?')
            params.append(np.array(done_flag).astype(int).reshape(-1).tolist())
        if tag is not None:
            qs.append('tag LIKE ? ')
            params.append(tag)
        if minTOL is not None:
            qs.append('TOL >= ?')
            params.append(minTOL)
        if maxTOL is not None:
            qs.append('TOL <= ?')
            params.append(maxTOL)
        if TOL is not None:
            qs.append('TOL in ?')
            params.append(np.array(TOL).reshape(-1).tolist())
        if from_date is not None:
            qs.append('creation_date >= ?')
            params.append(from_date)
        if to_date is not None:
            qs.append('creation_date <= ?')
            params.append(to_date)
        wherestr = ("WHERE " + " AND ".join(qs)) if len(qs) > 0 else ''
        query = '''SELECT run_id FROM tbl_runs {wherestr} ORDER BY tag,
        TOL'''.format(wherestr=wherestr)

        ids = self._fetchArray(query, params)
        if ids.size > 0:
            return ids[:, 0]
        return ids

    def readRuns(self, minTOL=None, maxTOL=None, tag=None,
                 TOL=None, from_date=None, to_date=None,
                 done_flag=None, discard_0_itr=True):
        runs_ids = self.getRunsIDs(minTOL=minTOL, maxTOL=maxTOL,
                                   tag=tag, TOL=TOL,
                                   from_date=from_date, to_date=to_date,
                                   done_flag=done_flag)
        if len(runs_ids) == 0:
            return []
        runs = self.readRunsByID(runs_ids)
        if discard_0_itr:
            return list(filter(lambda r: len(r.iters) > 0, runs))
        return runs

    def update_exact_errors(self, runs):        
        with self.connect() as cur:
            for run in runs:
                for itr in run.iters:
                    cur.execute('''
UPDATE tbl_iters SET exact_error = ? WHERE iter_id = ?''',
                                _nan2none([itr.exact_error])
                                +[itr.db_data.iter_id])

    def deleteRuns(self, run_ids):
        if len(run_ids) == 0:
            return 0
        with self.connect() as cur:
            cur.execute("DELETE from tbl_runs where run_id in ?",
                        [np.array(run_ids).astype(int).reshape(-1).tolist()])
            return cur.getRowCount()

def export_db(tag, from_db, to_db, verbose=False):
    with from_db.connect() as from_cur:
        with to_db.connect() as to_cur:
            if verbose:
                print("Getting runs")
            runs = from_cur.execute(
                'SELECT run_id, creation_date, TOL, done_flag, tag, totalTime,\
 comment, fn, params FROM tbl_runs WHERE tag LIKE ?',
                [tag]).fetchall()
            for i, r in enumerate(runs):
                to_cur.execute('INSERT INTO tbl_runs(creation_date, TOL, \
done_flag, tag, totalTime, comment, fn, params)\
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)', r[1:])
                new_run_id = to_cur.getLastRowID()
                iters = from_cur.execute(
                    'SELECT iter_id, TOL, bias, stat_error, creation_date, \
totalTime, Qparams, userdata, iteration_idx, exact_error \
FROM tbl_iters WHERE run_id=?',
                    [r[0]]).fetchall()
                for j, itr in enumerate(iters):
                    if verbose:
                        sys.stdout.write("\rDoing itr {}/{} {}/{}".format(i, len(runs), j, len(iters)))
                        sys.stdout.flush()
                    to_cur.execute('INSERT INTO tbl_iters(run_id, TOL, bias, \
stat_error, creation_date, totalTime, Qparams, userdata, \
iteration_idx, exact_error) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                   (new_run_id, ) + itr[1:])
                    new_iter_id = to_cur.getLastRowID()
                    lvls = from_cur.execute(
                        'SELECT lvl, lvl_hash, El, Vl, tT, tW, Ml, psums_delta, \
psums_fine, active FROM tbl_lvls WHERE iter_id=?',
                        [itr[0]]).fetchall()
                    for lvl in lvls:
                        to_cur.execute('INSERT INTO tbl_lvls(iter_id, lvl, \
lvl_hash, El, Vl, tT, tW, Ml, psums_delta, psums_fine, active)\
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                       (new_iter_id, ) + lvl)
                if verbose:
                    sys.stdout.write('\n')
