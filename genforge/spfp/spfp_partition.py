# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

# -----------------------------------------------------------------------------
# Semantic Preserving Feature Partitioning (SPFP) - class API
# -----------------------------------------------------------------------------
# This module reproduces the original SPFP algorithm (forward selection with
# MI/|corr|/interaction/redundancy objectives, optional backward selection,
# and stream pruning) in a class you can instantiate and reuse.
#
# Dependencies:
#   - .StandardScaler
#   - .LabelEncoder
#   - .Binning.Binning                (equal-distance, parallel)
#   - .JointHistogram.JointEntropy    (pandas-based entropy via joint hist)
#
# Notes:
# - The algorithm is computed on the (optionally standardized) TRAIN matrix.
# - Test/valid arrays are optional and only used for reporting (same as script).
# - We keep the exact stopping thresholds / objective combination logic.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import math
import random
from time import time
from joblib import Parallel, delayed

from .StandardScaler import StandardScaler
from .LabelEncoder import LabelEncoder
from .Binning import Binning
from .JointHistogram import JointEntropy

# -----------------------------------------------------------------------------


def _fmt_val(val: float) -> str:
    return f"{val:.4f}" if int(val) < 10 else f"{val:.3f}"


def _floor_n(x: float, decimals: int = 4) -> float:
    m = 10 ** decimals
    return math.floor(x * m) / m


@dataclass
class SPFPPartitioner:
    # Core controls (match original defaults/semantics)
    n_groups: int = 5
    n_bins: int = 100
    wcriadd: float = 0.999
    wcrirem: float = 0.999
    whmag: float = 1.01
    perfs: float = 0.10            # minimum fraction of features per group
    remp: float = 0.60             # fraction of selected features removed from stream
    objective: str = "mi"          # {'mi','cor','micor'}
    override: bool = False         # allow overriding objective by compound 'obj'
    backward: bool = False         # backward removal loop
    standardize: bool = True
    test_size: float = 0.0         # 0 => no split inside the class (you can pass test manually)
    random_state: Optional[int] = None
    n_jobs: int = -1               # parallel for entropy pairs
    verbose: int = 1               # 0 = quiet, 1 = progress

    # Fitted / derived
    scaler_: Optional[StandardScaler] = field(default=None, init=False)
    encoder_: Optional[LabelEncoder] = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    # Data holders (train/test in the same style as original script)
    Xtr_: Optional[pd.DataFrame] = field(default=None, init=False)
    ytr_: Optional[pd.DataFrame] = field(default=None, init=False)
    Xts_: Optional[pd.DataFrame] = field(default=None, init=False)
    yts_: Optional[pd.DataFrame] = field(default=None, init=False)

    # Binned data (train/test)
    Xtr_bin_: Optional[pd.DataFrame] = field(default=None, init=False)
    Xts_bin_: Optional[pd.DataFrame] = field(default=None, init=False)
    bin_edge_: Optional[Dict[int, np.ndarray]] = field(default=None, init=False)

    # Entropy / ranking tables
    Hy_: Optional[float] = field(default=None, init=False)
    H_table_: Optional[pd.DataFrame] = field(default=None, init=False)  # columns: ind,Hx,Hxy,mi,cor,micor
    stream_: Optional[List[int]] = field(default=None, init=False)       # sorted by 'mi' desc

    # Results
    groups_: Optional[Dict[int, Dict[str, Any]]] = field(default=None, init=False)

    # -------------------------------------------------------------------------

    def set_params(self, **kwargs):
        """Update parameters to mirror sklearn-like API convenience."""
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        return self

    # -------------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | pd.DataFrame,
        X_test: Optional[np.ndarray | pd.DataFrame] = None,
        y_test: Optional[np.ndarray | pd.Series | pd.DataFrame] = None,
    ) -> "SPFPPartitioner":
        """
        Prepare the inputs (optional standardization + label encode y),
        compute binning on TRAIN, build the per-feature entropy/MI/corr table,
        and the initial 'stream' (features sorted by MI desc).
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # ---------- to DataFrame (fast) ----------
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        # Drop constant columns (exactly as original script does after transform)
        nunique = X.nunique()
        if self.verbose:
            pass  # keep quiet; same behavior as script until printed after transform
        X = X.loc[:, nunique != 1]
        X.columns = range(X.shape[1])

        # ---------- scale + encode ----------
        if self.standardize:
            self.scaler_ = StandardScaler().fit(X)
            Xs = pd.DataFrame(self.scaler_.transform(X))
        else:
            Xs = X.copy()

        self.encoder_ = LabelEncoder().fit(y.values)
        ys = pd.Series(self.encoder_.transform(y.values))

        # test (optional) follows train's scaler and encoder
        Xts, yts = None, None
        if X_test is not None and y_test is not None:
            X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test.copy()
            X_test = X_test.loc[:, nunique != 1]  # align kept columns
            X_test.columns = range(X_test.shape[1])

            Xts = pd.DataFrame(self.scaler_.transform(X_test)) if self.standardize else X_test.copy()
            yts = pd.Series(self.encoder_.transform(pd.Series(y_test).values))

        # ---------- binning (equal-distance) ----------
        Xtr_bin, bin_edge = Binning(Xs, n_bins=self.n_bins, method='equal-distance', bin_edge=None)
        Xts_bin = None
        if Xts is not None:
            Xts_bin, _ = Binning(Xts, n_bins=self.n_bins, method='equal-distance', bin_edge=bin_edge)

        # Combine with y for entropy
        dbintr = pd.DataFrame(pd.concat([Xtr_bin, ys], axis=1).values)
        dbints = pd.DataFrame(pd.concat([Xts_bin, yts], axis=1).values) if Xts_bin is not None else None
        cindex = [Xtr_bin.shape[1]]  # target col index in the concatenated DF

        # ---------- per-feature entropies & metrics ----------
        Hy = JointEntropy(pd.DataFrame(ys), [0])  # H(Y)
        rows = []
        for i in range(Xtr_bin.shape[1]):
            Hx_i = JointEntropy(dbintr, [i])  # H(X_i)
            Hxy_i = JointEntropy(dbintr, [i] + cindex)  # H(X_i, Y)
            rows.append((i, Hx_i, Hxy_i))

        H = pd.DataFrame(rows, columns=['ind', 'Hx', 'Hxy'])
        H['mi'] = (H['Hx'] + Hy - H['Hxy']) / Hy

        # correlation on standardized train X vs encoded Y (as in script)
        # (pandas vectorized corr with a Series)
        H['cor'] = Xs.corrwith(ys).values
        H['micor'] = H['mi'] + H['cor'].abs()

        # global totals (for reporting—match the script)
        Hxyt = JointEntropy(dbintr, list(range(dbintr.shape[1])))
        Hxt = JointEntropy(dbintr, list(range(dbintr.shape[1] - 1)))
        mit = (Hxt + Hy - Hxyt) / Hy
        if self.verbose:
            print(f"SPFP: MI(all)={_fmt_val(mit)} | "
                  f"mi[min]={_fmt_val(H['mi'].min())}, mi[max]={_fmt_val(H['mi'].max())} | "
                  f"Hx[min]={_fmt_val(H['Hx'].min())}, Hx[max]={_fmt_val(H['Hx'].max())} | Hy={_fmt_val(Hy)}")

        # sort stream by MI desc (exactly as script)
        H_sorted = H.sort_values(by='mi', ascending=False).reset_index(drop=True)
        stream = list(H_sorted['ind'].copy())

        # store
        self.Xtr_, self.ytr_ = Xs, pd.DataFrame(ys)
        self.Xts_, self.yts_ = Xts, (pd.DataFrame(yts) if yts is not None else None)
        self.Xtr_bin_, self.Xts_bin_ = Xtr_bin, Xts_bin
        self.bin_edge_ = bin_edge
        self.Hy_, self.H_table_ = Hy, H_sorted
        self.stream_ = stream
        self.fitted_ = True
        return self

    # -------------------------------------------------------------------------

    def partition(self) -> Dict[int, Dict[str, Any]]:
        """
        Run SPFP to produce n_groups of feature indices (findex) for multi-pop GP.
        Returns a dict like: { group_id: {'findex': [...], 'IntS': DataFrame, 'RedS': DataFrame, ...}, ...}
        """
        if not self.fitted_:
            raise RuntimeError("Call fit(...) before partition().")

        Xs = self.Xtr_
        ys = self.ytr_.iloc[:, 0]
        Xtr_bin = self.Xtr_bin_
        Xts_bin = self.Xts_bin_
        Hy = float(self.Hy_)
        H = self.H_table_.copy()
        stream = list(self.stream_)  # working copy

        # Entropy helpers
        dbintr = pd.DataFrame(pd.concat([Xtr_bin, ys], axis=1).values)
        dbints = None
        if self.Xts_ is not None and self.yts_ is not None and Xts_bin is not None:
            dbints = pd.DataFrame(pd.concat([Xts_bin, self.yts_.iloc[:, 0]], axis=1).values)
        cindex = [Xtr_bin.shape[1]]

        groups: Dict[int, Dict[str, Any]] = {}
        n_features = Xs.shape[1]

        for g in range(self.n_groups):
            if not stream:
                break

            groups[g] = {}
            groups[g]['String'] = stream.copy()

            rndtemp = stream.copy()
            # HxA and HxyA on the remaining stream (same as script)
            HxA = JointEntropy(dbintr, rndtemp)
            HxyA = JointEntropy(dbintr, rndtemp + cindex)
            mitA = (HxA + Hy - HxyA) / Hy

            if self.verbose:
                print(f"SPFP: Group {g} | pre-loop: HxA={_fmt_val(HxA)} HxyA={_fmt_val(HxyA)} mitA={_fmt_val(mitA)}")

            t = 1
            sindex: List[int] = []
            # htemp has the live objective columns; initialize like the script
            zeros_data = pd.DataFrame(
                0,
                index=range(len(rndtemp)),
                columns=['obj','utilint','objint','utilred','objred','utilent','objent','utilmi','objmi','utilcor','objcor'] + rndtemp
            )
            htemp = pd.concat([H.copy(), zeros_data], axis=1)

            # store int/red matrices per step
            intstemp = pd.DataFrame(0, index=range(htemp.shape[0]), columns=['ind','hx','hxy','mi','cor'])
            redstemp = pd.DataFrame(0, index=range(htemp.shape[0]), columns=['ind','hx','hxy','mi','cor'])

            # caches for pair entropies
            hhentind = pd.DataFrame([(i, 0, 0) for i in range(htemp.shape[0])], columns=['ind','hsfc','hsf'])
            hhred = pd.DataFrame(0, index=range(htemp.shape[0]), columns=['ind'] + rndtemp)
            hhint = pd.DataFrame(0, index=range(htemp.shape[0]), columns=['ind'] + rndtemp)
            hhentind['ind'] = htemp['ind']
            hhred['ind'] = htemp['ind']
            hhint['ind'] = htemp['ind']

            Hsy = 0.0
            Hs = 0.0
            nn = 1

            # -------------------- forward selection loop --------------------
            while (
                (Hsy < self.wcriadd * _floor_n(HxyA, 4)) or
                (Hs  < self.wcriadd * _floor_n(HxA, 4)) or
                (len(sindex) < math.ceil(self.perfs * n_features))
            ) and len(rndtemp) > 0:
                # --- candidate selection logic (exactly per script) ---
                if nn == 1:
                    candf = rndtemp[0]
                    maxobj = 0.0
                    Hstemp = JointEntropy(dbintr, sindex + [candf])
                    Hsytemp = JointEntropy(dbintr, sindex + [candf] + cindex)
                else:
                    hobj = pd.concat(
                        [htemp['ind'], htemp['obj'], htemp['cor'].abs(), htemp['micor'], htemp['mi']],
                        axis=1
                    )
                    if (
                        (Hsy >= self.wcriadd * _floor_n(HxyA, 4)) and
                        (Hs  >= self.wcriadd * _floor_n(HxA, 4)) and
                        (len(sindex) < math.ceil(self.perfs * n_features))
                    ):
                        # choose by configured primitive (mi / cor / micor)
                        if self.objective not in ('mi', 'cor', 'micor'):
                            raise ValueError("objective must be one of {'mi','cor','micor'}")
                        col = self.objective
                        maxobj = float(hobj[col].max())
                        candf = int(hobj.loc[hobj[col] == maxobj, 'ind'].iloc[0])
                        Hstemp = JointEntropy(dbintr, sindex + [candf])
                        Hsytemp = JointEntropy(dbintr, sindex + [candf] + cindex)

                    elif self.override:
                        maxobj = float(hobj['obj'].max())
                        candf = int(hobj.loc[hobj['obj'] == maxobj, 'ind'].iloc[0])
                        Hstemp = JointEntropy(dbintr, sindex + [candf])
                        Hsytemp = JointEntropy(dbintr, sindex + [candf] + cindex)

                    else:
                        dtemp = pd.DataFrame(0, index=range(htemp.shape[0]), columns=['ind','ds','dsy'])
                        dtemp['ind'] = htemp['ind']
                        # greedy suppression of 'obj' for candidates that barely move Hs/Hsy
                        for kk in range(hobj.shape[0]):
                            maxobj = float(hobj['obj'].max())
                            candf = int(hobj.loc[hobj['obj'] == maxobj, 'ind'].iloc[0])
                            Hstemp = JointEntropy(dbintr, sindex + [candf])
                            Hsytemp = JointEntropy(dbintr, sindex + [candf] + cindex)

                            if kk == 0:
                                maxobjtemp = maxobj
                                candftemp = candf
                                Hstemp2 = Hstemp
                                Hsytemp2 = Hsytemp

                            # within magnification bounds?
                            cond = (
                                ((Hsy < self.wcriadd * _floor_n(HxyA, 4)) or (Hs < self.wcriadd * _floor_n(HxA, 4)))
                                and (
                                    (Hsytemp <= min(self.whmag * Hsy, self.wcriadd * _floor_n(HxyA, 4))) or
                                    (Hstemp <= min(self.whmag * Hs,  self.wcriadd * _floor_n(HxA, 4)))
                                )
                            )
                            if cond:
                                dtemp.loc[dtemp['ind'] == candf, 'ds']  = (Hstemp - Hs)  / (self.wcriadd * _floor_n(HxA, 4))
                                dtemp.loc[dtemp['ind'] == candf, 'dsy'] = (Hsytemp - Hsy) / (self.wcriadd * _floor_n(HxyA, 4))
                                hobj.loc[hobj['ind'] == candf, 'obj'] = -np.inf
                            else:
                                break

                            if kk == (hobj.shape[0] - 1):
                                aa1 = (dtemp['ds'] > 0) & (dtemp['dsy'] > 0)
                                aa2 = (dtemp['dsy'] > 0)
                                aa3 = (dtemp['ds'] > 0)
                                if aa1.any():
                                    dtemp['sum'] = dtemp[['ds','dsy']].sum(axis=1)
                                    tmp = dtemp[aa1]
                                    candf = int(tmp.loc[tmp['sum'] == tmp['sum'].max(), 'ind'].iloc[0])
                                    maxobj = float(htemp.loc[htemp['ind'] == candf, 'obj'].iloc[0])
                                    Hstemp = JointEntropy(dbintr, sindex + [candf])
                                    Hsytemp = JointEntropy(dbintr, sindex + [candf] + cindex)
                                elif aa2.any():
                                    candf = int(dtemp.loc[dtemp['dsy'] == dtemp['dsy'].max(), 'ind'].iloc[0])
                                    maxobj = float(htemp.loc[htemp['ind'] == candf, 'obj'].iloc[0])
                                    Hstemp = JointEntropy(dbintr, sindex + [candf])
                                    Hsytemp = JointEntropy(dbintr, sindex + [candf] + cindex)
                                elif aa3.any():
                                    candf = int(dtemp.loc[dtemp['ds'] == dtemp['ds'].max(), 'ind'].iloc[0])
                                    maxobj = float(htemp.loc[htemp['ind'] == candf, 'obj'].iloc[0])
                                    Hstemp = JointEntropy(dbintr, sindex + [candf])
                                    Hsytemp = JointEntropy(dbintr, sindex + [candf] + cindex)
                                else:
                                    candf = candftemp
                                    maxobj = maxobjtemp
                                    Hstemp = Hstemp2
                                    Hsytemp = Hsytemp2

                # record instantaneous rows in intstemp/redstemp
                misy = float(htemp.loc[htemp['ind'] == candf, 'mi'].iloc[0])
                corv = float(htemp.loc[htemp['ind'] == candf, 'cor'].iloc[0])

                intstemp.loc[t - 1, 'ind'] = candf
                intstemp.loc[t - 1, 'hx']  = float(htemp.loc[htemp['ind'] == candf, 'Hx'].iloc[0])
                intstemp.loc[t - 1, 'hxy'] = float(htemp.loc[htemp['ind'] == candf, 'Hxy'].iloc[0])
                intstemp.loc[t - 1, 'mi']  = misy
                intstemp.loc[t - 1, 'cor'] = corv

                redstemp.loc[t - 1, 'ind'] = candf
                redstemp.loc[t - 1, 'hx']  = float(htemp.loc[htemp['ind'] == candf, 'Hx'].iloc[0])
                redstemp.loc[t - 1, 'hxy'] = float(htemp.loc[htemp['ind'] == candf, 'Hxy'].iloc[0])
                redstemp.loc[t - 1, 'mi']  = misy
                redstemp.loc[t - 1, 'cor'] = corv

                # accept: update group sets and remove cand from pools
                sindex.append(candf)
                Hs, Hsy = float(Hstemp), float(Hsytemp)
                Hf  = float(htemp.loc[htemp['ind'] == candf, 'Hx'].iloc[0])
                Hfy = float(htemp.loc[htemp['ind'] == candf, 'Hxy'].iloc[0])

                # drop this feature from all live holders (exactly as script)
                for df_ in (htemp, hhentind, hhred, hhint):
                    idx = df_.index[df_['ind'] == candf]
                    df_.drop(index=idx, inplace=True)
                    df_.reset_index(drop=True, inplace=True)
                rndtemp.remove(candf)

                # update pairwise entropies wrt remaining rndtemp
                if len(rndtemp) > 0:
                    hsfc = Parallel(n_jobs=self.n_jobs)(
                        delayed(JointEntropy)(dbintr, [candf, f, cindex[0]]) for f in rndtemp
                    )
                    hsf  = Parallel(n_jobs=self.n_jobs)(
                        delayed(JointEntropy)(dbintr, [candf, f]) for f in rndtemp
                    )
                    hhentind.loc[:, 'hsfc'] = 0.0
                    hhentind.loc[:, 'hsf']  = 0.0
                    # map to rows aligned with rndtemp order
                    for j, f in enumerate(rndtemp):
                        row = hhentind.index[hhentind['ind'] == f]
                        if len(row):
                            hhentind.loc[row, 'hsfc'] = hsfc[j]
                            hhentind.loc[row, 'hsf']  = hsf[j]

                # fill interaction / redundancy columns for candf
                # I(Xj;Xk|Y) = -H(Y) - H(xj,xk,y) + H(xj,y) + H(k,y)  (divide by Hy)
                if len(rndtemp) > 0:
                    # htemp rows correspond to rndtemp features
                    hhint[candf] = (-Hy - hhentind['hsfc'] + Hfy + htemp['Hxy']) / Hy
                    hhred[candf] = ( Hf + htemp['Hx'] - hhentind['hsf']) / Hy

                    # update live objective columns
                    htemp['utilmi']  = htemp['mi']
                    htemp['objmi']   = htemp['utilmi']
                    htemp['utilcor'] = htemp['cor'].abs()
                    htemp['objcor']  = htemp['utilcor']
                    htemp['utilint'] = hhint[sindex].sum(axis=1)
                    htemp['objint']  = (1.0 / len(sindex)) * htemp['utilint']
                    htemp['utilred'] = hhred[sindex].sum(axis=1)
                    htemp['objred']  = (1.0 / len(sindex)) * htemp['utilred']
                    htemp['obj']     = htemp['objmi'] + htemp['objcor'] + htemp['objint'] + htemp['objred']

                # progress
                if self.verbose:
                    MI = (Hs + Hy - Hsy) / Hy
                    print(f"SPFP: G {g} | step {t} | MIS={_fmt_val(MI)} "
                          f"| Obj={_fmt_val(maxobj)} | Hs={_fmt_val(Hs)} Hsy={_fmt_val(Hsy)} "
                          f"| Ds={_fmt_val(HxA - Hs)} Dsy={_fmt_val(HxyA - Hsy)} | |S|={len(sindex)}")

                t += 1
                nn += 1

            # sanitize
            intstemp.dropna(inplace=True)
            redstemp.dropna(inplace=True)

            # -------------------- optional backward step --------------------
            if self.backward and len(sindex) > 0:
                HsB  = JointEntropy(dbintr, sindex)
                HsyB = JointEntropy(dbintr, sindex + cindex)
                countr = 1
                while (
                    (Hsy >= self.wcrirem * _floor_n(HsyB, 4)) and
                    (Hs  >= self.wcrirem * _floor_n(HsB, 4)) and
                    (len(sindex) > math.ceil(self.perfs * Xs.shape[1]))
                ):
                    # build removal table hhs (like script)
                    hhs = pd.DataFrame(0, index=range(len(sindex)), columns=['ind','mi','hs','hsy','int','red','obj'])
                    hhs['ind'] = sindex
                    # recompute entropies if each feature removed
                    hhs['hs']  = [JointEntropy(dbintr, [f for f in sindex if f != k]) for k in sindex]
                    hhs['hsy'] = [JointEntropy(dbintr, [f for f in sindex if f != k] + cindex) for k in sindex]

                    # pairwise sums for int/red per candidate (costly but faithful)
                    hhintr = pd.DataFrame(0, index=range(len(sindex) - 1), columns=sindex) if len(sindex) > 1 else pd.DataFrame()
                    hhredr = pd.DataFrame(0, index=range(len(sindex) - 1), columns=sindex) if len(sindex) > 1 else pd.DataFrame()
                    for kk in sindex:
                        others = [f for f in sindex if f != kk]
                        if others:
                            hhintr[kk] = [JointEntropy(dbintr, [f, kk, cindex[0]]) for f in others]
                            hhredr[kk] = [JointEntropy(dbintr, [f, kk]) for f in others]

                    aa = hhintr.sum().to_dict() if not hhintr.empty else {k: 0.0 for k in sindex}
                    bb = hhredr.sum().to_dict() if not hhredr.empty else {k: 0.0 for k in sindex}
                    hhs['mi']  = (hhs['hs'] + Hy - hhs['hsy']) / Hy
                    hhs['int'] = hhs['ind'].map(aa) / Hy / max(1, len(sindex))
                    hhs['red'] = -hhs['ind'].map(bb) / Hy / max(1, len(sindex))
                    hhs['obj'] = hhs['mi'] + hhs['int'] + hhs['red']

                    inds  = hhs.loc[hhs['hs']  >= self.wcrirem * _floor_n(HsB,  4), 'ind']
                    indsy = hhs.loc[hhs['hsy'] >= self.wcrirem * _floor_n(HsyB, 4), 'ind']
                    indmi = hhs.loc[hhs['mi']  >= mitA, 'ind']
                    common = indmi[indmi.isin(inds[inds.isin(indsy)])]
                    if common.empty:
                        Hsy, Hs = 0.0, 0.0
                    else:
                        hcommon = hhs.loc[hhs['ind'].isin(common), :]
                        candf = int(hcommon.loc[hcommon['obj'] == hcommon['obj'].min(), 'ind'].iloc[0])
                        sindex.remove(candf)
                        Hs  = float(hhs.loc[hhs['ind'] == candf, 'hs'].iloc[0])
                        Hsy = float(hhs.loc[hhs['ind'] == candf, 'hsy'].iloc[0])

                        intstemp = intstemp[intstemp['ind'] != candf].reset_index(drop=True)
                        redstemp = redstemp[redstemp['ind'] != candf].reset_index(drop=True)

                        if self.verbose:
                            print(f"SPFP: G {g} | backward {countr} | removed {candf} "
                                  f"| Hs={_fmt_val(Hs)} Hsy={_fmt_val(Hsy)} | |S|={len(sindex)}")
                        countr += 1

            # -------------------- end one group --------------------
            # prune from the global stream (exactly the same)
            if len(sindex) > 0 and self.remp > 0:
                k = max(1, round(self.remp * len(sindex)))
                idxs = sorted(random.sample(range(len(sindex)), k))
                remove_feats = [sindex[i] for i in idxs]
            else:
                remove_feats = []

            for f in remove_feats:
                if f in stream:
                    stream.remove(f)

            # drop from H table as well (as in script)
            H = H[~H['ind'].isin(remove_feats)].reset_index(drop=True)

            # record outputs
            groups[g]['findex'] = sindex.copy()
            groups[g]['IntS']   = intstemp.copy()
            groups[g]['RedS']   = redstemp.copy()

            # reporting entropies (train)
            Hxytr = JointEntropy(dbintr, list(range(dbintr.shape[1])))
            Hxtr  = JointEntropy(dbintr, list(range(dbintr.shape[1] - 1)))
            Hytr  = JointEntropy(dbintr, cindex)
            Hsytr = JointEntropy(dbintr, sindex + cindex) if sindex else 0.0
            Hstr  = JointEntropy(dbintr, sindex) if sindex else 0.0

            groups[g]['Hstr']  = Hstr
            groups[g]['Hxtr']  = Hxtr
            groups[g]['Dxtr']  = Hxtr - Hstr
            groups[g]['Hsytr'] = Hsytr
            groups[g]['Hxytr'] = Hxytr
            groups[g]['Dxytr'] = Hxytr - Hsytr
            groups[g]['MIstr'] = Hstr + Hytr - Hsytr
            groups[g]['MIxtr'] = Hxtr + Hytr - Hxytr

            # (optional) test entropies
            if dbints is not None and sindex:
                Hxyts = JointEntropy(dbints, list(range(dbints.shape[1])))
                Hxts  = JointEntropy(dbints, list(range(dbints.shape[1] - 1)))
                Hyts  = JointEntropy(dbints, [Xtr_bin.shape[1]])
                Hsyts = JointEntropy(dbints, sindex + [Xtr_bin.shape[1]])
                Hsts  = JointEntropy(dbints, sindex)
                groups[g]['Hsts']  = Hsts
                groups[g]['Hxts']  = Hxts
                groups[g]['Dxts']  = Hxts - Hsts
                groups[g]['Hsyts'] = Hsyts
                groups[g]['Hxyts'] = Hxyts
                groups[g]['Dxyts'] = Hxyts - Hsyts
                groups[g]['MIsts'] = Hsts + Hyts - Hsyts
                groups[g]['MIxts'] = Hxts + Hyts - Hxyts

            if self.verbose:
                print(f"SPFP: Group {g} done | |S|={len(sindex)} | removed {len(remove_feats)} from stream")

            # update stream reference for the next group
            self.stream_ = stream

        self.groups_ = groups
        return groups

    # -------------------------------------------------------------------------

    def get_partitions(self) -> List[List[int]]:
        """Convenience: return only the list of feature-index sets, one per group."""
        if self.groups_ is None:
            raise RuntimeError("No groups yet. Call partition() first.")
        out = []
        for g in range(self.n_groups):
            if g in self.groups_:
                out.append(list(map(int, self.groups_[g]['findex'])))
        return out

    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SPFPPartitioner(n_groups={self.n_groups}, n_bins={self.n_bins}, "
            f"objective='{self.objective}', override={self.override}, backward={self.backward}, "
            f"perfs={self.perfs}, remp={self.remp}, wcriadd={self.wcriadd}, wcrirem={self.wcrirem}, "
            f"whmag={self.whmag}, standardize={self.standardize}, test_size={self.test_size}, "
            f"random_state={self.random_state}, n_jobs={self.n_jobs}, verbose={self.verbose})"
        )
