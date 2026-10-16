#!/usr/bin/env python
"""Derive NPV (pileup) weights MC -> data from the producers' npv_<trigger> histograms.

Usage (after a first MC pass with no pu_weights_file; rerun the MC producer afterwards):
  python make_pu_weights.py --data eff_hists/10_14_26/effs_L1_6p0_HLT_4p0.root \
                            --mc   eff_hists/10_14_26/effs_jpsi_resonant_L1_6p0_HLT_4p0.root \
                            --trigger L1_6p0_HLT_4p0 --out pu_weights.json

The npv histograms are filled from the efficiency denominators (same kinematic selection and
L1 DeltaR acceptance in data and MC). Data still contains some combinatorial background; the
weights are a closure correction, not an exact pileup profile.
"""
import argparse
import json
import os

import numpy as np


def compute_pu_weights(data_counts, mc_counts, cap=5.0, min_mc=5.0):
    """data/MC ratio of area-normalised NPV distributions. Bins with too little MC (or no data)
    get weight 1 rather than a noisy ratio; the ratio is capped to limit the effective-stat loss."""
    d = np.asarray(data_counts, dtype=float)
    m = np.asarray(mc_counts, dtype=float)
    if d.sum() <= 0 or m.sum() <= 0:
        raise ValueError('empty NPV histogram')
    d, m = d / d.sum(), m / m.sum()
    w = np.ones_like(d)
    ok = (np.asarray(mc_counts) >= min_mc) & (d > 0)
    w[ok] = d[ok] / m[ok]
    return np.minimum(w, cap)


def main():
    import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--mc', required=True)
    ap.add_argument('--trigger', required=True, help='path name, e.g. L1_6p0_HLT_4p0 (or trigger_OR)')
    ap.add_argument('--out', default='pu_weights.json')
    ap.add_argument('--cap', type=float, default=5.0)
    args = ap.parse_args()

    def counts(fname):
        f = ROOT.TFile.Open(fname)
        h = f.Get(f'hists/npv_{args.trigger}')
        n = h.GetNbinsX()
        edges = [h.GetXaxis().GetBinLowEdge(i) for i in range(1, n + 2)]
        c = [h.GetBinContent(i) for i in range(1, n + 1)]
        f.Close()
        return np.array(edges), np.array(c)

    de, dc = counts(args.data)
    me, mc = counts(args.mc)
    if not np.allclose(de, me):
        raise ValueError('data and MC npv histograms have different binning')
    w = compute_pu_weights(dc, mc, cap=args.cap)

    table = {}
    if os.path.isfile(args.out):
        with open(args.out) as f:
            table = json.load(f)
    table[args.trigger] = {'edges': de.tolist(), 'weights': w.tolist()}
    with open(args.out, 'w') as f:
        json.dump(table, f, indent=1)
    print(f'wrote {args.out}[{args.trigger}]  (min/max weight {w.min():.2f}/{w.max():.2f})')


if __name__ == '__main__':
    main()
