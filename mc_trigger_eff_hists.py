#!/usr/bin/env python
"""MC (B+ -> J/psi K+, J/psi -> ee) trigger-efficiency histograms.

Differences from the data producer (all in event_weight):
  * truth matching: both reco electrons must come from the same J/psi, itself from a B+  (fix 3, 15)
  * FONLL weight as a function of the B+ pT
  * pileup weight (NPV data/MC) if a pu_weights_file is configured                       (fix 4)
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import yaml

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

from trigger_utils import DotDict, get_input_files
from producer_base import TriggerEfficiencyProducerBase

N_THREADS = 4


def load_weights_json(json_name, table_name='FONLLweightRun3'):
    with open(json_name) as f:
        weight_table = json.load(f)
    return np.array(weight_table[table_name]['edges']), np.array(weight_table[table_name]['weights'])


def load_pu_table(json_name, key):
    """{'edges': [...], 'weights': [...]} for this trigger from make_pu_weights.py output, or None."""
    if not json_name or not os.path.isfile(json_name):
        if json_name:
            print(f'WARNING: pu_weights_file {json_name} not found -> PU weight = 1')
        return None
    with open(json_name) as f:
        table = json.load(f)
    if key not in table:
        print(f'WARNING: no PU table for "{key}" in {json_name} -> PU weight = 1')
        return None
    return np.array(table[key]['edges']), np.array(table[key]['weights'])


def lookup(edges, weights, x):
    idx = int(np.searchsorted(edges, x, side='right')) - 1
    return float(weights[min(max(idx, 0), len(weights) - 1)])


# ---------------------------------------------------------------- GenPart navigation
def _mother_skipping(gen, i, pdg):
    """Mother of particle i, skipping ancestors that are copies of the same species (FSR/Pythia chains)."""
    m = gen[i].genPartIdxMother
    while m >= 0 and abs(gen[m].pdgId) == pdg:
        m = gen[m].genPartIdxMother
    return m


def _top_copy(gen, i, pdg):
    """Climb to the first copy of particle species `pdg` in the chain (production-level copy)."""
    while True:
        m = gen[i].genPartIdxMother
        if m >= 0 and abs(gen[m].pdgId) == pdg:
            i = m
        else:
            return i


def truth_jpsi(gen, el):
    """Index of the top J/psi copy this reco electron descends from, or -1."""
    i = el.genPartIdx
    if i < 0 or abs(gen[i].pdgId) != 11:
        return -1
    m = _mother_skipping(gen, i, 11)
    if m < 0 or gen[m].pdgId != 443:
        return -1
    return _top_copy(gen, m, 443)


def b_plus_of(gen, jpsi_idx):
    """B+- (top copy) that produced this J/psi, or None."""
    m = gen[jpsi_idx].genPartIdxMother
    if m < 0 or abs(gen[m].pdgId) != 521:
        return None
    return gen[_top_copy(gen, m, 521)]


class MCTriggerEfficiencyProducer(TriggerEfficiencyProducerBase):
    def __init__(self, params):
        super().__init__(params)
        self.theory_bins, self.theory_weights = load_weights_json(self.params.theory_weights_file)
        self.pu_table = load_pu_table(self.params.pu_weights_file, self.params.trigger or 'trigger_OR')

    def get_fonll_weight(self, b_pt):
        idx = np.searchsorted(self.theory_bins, b_pt, side='right') - 1
        if idx < 0:
            return self.theory_weights[0]
        if idx >= len(self.theory_weights):
            return self.theory_weights[-1]
        return self.theory_weights[idx]

    def get_pu_weight(self, npv):
        return lookup(*self.pu_table, npv) if self.pu_table else 1.0

    def event_weight(self, event, electrons):
        gen = Collection(event, 'GenPart')
        j0 = truth_jpsi(gen, electrons[0])
        j1 = truth_jpsi(gen, electrons[1])
        if j0 < 0 or j0 != j1:                 # not the two J/psi daughters -> not signal
            return None
        b = b_plus_of(gen, j0)
        if b is None:                          # J/psi not from a B+ (fix 15: no .pt on None)
            return None
        npv = event.PV_npvs
        return float(self.get_fonll_weight(b.pt)) * self.get_pu_weight(npv)


def worker(params):
    p = PostProcessor(
        params['output_dir'],
        params['input_files'],
        cut=params['presel'],
        branchsel=None,
        modules=[MCTriggerEfficiencyProducer(params)],
        noOut=True,
        histDirName='hists',
        histFileName=str(params['output_file']),
    )
    p.run()


def main(cfg):
    global_cfg = DotDict(cfg.global_cfg)

    out_path = Path(global_cfg.output_dir)
    if global_cfg.test:
        out_path = out_path.parent / 'test'
    os.makedirs(out_path, exist_ok=True)

    job_configs = []
    for job in cfg.mc_samples:
        job = DotDict(job)
        if (global_cfg.test and ('test' not in job.name)) or (not global_cfg.test and ('test' in job.name)):
            continue
        job.input_files = get_input_files(job.inputs)
        print(f'{job.name}: found {len(job.input_files)} files')
        if not job.input_files:
            print(f'  WARNING: no files found for input pattern: {job.inputs}')
            continue
        job.output_dir = out_path
        job.output_file = out_path / f'effs_{job.name}.root'
        job.theory_weights_file = global_cfg.theory_weights_file
        job.pu_weights_file = global_cfg.pu_weights_file if global_cfg.pu_weights_file else None
        job.trigger_map = cfg.trigger_map
        job.hlt_filter_bits = cfg.hlt_filter_bits if cfg.hlt_filter_bits else {}
        job.trigobj_match = bool(global_cfg.trigobj_match)
        job.trigobj_dr = global_cfg.trigobj_dr if global_cfg.trigobj_dr else 0.1
        job_configs.append(copy.deepcopy(job))

    print('Starting jobs')
    start_time = time.perf_counter()
    if 'mp' in global_cfg.run_strategy:
        with mp.Pool(processes=N_THREADS) as pool:
            pool.map(worker, [job.to_dict() for job in job_configs])
    else:
        for job in job_configs:
            worker(job)
    print(f'Finished in {time.perf_counter() - start_time} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='eff_hist_cfg_v2.yml', help='configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='only run test samples')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = DotDict(yaml.safe_load(f))
    cfg.global_cfg.test = args.test

    main(cfg)
