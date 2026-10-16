#!/usr/bin/env python
import os
import glob
import time
import argparse
import copy
import re
import numpy as np
import multiprocessing as mp
from pathlib import Path
from pprint import pprint
import json
import yaml
import ROOT
from binnings import *
from trigger_utils import *

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

ROOT.PyConfig.IgnoreCommandLineOptions = True
N_THREADS = 4

def get_input_files(inputs, n_files=None):
    def _process(p):
        if '*' in str(p):
            return sorted(p.parent.glob(p.name))
        if p.is_dir():
            return sorted(p.glob('*.root'))
        if p.is_file() and p.suffix == '.root':
            return [p]
        return []

    files = sorted(set(str(f) for path in ([inputs] if isinstance(inputs, str) else inputs)
                       for f in _process(Path(path))))
    return files[:n_files] if n_files and n_files <= len(files) else files

def load_weights_json(json_name, table_name='FONLLweightRun3'):
    with open(json_name) as f:
        weight_table = json.load(f)

        edges = np.array(weight_table[table_name]['edges'])
        weights = np.array(weight_table[table_name]['weights'])

    return edges, weights

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


class MCTriggerEfficiencyProducer(Module):
    def __init__(self, params):
        self.params = DotDict(params)
        self.writeHistFile = True
        self.trigger_map = self.params.trigger_map
        self.trigger_map['trigger_OR'] = 0
        self.trigger = self.trigger_map[self.params.trigger] if self.params.trigger else None
        self.theory_bins, self.theory_weights = load_weights_json(self.params.theory_weights_file)
        self.max_dr = {k: max_dr_for(k, v[0] if v else '') for k, v in trigger_map.items()}


    def get_fonll_weight(self, b_pt):
        idx = np.searchsorted(self.theory_bins, b_pt, side="right") - 1
        if idx < 0:
            return self.theory_weights[0]
        if idx >= len(self.theory_weights):
            return self.theory_weights[-1]
        return self.theory_weights[idx]


    def get_b_meson(self, gen_parts):
        for gp in gen_parts:
            if gp.pdgId == 443:
                mom_idx = gp.genPartIdxMother
                if mom_idx >= 0:
                    mom = gen_parts[mom_idx]
                    if abs(mom.pdgId) == 521:  # B±
                        return mom
        return None


    def make_th1(self, name, xbins):
        h = ROOT.TH1F(name, name, len(xbins)-1, xbins)
        self.addObject(h)
        return h


    def make_th2(self, name, xbins, ybins):
        h = ROOT.TH2F(name, name, len(xbins)-1, xbins, len(ybins)-1, ybins)
        self.addObject(h)
        return h


    def fill_th1(self, h, arr, w):
        h.Fill(arr, np.ones_like(arr)*w)


    def fill_th2(self, h, arr_x, arr_y, w):
        shape = np.broadcast_shapes(np.shape(arr_x), np.shape(arr_y))
        h.Fill(arr_x, arr_y, (np.ones_like(shape) if shape else 1)*w)


    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)

        # Hist Binnings
        # self.dr_bins     = np.array([0, 0.12, 0.2, 0.28, 0.44, 1.], dtype=np.double)
        self.diel_m_bins = DIEL_M_BINS
        self.pt_bins = PT_BINS
        self.npv_bins = NPV_BINS

        # 1D Hists
        self.h_npv_dict    = {i : self.make_th1(f'npv_{i}', self.npv_bins) for i in self.trigger_map.keys()}

        # Pt Eff
        self.h_diel_m_num_ptbinned_dict      = {i : self.make_th2(f'diel_m_{i}_num_ptbinned', self.diel_m_bins,  self.pt_bins) for i in self.trigger_map.keys()}
        self.h_diel_m_denom_ptbinned_dict    = {i : self.make_th2(f'diel_m_{i}_denom_ptbinned', self.diel_m_bins,  self.pt_bins) for i in self.trigger_map.keys()}


    def analyze(self, event):
        # Define Physics Objects
        electrons = sorted(Collection(event,'Electron'), key=lambda e: e.pt, reverse=True)
        # electrons = Collection(event, 'Electron')
        trig_L1   = Object(event, 'L1')
        trig_HLT  = Object(event, 'HLT')
        pv        = Object(event, 'PV')


        # Find B meson & theory weight
        gen_parts = Collection(event, 'GenPart')

        m0 = self.jpsi_mother(electrons[0], gen_parts); m1 = self.jpsi_mother(electrons[1], gen_parts)
        if m0 < 0 or m0 != m1: return False
        b = self.get_b_meson(gen_parts)          # 15
        if b is None: return False
        b_pt = b.pt
        w_theory = self.get_fonll_weight(b_pt)

        # Define Kinematic Variables
        # lead_el_pt    = electrons[0].pt
        sublead_el_pt = electrons[1].pt
        # sublead_eta   = electrons[1].eta
        # sublead_phi   = electrons[1].phi
        dr            = electrons[0].DeltaR(electrons[1])
        # diel_pt       = (electrons[0].p4() + electrons[1].p4()).Pt()
        diel_m        = (electrons[0].p4() + electrons[1].p4()).M()

        trig_bit_dict = {}
        or_bits = [] 

        # Define Trigger Paths
        if self.trigger:
            bit = getattr(trig_L1, self.trigger[0]) and getattr(trig_HLT, self.trigger[1])
            trig_bit_dict[self.params.trigger] = bit
        else:
            for k, v in self.trigger_map.items():
                if k == 'trigger_OR':
                    continue
                else:
                    # Original Implementation: Require both L1 and HLT
                    bit = getattr(trig_L1, v[0]) and getattr(trig_HLT, v[1]) if v else 0
                    trig_bit_dict[k] = bit
                    or_bits.append(bit)
            
            # trigger_OR remains strictly the OR of your L1+HLT pairs
            trig_bit_dict['trigger_OR'] = any(or_bits)

        # Trigger Efficiencies
        for path in trig_bit_dict.keys():
            if dr > self.max_dr[path]: continue

            self.fill_th2(self.h_diel_m_denom_ptbinned_dict[path], diel_m, sublead_el_pt, w_theory) # Note: use 1 for Data
            
            if trig_bit_dict[path]:
                self.fill_th2(self.h_diel_m_num_ptbinned_dict[path], diel_m, sublead_el_pt, w_theory)

            self.fill_th1(self.h_npv_dict[path], pv.npvs, w_theory)

        return True


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
        job.output_dir = out_path
        job.output_file = out_path / f'effs_{job.name}.root'
        job.theory_weights_file = global_cfg.theory_weights_file
        job.trigger_map = cfg.trigger_map
        # job.hlt_filter if job.hlt_filter else None
        job_configs.append(copy.deepcopy(job))

    if 'mp' in global_cfg.run_strategy:
        start_time = time.perf_counter()
        print("Starting jobs")
        with mp.Pool(processes=N_THREADS) as pool:
            pool.map(worker, [job.to_dict() for job in job_configs])

        finish_time = time.perf_counter()
        print(f'Finished in {finish_time - start_time} seconds')

    else:
        start_time = time.perf_counter()
        print("Starting jobs")
        for job in job_configs:
            worker(job)

        finish_time = time.perf_counter()
        print(f'Finished in {finish_time - start_time} seconds')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='eff_hist_cfg_v2.yml', help='configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='only run test samples')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = DotDict(yaml.safe_load(f))
    cfg.global_cfg.test = args.test

    main(cfg)
