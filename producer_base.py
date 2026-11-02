"""Shared trigger-efficiency histogram producer (NanoAODTools Module).

The data and MC producers differ only in `event_weight()` (data: 1; MC: truth match + FONLL + PU),
so everything else lives here and cannot drift between the two.
"""
import numpy as np
import math
import ROOT

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object

from binning import (DIEL_M_BINS, PT_BINS, NPV_BINS, LEADPT_BINS, DR_FINE_BINS, DXY_FINE_BINS, DZ_FINE_BINS, JPSIPT_BINS)
from trigger_utils import DotDict, build_max_dr, delta_r, match_trigger_objects, both_matched
from jpsi_vtx import jpsi_dxy_lxy

ROOT.PyConfig.IgnoreCommandLineOptions = True


class TriggerEfficiencyProducerBase(Module):
    def __init__(self, params):
        Module.__init__(self)
        self.params = DotDict(params)
        self.writeHistFile = True

        tm = dict(self.params.trigger_map)
        tm['trigger_OR'] = 0                      # pseudo-path: OR of all real paths
        self.trigger_map = tm

        # One named path -> only that path. No name (or 'trigger_OR') -> every path + the OR.
        name = self.params.trigger
        self.single_path = name if (name and tm.get(name)) else None
        self.paths = [self.single_path] if self.single_path else list(tm.keys())

        # Fix 2: the denominator of each path is restricted to the L1 DeltaR acceptance of that path
        self.max_dr = build_max_dr(tm)

        # Fix 1: optional HLT-object matching of BOTH offline electrons (needs TrigObj + filter bits)
        self.trigobj_match = bool(self.params.trigobj_match)
        self.hlt_filter_bits = dict(self.params.hlt_filter_bits) if self.params.hlt_filter_bits else {}
        self.trigobj_dr = float(self.params.trigobj_dr or 0.1)
        if self.trigobj_match:
            for p in self.paths:
                if p == 'trigger_OR':
                    continue
                hlt_name = tm[p][1]
                if self.hlt_filter_bits.get(hlt_name) is None:
                    raise ValueError(
                        f'trigobj_match is on but hlt_filter_bits has no entry for {hlt_name}. '
                        f'Read the bit from the TrigObj_filterBits branch description and add it to the yml.')

    # ------------------------------------------------------------------ hooks
    def event_weight(self, event, electrons):
        """Return the event weight, or None to drop the event from all histograms."""
        raise NotImplementedError

    # ------------------------------------------------------------------ hist helpers
    def _make(self, cls, name, *edges):
        args = [name, name]
        for e in edges:
            args += [len(e) - 1, e]
        h = cls(*args)
        h.Sumw2()
        self.addObject(h)
        return h

    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)
        self.h_npv, self.h_num, self.h_denom, self.h_leadpt_dr, self.h_m_leadpt, self.h_m_dr, self.h_m_dr_num = {}, {}, {}, {}, {}, {}, {}
        self.h_m_dxy, self.h_m_dxy_num, self.h_m_dz, self.h_m_dz_num, self.h_m_jpsipt, self.h_m_jpsipt_num = {}, {}, {}, {}, {}, {}
        self.h_m_jdxy, self.h_m_jdxy_num = {}, {}
        for p in self.paths:
            self.h_npv[p] = self._make(ROOT.TH1F, f'npv_{p}', NPV_BINS)
            self.h_num[p] = self._make(ROOT.TH2F, f'diel_m_{p}_num_ptbinned', DIEL_M_BINS, PT_BINS)
            self.h_denom[p] = self._make(ROOT.TH2F, f'diel_m_{p}_denom_ptbinned', DIEL_M_BINS, PT_BINS)
            self.h_leadpt_dr[p] = self._make(ROOT.TH2F, f'leadpt_dr_{p}_denom', LEADPT_BINS, DR_FINE_BINS)
            self.h_m_leadpt[p] = self._make(ROOT.TH2F, f'diel_m_{p}_kin_leadpt', DIEL_M_BINS, LEADPT_BINS)
            self.h_m_dr[p]     = self._make(ROOT.TH2F, f'diel_m_{p}_kin_dr', DIEL_M_BINS, DR_FINE_BINS)
            self.h_m_dr_num[p] = self._make(ROOT.TH2F, f'diel_m_{p}_kin_dr_num', DIEL_M_BINS, DR_FINE_BINS)
            self.h_m_dxy[p]     = self._make(ROOT.TH2F, f'diel_m_{p}_kin_dxy',     DIEL_M_BINS, DXY_FINE_BINS)
            self.h_m_dxy_num[p] = self._make(ROOT.TH2F, f'diel_m_{p}_kin_dxy_num', DIEL_M_BINS, DXY_FINE_BINS)
            self.h_m_dz[p]      = self._make(ROOT.TH2F, f'diel_m_{p}_kin_dz',      DIEL_M_BINS, DZ_FINE_BINS)
            self.h_m_dz_num[p]  = self._make(ROOT.TH2F, f'diel_m_{p}_kin_dz_num',  DIEL_M_BINS, DZ_FINE_BINS)
            self.h_m_jpsipt[p]     = self._make(ROOT.TH2F, f'diel_m_{p}_jpsipt',     DIEL_M_BINS, JPSIPT_BINS)
            self.h_m_jpsipt_num[p] = self._make(ROOT.TH2F, f'diel_m_{p}_jpsipt_num', DIEL_M_BINS, JPSIPT_BINS)
            self.h_m_jdxy[p]     = self._make(ROOT.TH2F, f'diel_m_{p}_kin_jpsidxy',     DIEL_M_BINS, DXY_FINE_BINS)
            self.h_m_jdxy_num[p] = self._make(ROOT.TH2F, f'diel_m_{p}_kin_jpsidxy_num', DIEL_M_BINS, DXY_FINE_BINS)


    # ------------------------------------------------------------------ trigger decision
    def _path_bits(self, event, e0, e1):
        trig_L1 = Object(event, 'L1')
        trig_HLT = Object(event, 'HLT')
        trig_objs = Collection(event, 'TrigObj') if self.trigobj_match else None

        bits = {}
        for p in self.paths:
            if p == 'trigger_OR':
                continue
            l1_name, hlt_name = self.trigger_map[p]
            bit = bool(getattr(trig_L1, l1_name)) and bool(getattr(trig_HLT, hlt_name))
            if bit and self.trigobj_match:
                fb = self.hlt_filter_bits[hlt_name]
                m0 = match_trigger_objects(e0.eta, e0.phi, trig_objs, fb, self.trigobj_dr)
                m1 = match_trigger_objects(e1.eta, e1.phi, trig_objs, fb, self.trigobj_dr)
                bit = both_matched(m0, m1)
            bits[p] = bit
        if 'trigger_OR' in self.paths:
            bits['trigger_OR'] = any(bits.values())
        return bits

    # ------------------------------------------------------------------ event loop
    def analyze(self, event):
        # Fix 3b: never rely on the stored ordering
        electrons = sorted(Collection(event, 'Electron'), key=lambda e: e.pt, reverse=True)
        if len(electrons) < 2:
            return False
        e0, e1 = electrons[0], electrons[1]

        w = self.event_weight(event, electrons)
        if w is None:
            return False

        pv = Object(event, 'PV')
        sublead_pt = e1.pt
        dr = delta_r(e0.eta, e0.phi, e1.eta, e1.phi)
        diel_m = (e0.p4() + e1.p4()).M()
        dxy_max = max(abs(e0.dxy), abs(e1.dxy))
        dz_max  = max(abs(e0.dz),  abs(e1.dz))
        jpsi_pt = math.hypot(e0.pt*math.cos(e0.phi) + e1.pt*math.cos(e1.phi),
                     e0.pt*math.sin(e0.phi) + e1.pt*math.sin(e1.phi))
        v = jpsi_dxy_lxy(e0.pt, e0.phi, e0.dxy, e1.pt, e1.phi, e1.dxy)

        bits = self._path_bits(event, e0, e1)

        for p in self.paths:
            if dr > self.max_dr[p]:           # outside this path's L1 geometric acceptance
                continue
            self.h_denom[p].Fill(float(diel_m), float(sublead_pt), float(w))
            if bits[p]:
                self.h_num[p].Fill(float(diel_m), float(sublead_pt), float(w))
                self.h_m_dr_num[p].Fill(float(diel_m), float(dr), float(w))
                self.h_m_dxy_num[p].Fill(float(diel_m), float(dxy_max), float(w))
                self.h_m_dz_num[p].Fill(float(diel_m), float(dz_max), float(w))
                self.h_m_jpsipt_num[p].Fill(float(diel_m), jpsi_pt, float(w))
            self.h_npv[p].Fill(float(pv.npvs), float(w))
            self.h_leadpt_dr[p].Fill(float(e0.pt), float(dr), float(w))
            self.h_m_leadpt[p].Fill(float(diel_m), float(e0.pt), float(w))
            self.h_m_dr[p].Fill(float(diel_m), float(dr), float(w))
            self.h_m_dxy[p].Fill(float(diel_m), float(dxy_max), float(w))
            self.h_m_dz[p].Fill(float(diel_m), float(dz_max), float(w))
            self.h_m_jpsipt[p].Fill(float(diel_m), jpsi_pt, float(w))
            if v is not None:
                self.h_m_jdxy[p].Fill(float(diel_m), abs(v[0]), float(w))
                if bits[p]:
                    self.h_m_jdxy_num[p].Fill(float(diel_m), abs(v[0]), float(w))
        return True
