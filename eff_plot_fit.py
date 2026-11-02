#!/usr/bin/env python
"""Trigger scale factors for J/psi -> ee from fitted data and MC.

Data : simultaneous pass/fail fit of the dielectron mass in each sublead-pT bin. The efficiency is
       a fit parameter, the signal shape is shared between pass and fail, and the shape is taken
       from truth-matched MC (only a mass shift and a resolution scale float in data).
MC   : weighted-binomial efficiency of the truth-matched J/psi, counted in the same mass window
       the signal fit covers.
SF   : eff_data / eff_mc with uncertainties propagated in quadrature.
"""
import argparse
import json
import math
import os
import pickle as pkl
import re
from pathlib import Path
import zlib

import numpy as np
import yaml

import ROOT

import eff_math as em
from binning import FIT_RANGE, PT_BINS, DR_BINS, ETA_BINS
from plotting_scripts.EfficiencyPlot import EfficiencyPlot
from trigger_utils import DotDict

MODEL_TYPES = ('two_sided_cb', 'double_cb')

# Used only when there is no MC to take the signal shape from
DEFAULT_SHAPE = {
    'model_type': 'two_sided_cb',
    'params': {'mean': (3.0969, 0.0), 'sigma': (0.04, 0.0), 'alpha': (1.5, 0.0), 'n': (3.0, 0.0),
               'alpha2': (2.0, 0.0), 'n2': (5.0, 0.0)},
}


# ====================================================================== ROOT helpers
def set_verbosity(verb):
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kInfo if verb else ROOT.kError
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.INFO if verb else ROOT.RooFit.ERROR)
    return ROOT.RooFit.PrintLevel(1 if verb else -1)


def iter_args(coll):
    for item in coll:
        yield item


def fit_args(printlevel, extra=()):
    """Common fitTo arguments. IntegrateBins makes the binned fit use the PDF integrated over each
    bin (not its value at the bin centre), consistent with the bin-averaged curve in chiSquare."""
    args = [ROOT.RooFit.Save(True), printlevel] + list(extra)
    if hasattr(ROOT.RooFit, 'IntegrateBins'):
        args.append(ROOT.RooFit.IntegrateBins(1e-3))
    else:
        print('WARNING: this ROOT has no RooFit::IntegrateBins; binned fit uses bin-centre values')
    return args


def crop_hist(h, lo, hi, name):
    """Copy of the bins of `h` fully inside [lo, hi] into a fresh TH1D with exactly those edges.
    Keeps Sumw2 errors. The mass histogram edges (0.01 GeV) must line up with lo/hi."""
    ax = h.GetXaxis()
    b1, b2 = ax.FindBin(lo + 1e-6), ax.FindBin(hi - 1e-6)
    edges = np.array([ax.GetBinLowEdge(i) for i in range(b1, b2 + 2)], dtype=np.double)
    if abs(edges[0] - lo) > 1e-6 or abs(edges[-1] - hi) > 1e-6:
        raise ValueError(f'histogram edges do not line up with fit range [{lo},{hi}]: '
                         f'got [{edges[0]},{edges[-1]}]. Check binning.DIEL_M_BINS.')
    out = ROOT.TH1D(name, name, len(edges) - 1, edges)
    out.SetDirectory(0)
    out.Sumw2()
    for i in range(len(edges) - 1):
        out.SetBinContent(i + 1, h.GetBinContent(b1 + i))
        out.SetBinError(i + 1, h.GetBinError(b1 + i))
    return out


def make_fail_hist(den_c, num_c, name):
    """denominator - numerator, bin by bin (both already cropped). Poisson errors."""
    out = den_c.Clone(name)
    out.SetDirectory(0)
    for i in range(1, out.GetNbinsX() + 1):
        v = max(den_c.GetBinContent(i) - num_c.GetBinContent(i), 0.0)
        out.SetBinContent(i, v)
        out.SetBinError(i, math.sqrt(v))
    return out


def project_x(h2, ibin, name):
    p = h2.ProjectionX(name, ibin, ibin)
    p.SetDirectory(0)
    return p


def estimate_signal(h):
    """Crude J/psi yield: central window minus linear sideband estimate. Only for start values."""
    def s(a, b):
        return sum(h.GetBinContent(i) for i in range(1, h.GetNbinsX() + 1)
                   if a <= h.GetBinCenter(i) < b)
    side = (s(2.6, 2.9) + s(3.3, 3.4)) / 0.4        # events per GeV in the sidebands
    return max(s(3.0, 3.2) - side * 0.2, 0.05 * h.GetSumOfWeights())


def perturb_free(params, frac=0.02, rng=None):
    rng = rng or np.random
    for p in iter_args(params):
        if p.isConstant():
            continue
        rngw = (p.getMax() - p.getMin()) * frac
        p.setVal(min(p.getMax(), max(p.getMin(), p.getVal() + rng.uniform(-rngw, rngw))))

def check_parameter_limits(savename, fit_result):
    if not fit_result:
        return
    hits = []
    for p in iter_args(fit_result.floatParsFinal()):
        if abs(p.getVal() - p.getMin()) < 1e-6:
            hits.append(f'  - {p.GetName()} (at MIN limit: {p.getMin()})')
        if abs(p.getVal() - p.getMax()) < 1e-6:
            hits.append(f'  - {p.GetName()} (at MAX limit: {p.getMax()})')
    if hits:
        print(f'--- Boundary Check for: {savename} ---')
        print('WARNING: the following parameters have final values at their limits:')
        print('\n'.join(hits))
        print(50 * '-')


# ====================================================================== PDFs
def build_signal_template(mass, model_type):
    """Free-shape signal PDF used to extract the shape from MC. Returns (pdf, shape_vars, keepalive)."""
    if model_type == 'two_sided_cb':
        mean = ROOT.RooRealVar('mean', 'mean', 3.0969, 3.0, 3.15)
        sigma = ROOT.RooRealVar('sigma', 'sigma', 0.03, 0.01, 0.1)
        alpha = ROOT.RooRealVar('alpha', 'alpha', 0.5, 0.1, 30.)
        n = ROOT.RooRealVar('n', 'n', 10.0, 1.01, 40.)
        alpha2 = ROOT.RooRealVar('alpha2', 'alpha2', 1.5, 0.1, 5.0)
        n2 = ROOT.RooRealVar('n2', 'n2', 2.0, 1.0, 15.0)
        pdf = ROOT.RooCrystalBall('sig_pdf', 'Signal Fit', mass, mean, sigma, alpha, n, alpha2, n2)
        return pdf, {'mean': mean, 'sigma': sigma, 'alpha': alpha, 'n': n, 'alpha2': alpha2, 'n2': n2}, []

    if model_type == 'double_cb':
        mean = ROOT.RooRealVar('mean', 'mean', 3.0969, 3.0, 3.15)
        sigma1 = ROOT.RooRealVar('sigma1', 'sigma1', 0.02, 0.005, 0.05)
        alpha1 = ROOT.RooRealVar('alpha1', 'alpha1', 1.5, 0.3, 5.0)
        n1 = ROOT.RooRealVar('n1', 'n1', 3.0, 1.0, 15.0)
        alpha1b = ROOT.RooRealVar('alpha1b', 'alpha1b', 1.5, 0.3, 5.0)
        n1b = ROOT.RooRealVar('n1b', 'n1b', 3.0, 1.0, 15.0)
        narrow = ROOT.RooCrystalBall('sig_narrow', 'Narrow core', mass, mean, sigma1, alpha1, n1, alpha1b, n1b)
        sigma2 = ROOT.RooRealVar('sigma2', 'sigma2', 0.08, 0.03, 0.2)
        alpha2 = ROOT.RooRealVar('alpha2', 'alpha2', 1.0, 0.1, 5.0)
        n2 = ROOT.RooRealVar('n2', 'n2', 3.0, 1.0, 15.0)
        broad = ROOT.RooCrystalBall('sig_broad', 'Broad radiative tail', mass, mean, sigma2, alpha2, n2)
        frac = ROOT.RooRealVar('frac_narrow', 'frac_narrow', 0.5, 0.0, 1.0)
        pdf = ROOT.RooAddPdf('sig_pdf', 'Double CB signal', ROOT.RooArgList(narrow, broad), ROOT.RooArgList(frac))
        shape_vars = {'mean': mean, 'sigma1': sigma1, 'alpha1': alpha1, 'n1': n1, 'alpha1b': alpha1b,
                      'n1b': n1b, 'sigma2': sigma2, 'alpha2': alpha2, 'n2': n2, 'frac_narrow': frac}
        return pdf, shape_vars, [narrow, broad]

    raise ValueError(f'Unknown model_type {model_type}')


def build_signal_from_template(mass, shape):
    """Data signal PDF: the MC shape with every tail/fraction parameter constant. Only a common
    mass shift `dm` and a common resolution scale `sig_scale` float. Returns (pdf, nuisances, keepalive)."""
    mt = shape['model_type']
    p = {k: v[0] for k, v in shape['params'].items() if isinstance(v, (tuple, list))}

    def const(name):
        v = ROOT.RooRealVar(f'{name}_mc', f'{name}_mc', p[name])
        v.setConstant(True)
        return v

    dm = ROOT.RooRealVar('dm', 'dm', 0.0, -0.03, 0.03)
    scale = ROOT.RooRealVar('sig_scale', 'sig_scale', 1.0, 0.9, 1.3)
    mean_mc = const('mean')
    mean = ROOT.RooFormulaVar('mean_data', '@0+@1', ROOT.RooArgList(mean_mc, dm))
    keep = [mean_mc, mean]

    def scaled(name):
        s_mc = const(name)
        s = ROOT.RooFormulaVar(f'{name}_data', '@0*@1', ROOT.RooArgList(s_mc, scale))
        keep.extend([s_mc, s])
        return s

    if mt == 'two_sided_cb':
        sigma = scaled('sigma')
        alpha, n, alpha2, n2 = const('alpha'), const('n'), const('alpha2'), const('n2')
        pdf = ROOT.RooCrystalBall('sig_pdf', 'Signal', mass, mean, sigma, alpha, n, alpha2, n2)
        keep.extend([alpha, n, alpha2, n2])
    elif mt == 'double_cb':
        sigma1, sigma2 = scaled('sigma1'), scaled('sigma2')
        a1, nn1, a1b, nn1b = const('alpha1'), const('n1'), const('alpha1b'), const('n1b')
        a2, nn2, frac = const('alpha2'), const('n2'), const('frac_narrow')
        narrow = ROOT.RooCrystalBall('sig_narrow', 'Narrow', mass, mean, sigma1, a1, nn1, a1b, nn1b)
        broad = ROOT.RooCrystalBall('sig_broad', 'Broad', mass, mean, sigma2, a2, nn2)
        pdf = ROOT.RooAddPdf('sig_pdf', 'Signal', ROOT.RooArgList(narrow, broad), ROOT.RooArgList(frac))
        keep.extend([a1, nn1, a1b, nn1b, a2, nn2, frac, narrow, broad])
    else:
        raise ValueError(f'Unknown model_type {mt}')
    return pdf, {'dm': dm, 'sig_scale': scale}, keep


# ====================================================================== plotting of fits
def _pad_pair(c, x0, x1):
    top = ROOT.TPad(f'top_{x0}', 'top', x0, 0.3, x1, 1.0)
    top.SetBottomMargin(0.02)
    top.SetGridx()
    top.Draw()
    c.cd()
    bot = ROOT.TPad(f'bot_{x0}', 'bot', x0, 0.05, x1, 0.3)
    bot.SetTopMargin(0.02)
    bot.SetBottomMargin(0.2)
    bot.SetGridx()
    bot.Draw()
    return top, bot


def _draw_frame_with_pull(top, bot, frame, h_pull, texts):
    top.cd()
    frame.Draw()
    frame.GetXaxis().SetLabelOffset(3.)
    keep = []
    for (x, y, s) in texts:
        t = ROOT.TLatex(x, y, s)
        t.SetTextSize(0.05)
        t.SetNDC(True)
        t.Draw()
        keep.append(t)
    bot.cd()
    fp = frame.GetXaxis()
    frame_pull = ROOT.RooPlot(frame.GetXaxis().GetXmin(), frame.GetXaxis().GetXmax())
    frame_pull.addPlotable(h_pull, 'P')
    frame_pull.SetTitle(' ')
    frame_pull.Draw()
    line = ROOT.TLine(fp.GetXmin(), 0, fp.GetXmax(), 0)
    line.SetLineStyle(7)
    line.Draw()
    ay, ax = frame_pull.GetYaxis(), frame_pull.GetXaxis()
    ax.SetTitle('m(e^{+}e^{-}) [GeV]')
    ax.SetTitleOffset(0.9)
    ay.SetNdivisions(8)
    for a, ref in ((ay, frame.GetYaxis()), (ax, frame.GetXaxis())):
        a.SetTitleSize(2.8 * ref.GetTitleSize())
        a.SetLabelSize(2.8 * ref.GetLabelSize())
    keep.extend([frame_pull, line])
    return keep


# ====================================================================== MC shape template
def fit_with_sumw2_fallback(model, data, args):
    """SumW2Error(True) is the right treatment of weighted MC; fall back to False if the weight^2
    covariance correction is singular (flagged in the return value)."""
    result = model.fitTo(data, *(args + [ROOT.RooFit.SumW2Error(True)]))
    used_fallback = False
    if result is None or result.covQual() < 2:
        result = model.fitTo(data, *(args + [ROOT.RooFit.SumW2Error(False)]))
        used_fallback = True
    return result, used_fallback


def fit_mc_template(h, savename, printlevel, model_type='two_sided_cb', max_tries=10):
    """Free fit of a truth-matched (signal-only) MC histogram. Returns a shape dict or None.
    Keeps the best converged attempt (lowest NLL), not the first one that passes."""
    hc = crop_hist(h, FIT_RANGE[0], FIT_RANGE[1], f'mc_templ_{id(h)}_{model_type}')
    n_ev = hc.GetSumOfWeights()
    if n_ev <= 0:
        print(f'  [{savename.stem}] empty MC histogram, no template')
        return None

    mass = ROOT.RooRealVar('mass', 'mass', FIT_RANGE[0], FIT_RANGE[1])
    mass.setBins(hc.GetNbinsX())
    data = ROOT.RooDataHist('data', 'data', ROOT.RooArgList(mass), hc)

    pdf, shape_vars, keep = build_signal_template(mass, model_type)
    n_sig = ROOT.RooRealVar('n_sig', 'n_sig', n_ev, 0, 1.5 * n_ev)
    model = ROOT.RooAddPdf('mc_model', 'MC signal', ROOT.RooArgList(pdf), ROOT.RooArgList(n_sig))
    params = model.getParameters(data)

    best = None
    for attempt in range(max_tries):
        res, used_fb = fit_with_sumw2_fallback(model, data, fit_args(printlevel))
        ok = res is not None and res.status() == 0 and res.covQual() >= 2
        if ok and (best is None or res.minNll() < best['nll']):
            best = {'nll': res.minNll(), 'res': res, 'fallback': used_fb,
                    'vals': {k: (v.getVal(), v.getError()) for k, v in shape_vars.items()}}
        if ok and attempt >= 2:            # one good fit is accepted after a few restarts
            break
        perturb_free(params)

    if best is None:
        print(f'WARNING: MC template fit ({model_type}) did not converge for {savename.stem}')
        return None
    if best['fallback']:
        print(f'  [{savename.stem}] WARNING: SumW2 covariance correction singular -> '
              f'SumW2Error(False) used; template parameter errors are approximate')

    # restore the best attempt for plotting
    params.assign(best['res'].floatParsFinal())
    check_parameter_limits(savename, best['res'])

    frame = mass.frame(ROOT.RooFit.Title(str(savename.stem)))
    data.plotOn(frame, ROOT.RooFit.Name('h_data'), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
    model.plotOn(frame, ROOT.RooFit.Name('fit_model'), ROOT.RooFit.LineColor(38))
    n_par = best['res'].floatParsFinal().getSize() if hasattr(best['res'].floatParsFinal(), 'getSize') \
        else best['res'].floatParsFinal().size()
    chi2 = frame.chiSquare('fit_model', 'h_data', n_par)
    h_pull = frame.pullHist('h_data', 'fit_model')

    c = ROOT.TCanvas('c_mc', ' ', 800, 600)
    top, bot = _pad_pair(c, 0.0, 1.0)
    keepme = _draw_frame_with_pull(top, bot, frame, h_pull, [(0.15, 0.3, f'#chi^{{2}}/ndf = {chi2:.1f}')])
    c.SaveAs(str(savename))
    c.Close()

    params_out = dict(best['vals'])
    return {'params': params_out, 'model_type': model_type, 'chi2_ndf': chi2, 'n_events': n_ev}


def fit_signal_shape_with_fallback(hist, savename, printlevel, chi2_flag_threshold=100.0):
    """Two-sided CB first; if chi2/ndf flags real mismodelling, also try the double CB and keep
    whichever has the lower chi2/ndf. Returns a shape dict or None."""
    shape = fit_mc_template(hist, savename, printlevel, 'two_sided_cb')
    if shape is None or shape['chi2_ndf'] > chi2_flag_threshold:
        tag = f"{savename.stem} two_sided_cb chi2/ndf=" + ('n/a' if shape is None else f"{shape['chi2_ndf']:.1f}")
        print(f'  [{tag}] -- also trying double_cb')
        alt = fit_mc_template(hist, savename.with_stem(savename.stem + '_doublecb'), printlevel, 'double_cb')
        if alt is not None and (shape is None or alt['chi2_ndf'] < shape['chi2_ndf']):
            shape = alt
    return shape


def fit_mc_shapes(plot_cfg, mc_file, fit_output_file, printlevel):
    """{bin_index: shape} from the truth-matched MC DENOMINATOR (all pairs), summed over the
    configured MC trigger paths with their weights. The same shape is used for pass and fail."""
    n_bins = len(plot_cfg.bins) - 1
    shapes = {}
    mc_paths = plot_cfg.mc_triggers or [{'path': plot_cfg.data_trigger, 'weight': 1.0}]
    for ibin in range(1, n_bins + 1):
        combined = None
        for mc_trig in mc_paths:
            path = str(mc_trig['path']).strip('"')
            den = mc_file.Get(f'hists/diel_m_{path}_denom_{plot_cfg.var}binned')
            proj = project_x(den, ibin, f'mcden_{ibin}_{path}')
            proj.Scale(mc_trig['weight'])
            if combined is None:
                combined = proj.Clone(f'mc_shape_denom_bin{ibin}')
                combined.SetDirectory(0)
            else:
                combined.Add(proj)
        shapes[ibin - 1] = fit_signal_shape_with_fallback(
            combined, fit_output_file.with_stem(f'mc_denom_shape_{plot_cfg.name}_bin{ibin - 1}'), printlevel)
    return shapes


# ====================================================================== data simultaneous fit
def do_simfit(h_pass, h_fail, savename, shape=None, bkg_deg=3, printlevel=None, max_tries=10,
              min_total=50, min_pass=10):
    """Simultaneous fit of pass/fail mass histograms; returns a dict with the efficiency
    (eff, err_hi, err_lo from MINOS) or None if the bin is unusable / the fit never converged."""
    shape = shape or DEFAULT_SHAPE
    lo, hi = FIT_RANGE
    hp = crop_hist(h_pass, lo, hi, f'simfit_pass_{id(h_pass)}')
    hf = crop_hist(h_fail, lo, hi, f'simfit_fail_{id(h_fail)}')
    n_p, n_f = hp.GetSumOfWeights(), hf.GetSumOfWeights()
    n_tot = n_p + n_f
    if n_tot < min_total or n_p < min_pass:
        print(f'  [{savename.stem}] too few events (pass={n_p:.0f}, all={n_tot:.0f}); no fit')
        return None

    mass = ROOT.RooRealVar('mass', 'mass', lo, hi)
    mass.setBins(hp.GetNbinsX())
    cat = ROOT.RooCategory('cat', 'cat')
    cat.defineType('pass')
    cat.defineType('fail')
    dh_p = ROOT.RooDataHist('dh_pass', 'dh_pass', ROOT.RooArgList(mass), hp)
    dh_f = ROOT.RooDataHist('dh_fail', 'dh_fail', ROOT.RooArgList(mass), hf)
    dh = ROOT.RooDataHist('dh', 'dh', ROOT.RooArgList(mass), ROOT.RooFit.Index(cat),
                          ROOT.RooFit.Import('pass', hp), ROOT.RooFit.Import('fail', hf))

    sig, nuis, keep = build_signal_from_template(mass, shape)

    est_all, est_p = estimate_signal(hp) + estimate_signal(hf), estimate_signal(hp)
    n_sig = ROOT.RooRealVar('n_sig', 'n_sig', min(est_all, n_tot), 0, 1.2 * n_tot)
    eff = ROOT.RooRealVar('eff', 'eff', min(max(est_p / max(est_all, 1.0), 0.01), 0.99), 0.0, 1.0)
    ns_p = ROOT.RooFormulaVar('ns_pass', '@0*@1', ROOT.RooArgList(n_sig, eff))
    ns_f = ROOT.RooFormulaVar('ns_fail', '@0*(1-@1)', ROOT.RooArgList(n_sig, eff))

    pdfs, bkgs = {}, {}
    for c_name, ns, n_c in (('pass', ns_p, n_p), ('fail', ns_f, n_f)):
        # pars = [ROOT.RooRealVar(f'cheb_{c_name}{i}', f'cheb_{c_name}{i}', 0.1, -1.0, 1.0) for i in range(bkg_deg)]
        # bkg = ROOT.RooChebychev(f'bkg_{c_name}', f'bkg_{c_name}', mass, ROOT.RooArgList(*pars))
        deg = bkg_deg if n_c > 2000 else min(bkg_deg, 2 if n_c > 500 else 1)
        pars = [ROOT.RooRealVar(f'bern_{c_name}{i}', f'bern_{c_name}{i}', 1.0, 0.0, 20.0)
                for i in range(deg + 1)]
        bkg = ROOT.RooBernstein(f'bkg_{c_name}', f'bkg_{c_name}', mass, ROOT.RooArgList(*pars))
        nb = ROOT.RooRealVar(f'nb_{c_name}', f'nb_{c_name}', max(n_c * 0.5, 1.0), 0, 1.5 * n_c + 10)
        pdfs[c_name] = ROOT.RooAddPdf(f'model_{c_name}', c_name, ROOT.RooArgList(sig, bkg), ROOT.RooArgList(ns, nb))
        bkgs[c_name] = (pars, bkg, nb)
    sim = ROOT.RooSimultaneous('sim', 'sim', cat)
    sim.addPdf(pdfs['pass'], 'pass')
    sim.addPdf(pdfs['fail'], 'fail')

    params = sim.getParameters(dh)
    rng = np.random.RandomState(zlib.crc32(savename.stem.encode()))   # same seed per bin every run
    init = {p.GetName(): p.getVal() for p in iter_args(params)}
    
    best = None
    for attempt in range(max_tries):
        for p in iter_args(params):
            if not p.isConstant():
                p.setVal(init[p.GetName()])
        if attempt > 0:
            perturb_free(params, frac=0.02, rng=rng)
    
        res = sim.fitTo(dh, *fit_args(printlevel))          # no Minos here
        ok = res is not None and res.status() in (0, 1) and res.covQual() >= 1
        if ok:
            cand = {'nll': res.minNll(), 'res': res, 'eff': eff.getVal(),
                    'eff_err': eff.getError(),
                    'n_sig': n_sig.getVal(), 'n_sig_err': n_sig.getError(),
                    'cov_qual': res.covQual(),
                    'dm': nuis['dm'].getVal(), 'sig_scale': nuis['sig_scale'].getVal(),
                    'vals': {p.GetName(): p.getVal() for p in iter_args(res.floatParsFinal())}}
            if best is None or cand['nll'] < best['nll']:
                best = cand
            if attempt >= 2:
                break
        perturb_free(params)

    if best is None:
        print(f'WARNING: simultaneous fit failed to converge for {savename.stem} after {max_tries} attempts')
        return None

    # restore best parameters, then try MINOS as a refinement only
    params.assign(best['res'].floatParsFinal())
    e_hi = e_lo = best['eff_err']
    try:
        minos_set = ROOT.RooArgSet(eff)
        res_m = sim.fitTo(dh, *fit_args(printlevel, [ROOT.RooFit.Minos(ROOT.RooArgSet(eff))]))
        if res_m is not None and eff.hasAsymError():
            e_hi, e_lo = abs(eff.getAsymErrorHi()), abs(eff.getAsymErrorLo())
            best['res'] = res_m
    except Exception as exc:
        print(f'  [{savename.stem}] MINOS failed, using symmetric HESSE error: {exc}')
    # keep the efficiency inside [0,1]
    e_hi = min(e_hi, 1.0 - best['eff'])
    e_lo = min(e_lo, best['eff'])
    best['err_hi'], best['err_lo'] = e_hi, e_lo
    params.assign(best['res'].floatParsFinal())

    check_parameter_limits(savename, best['res'])
    n_par = best['res'].floatParsFinal().getSize() if hasattr(best['res'].floatParsFinal(), 'getSize') \
        else best['res'].floatParsFinal().size()

    chi2 = {}
    frames, pulls = {}, {}
    for c_name, d in (('pass', dh_p), ('fail', dh_f)):
        fr = mass.frame(ROOT.RooFit.Title(f'{savename.stem} {c_name}'))
        d.plotOn(fr, ROOT.RooFit.Name(f'h_{c_name}'))
        pdfs[c_name].plotOn(fr, ROOT.RooFit.Name(f'm_{c_name}'), ROOT.RooFit.LineColor(38))
        chi2[c_name] = fr.chiSquare(f'm_{c_name}', f'h_{c_name}', n_par)
        pulls[c_name] = fr.pullHist(f'h_{c_name}', f'm_{c_name}')
        pdfs[c_name].plotOn(fr, ROOT.RooFit.Components('sig_pdf'), ROOT.RooFit.LineColor(32), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdfs[c_name].plotOn(fr, ROOT.RooFit.Components(f'bkg_{c_name}'), ROOT.RooFit.LineColor(46), ROOT.RooFit.LineStyle(ROOT.kDashed))
        frames[c_name] = fr
    for c_name in ('pass', 'fail'):
        if chi2[c_name] > 3.0:
            print(f'  [{savename.stem}] WARNING: {c_name} chi2/ndf = {chi2[c_name]:.2f}')

    c = ROOT.TCanvas('c_sim', ' ', 1400, 600)
    keepme = []
    for k, c_name in enumerate(('pass', 'fail')):
        c.cd()
        top, bot = _pad_pair(c, 0.5 * k, 0.5 * (k + 1))
        txt = [(0.15, 0.4, f'#chi^{{2}}/ndf = {chi2[c_name]:.1f}')]
        if c_name == 'pass':
            txt.append((0.15, 0.3, f'#epsilon = {best["eff"]:.3f} +{best["err_hi"]:.3f} -{best["err_lo"]:.3f}'))
        keepme.append(_draw_frame_with_pull(top, bot, frames[c_name], pulls[c_name], txt))
    c.SaveAs(str(savename))
    c.Close()

    return {'eff': best['eff'], 'err_hi': best['err_hi'], 'err_lo': best['err_lo'],
            'n_sig': best['n_sig'], 'n_sig_err': best['n_sig_err'], 'cov_qual': best['cov_qual'],
            'chi2_pass': chi2['pass'], 'chi2_fail': chi2['fail'],
            'dm': best['dm'], 'sig_scale': best['sig_scale'], 'model_type': shape['model_type']}


# ====================================================================== configuration / inputs
def assign_hist_format(h_name):
    if 'ptbinned' in h_name:
        return {'bins': PT_BINS, 'xlabel': 'Sublead Electron p_{T} [GeV]'}
    if 'drbinned' in h_name:
        return {'bins': DR_BINS, 'xlabel': r'\Delta R(e_{1},e_{2})'}
    if 'etabinned' in h_name:
        return {'bins': ETA_BINS, 'xlabel': r'Sublead \ Electron \ \eta'}
    raise ValueError(f'No formatting options for hist name {h_name}')


def make_plotlist(cfg):
    plots = []
    data_in = Path(cfg.inputs.data_dir).parent / 'test' if cfg.test else Path(cfg.inputs.data_dir)
    mc_in = Path(cfg.inputs.mc_dir).parent / 'test' if cfg.test else Path(cfg.inputs.mc_dir)
    for plot_name, plot_dict in cfg.plots.items():
        data_trigger = plot_dict.triggers['data_trigger'] if isinstance(plot_dict.triggers, dict) else plot_dict.triggers[0]
        for var in plot_dict.variables:
            plots.append(DotDict({
                'name': '_'.join([plot_name, f'{var}binned']),
                'data_trigger': data_trigger,
                'data_file': data_in / plot_dict.files.data,
                'mc_file': mc_in / plot_dict.files.mc if plot_dict.files.mc else None,
                'output_file': cfg.out_path / Path('_'.join(['eff', data_trigger, f'{var}binned'])).with_suffix('.pdf'),
                'mc_triggers': plot_dict.mc_triggers if plot_dict.mc_triggers else [],
                'var': var,
                'mc_label': plot_dict.mc_label or None,
                **assign_hist_format(f'{var}binned'),
            }))
    return plots


def get_data_hists(plot_cfg):
    """[(num_x, denom_x)] mass projections, one per bin of the binned variable."""
    f = ROOT.TFile.Open(str(plot_cfg.data_file))
    num_h = f.Get(f'hists/diel_m_{plot_cfg.data_trigger}_num_{plot_cfg.var}binned')
    den_h = f.Get(f'hists/diel_m_{plot_cfg.data_trigger}_denom_{plot_cfg.var}binned')
    if not num_h or not den_h:
        raise RuntimeError(f'histograms for {plot_cfg.data_trigger} not found in {plot_cfg.data_file}')
    out = []
    for i in range(1, den_h.GetYaxis().GetNbins() + 1):
        out.append((project_x(num_h, i, f'data_num_{i}'), project_x(den_h, i, f'data_den_{i}')))
    f.Close()
    return out


def compute_mc_effs(plot_cfg, n_bins):
    """Per-bin MC efficiency (weighted binomial in the fit window), mixing the configured MC trigger
    paths with their weights. List of (eff, err) or None."""
    mc_file = ROOT.TFile.Open(str(plot_cfg.mc_file))
    mc_paths = plot_cfg.mc_triggers or [{'path': plot_cfg.data_trigger, 'weight': 1.0}]
    per_bin = []
    for ibin in range(1, n_bins + 1):
        comps = []
        for mc_trig in mc_paths:
            path = str(mc_trig['path']).strip('"')
            num = project_x(mc_file.Get(f'hists/diel_m_{path}_num_{plot_cfg.var}binned'), ibin, f'mcn_{ibin}_{path}')
            den = project_x(mc_file.Get(f'hists/diel_m_{path}_denom_{plot_cfg.var}binned'), ibin, f'mcd_{ibin}_{path}')
            r = em.mc_efficiency(num, den, FIT_RANGE[0], FIT_RANGE[1])
            comps.append(None if r is None else (mc_trig['weight'], r[0], r[1]))
        per_bin.append(em.mix_efficiencies(comps))
    mc_file.Close()
    return per_bin


def make_eff_plot_dict(cfg):
    eff_plot_list = []
    fit_cfg = cfg.fit if cfg.fit else DotDict({})
    for plot_cfg in make_plotlist(cfg):
        print(f'processing hists for {plot_cfg.name}')
        n_bins = len(plot_cfg.bins) - 1
        eff_dict = {
            'name': plot_cfg.name,
            'data_trigger': plot_cfg.data_trigger,
            'output_file': plot_cfg.output_file,
            'data_eff': [],            # per bin: dict from do_simfit, or None
            'mc_eff': [],              # per bin: (eff, err) or None
            'mc_triggers': plot_cfg.mc_triggers,
            'mc_label': plot_cfg.mc_label, 
            'bins': plot_cfg.bins,
            'xlabel': plot_cfg.xlabel,
            'fit_window': FIT_RANGE,
        }
        fit_dir = plot_cfg.output_file.parent / 'fits'
        fit_out = fit_dir / plot_cfg.output_file.name

        shapes = {}
        if plot_cfg.mc_file:
            mc_f = ROOT.TFile.Open(str(plot_cfg.mc_file))
            shapes = fit_mc_shapes(plot_cfg, mc_f, fit_out, cfg.printlevel)
            mc_f.Close()
        else:
            print('  no MC file: using default signal-shape tails')

        for i, (num_h, den_h) in enumerate(get_data_hists(plot_cfg)):
            num_c = crop_hist(num_h, FIT_RANGE[0], FIT_RANGE[1], f'pass_bin{i}')
            den_c = crop_hist(den_h, FIT_RANGE[0], FIT_RANGE[1], f'den_bin{i}')
            fail_c = make_fail_hist(den_c, num_c, f'fail_bin{i}')
            res = do_simfit(num_c, fail_c, fit_out.with_stem(f'data_simfit_{plot_cfg.name}_bin{i}'),
                            shape=shapes.get(i), bkg_deg=int(fit_cfg.bkg_shape_deg or 3),
                            printlevel=cfg.printlevel,
                            min_total=fit_cfg.min_total or 50, min_pass=fit_cfg.min_pass or 10)
            eff_dict['data_eff'].append(res)

        eff_dict['mc_eff'] = compute_mc_effs(plot_cfg, n_bins) if plot_cfg.mc_file else [None] * n_bins
        eff_plot_list.append(eff_dict)

    with open(cfg.out_path / 'eff_plot_entries.pkl', 'wb') as f:
        pkl.dump(eff_plot_list, f)
    return eff_plot_list


# ====================================================================== plots / json
def _graph(bins, points):
    """points: list of (eff, err_hi, err_lo) or None per bin -> TGraphAsymmErrors (skipping None)."""
    g = ROOT.TGraphAsymmErrors()
    j = 0
    for i, p in enumerate(points):
        if p is None:
            continue
        x0, x1 = bins[i], bins[i + 1]
        g.SetPoint(j, 0.5 * (x0 + x1), p[0])
        g.SetPointError(j, 0.5 * (x1 - x0), 0.5 * (x1 - x0), p[2], p[1])
        j += 1
    return g, j


def collect_points(d):
    data_pts = [None if r is None else (r['eff'], r['err_hi'], r['err_lo']) for r in d['data_eff']]
    mc_pts = [None if r is None else (r[0], r[1], r[1]) for r in d['mc_eff']]
    sf_pts = []
    for dp, mp in zip(data_pts, mc_pts):
        sf = em.scale_factor(*(dp if dp else (None, 0, 0)), *((mp[0], mp[1]) if mp else (None, 0)))
        sf_pts.append(sf)
    return data_pts, mc_pts, sf_pts


def plot_efficiencies(eff_dicts, test=False):
    for d in eff_dicts:
        if test and ('test' not in d['name']):
            continue
        bins = np.asarray(d['bins'], dtype=np.double)
        data_pts, mc_pts, sf_pts = collect_points(d)
        g_data, nd = _graph(bins, data_pts)
        g_mc, nm = _graph(bins, mc_pts)
        g_sf, ns = _graph(bins, sf_pts)
        if nd == 0 or nm == 0:
            print(f'  {d["name"]}: nothing to plot (data points: {nd}, MC points: {nm})')
            continue

        eff_plot = EfficiencyPlot(init_params={
            'title_string': ' ;' + d['xlabel'] + ';Efficiency',
            'xrange': (bins[0], bins[-1]),
            'yrange': (0., 1.1),
            'rrange': (0.5, 2.0),
            'leg_scale': .65,
            'leg_header': '',
        })
        eff_plot.plotEfficiencyGraphs(
            g_data, g_mc, g_sf, bins,
            h1_title='J/#psi #rightarrow ee, 2022 data',
            h2_title=d.get('mc_label') or 'B^{+} #rightarrow J/#psi K^{+} MC',
            save=str(d['output_file']),
        )


def make_sf_json(cfg, eff_dicts):
    outputs = {}
    for d in eff_dicts:
        matches = list(re.finditer(r'_([^_]+)binned(?=_|$)', d['name']))
        if not matches:
            raise ValueError(f"Could not find binvar in {d['name']}")
        data_pts, mc_pts, sf_pts = collect_points(d)

        def pair(p, idx_hi=1, idx_lo=2):
            return None if p is None else (round(p[0], 3), round(em.sym_err(p[idx_hi], p[idx_lo]), 3))

        outputs[d['name']] = {
            'binvar': matches[-1].group(1),
            'bins': [float(b) for b in d['bins'][:-1]],
            'bin_edges': [float(b) for b in d['bins']],
            'sfs': [pair(p) for p in sf_pts],
            'data_effs': [pair(p) for p in data_pts],
            'mc_effs': [pair(p) for p in mc_pts],
            'sfs_err_up': [None if p is None else round(p[1], 4) for p in sf_pts],
            'sfs_err_down': [None if p is None else round(p[2], 4) for p in sf_pts],
            'data_effs_err_up': [None if p is None else round(p[1], 4) for p in data_pts],
            'data_effs_err_down': [None if p is None else round(p[2], 4) for p in data_pts],
            'fit_window': [float(x) for x in d.get('fit_window', FIT_RANGE)],
            'data_fit_cov_qual': [None if r is None else r['cov_qual'] for r in d['data_eff']],
        }
        print(f"{d['name']}:")
        for i, p in enumerate(sf_pts):
            lo, hi = d['bins'][i], d['bins'][i + 1]
            s = 'n/a' if p is None else f'{p[0]:.3f} +{p[1]:.3f} -{p[2]:.3f}'
            print(f'   [{lo:g},{hi:g}) SF = {s}')

    with open(cfg.out_path / 'sf_jsons' / 'trigger_sfs.json', 'w') as outfile:
        json.dump(outputs, outfile, indent=4)


def main(cfg):
    cfg = DotDict(cfg)
    if cfg.file is None:
        eff_dicts = make_eff_plot_dict(cfg)
    elif Path(cfg.file).is_file():
        with open(cfg.file, 'rb') as f:
            eff_dicts = pkl.load(f)
        if any('data_eff' not in d for d in eff_dicts):
            raise ValueError(f'{cfg.file} was written by the old fit script (data_num_yields); rerun without -f')
    else:
        raise ValueError(f'Efficiency dict file {str(cfg.file)} is not readable')

    plot_efficiencies(eff_dicts, test=cfg.test)
    make_sf_json(cfg, eff_dicts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='eff_plot_cfg_v2.yml', help='plot configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='only run test samples')
    parser.add_argument('-f', '--file', dest='file', default=None, help='make plots from pkl file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = DotDict(yaml.safe_load(f))

    cfg.test = args.test
    cfg.verbose = args.verbose
    cfg.printlevel = set_verbosity(args.verbose)
    cfg.file = args.file
    # One output dir for everything (plots, fits, pickle, json), test-aware
    cfg.out_path = Path('./eff_plots/test') if cfg.test else Path(cfg.output.output_dir)
    for sub in ('', 'fits', 'sf_jsons'):
        os.makedirs(cfg.out_path / sub, exist_ok=True)

    main(cfg)
