import os
import json
import math
import re
import yaml
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from uncertainties import ufloat
import argparse
import copy

import ROOT

from plotting_scripts.EfficiencyPlot import EfficiencyPlot


def check_parameter_limits(savename, fit_result):
    if not fit_result:
        return 

    params_at_limit = []
    final_params = fit_result.floatParsFinal()

    for i in range(final_params.getSize()):
        param = final_params.at(i)
        name = param.GetName()
        value = param.getVal()
        min_val = param.getMin()
        max_val = param.getMax()

        # Use a small tolerance for floating point comparison
        tolerance = 1e-6

        if abs(value - min_val) < tolerance:
            params_at_limit.append(f"  - {name} (at MIN limit: {min_val})")
        
        if abs(value - max_val) < tolerance:
            params_at_limit.append(f"  - {name} (at MAX limit: {max_val})")

    if params_at_limit:
        print(f"--- Boundary Check for: {savename} ---")
        print("WARNING: The following parameters have final values at their limits:")
        for param_info in params_at_limit:
            print(param_info)
        print(50*'-')


def set_verbosity(verb):
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kInfo if verb else ROOT.kError
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.INFO if verb else ROOT.RooFit.ERROR)
    printlevel = ROOT.RooFit.PrintLevel(1 if verb else -1)

    return printlevel


def get_event_count(h, xrange=None):
    nEvents = 0.0
    err = 0.0

    if xrange is not None:
        bin_low = h.GetXaxis().FindBin(xrange[0])
        bin_high = h.GetXaxis().FindBin(xrange[1])
         
        error_ref = ROOT.Double(0.0)
        
        nEvents = h.IntegralAndError(bin_low, bin_high, error_ref)
        err = float(error_ref)
 
    else:
        nEvents = h.GetSumOfWeights() 
        err = np.sqrt(nEvents)
    return nEvents, err


def estimateBins(h,nbins=5):
    evts = [h.GetBinContent(i) for i in range(h.GetXaxis().GetNbins())]
    tot = np.sum(evts)

    frac = 1
    n_evts = 0
    bins = f'{h.GetBinLowEdge(1)}'
    for i in range(1, h.GetXaxis().GetNbins()):
        n_evts += h.GetBinContent(i)
        if n_evts > frac * tot / nbins:
            bins += ', '+str(round(h.GetBinLowEdge(i),2))
            frac += 1
    bins += ', '+str(round(h.GetBinLowEdge(h.GetXaxis().GetNbins()+1),2))
    return bins

def assign_hist_format(h_name):
    if 'ptbinned' in h_name:
        h_format = {
            # 'bins' : np.array([5, 7, 8, 9, 10, 12, 20], dtype=np.double),
            'bins' : np.array([4, 6, 12, 20], dtype=np.double),
            'xlabel' : 'Sublead Electron p_{T} [GeV]',
        }
    elif 'drbinned' in h_name:
        h_format = {
            'bins' : np.array([0, 0.12, 0.2, 0.28, 0.44, 1.], dtype=np.double),
            'xlabel' : r'\Delta R(e_{1},e_{2})',
        }
    elif 'etabinned' in h_name:
        h_format = {
            'bins' : np.array([-1.22, -0.7, -.2, 0.2, .7, 1.22], dtype=np.double),
            'xlabel' : r'Sublead \ Electron \ \eta',
        }
    else:
        raise ValueError(f'No formatting options for hist name {h_name}')

    return h_format

def check_hist_sanity(h, label=""):
    """Quick sanity check on the input histogram before fitting."""
    nbins = h.GetNbinsX()
    neg_bins = 0
    zero_bins = 0
    total_w = 0.0
    total_w2 = 0.0
    for i in range(1, nbins + 1):
        c = h.GetBinContent(i)
        e = h.GetBinError(i)
        total_w += c
        total_w2 += e**2
        if c < 0:
            neg_bins += 1
        if c == 0:
            zero_bins += 1
    eff_stats = (total_w**2 / total_w2) if total_w2 > 0 else 0
    print(f"  [hist check {label}] nbins={nbins}, sum_w={total_w:.2f}, "
          f"eff_stats(Neff)={eff_stats:.1f}, neg_bins={neg_bins}, zero_bins={zero_bins}")
    if neg_bins > 0:
        print(f"  [hist check {label}] WARNING: negative bin content present — "
              f"likely from weighted MC subtraction or Scale(); check upstream.")
    if eff_stats < 50:
        print(f"  [hist check {label}] WARNING: very low effective statistics "
              f"(Neff={eff_stats:.1f}) — binned fit may be poorly constrained "
              f"regardless of shape/background choices.")


def print_fit_state(tag, fit_result, sig_coeff, bkg_coeff, bern_pars, shape_vars):
    status = fit_result.status() if fit_result else None
    covq = fit_result.covQual() if fit_result else None
    edm = fit_result.edm() if fit_result else None
    minnll = fit_result.minNll() if fit_result else None
    print(f"  [{tag}] status={status} covQual={covq} edm={edm} minNll={minnll}")
    print(f"  [{tag}] sig_coeff={sig_coeff.getVal():.2f}+-{sig_coeff.getError():.2f}  "
          f"bkg_coeff={bkg_coeff.getVal():.2f}+-{bkg_coeff.getError():.2f}")
    for name, var in shape_vars.items():
        at_lo = abs(var.getVal() - var.getMin()) < 1e-6
        at_hi = abs(var.getVal() - var.getMax()) < 1e-6
        flag = " <-- AT LIMIT" if (at_lo or at_hi) else ""
        print(f"  [{tag}]   {name}: {var.getVal():.4f} +- {var.getError():.4f}"
              f"  [{var.getMin()},{var.getMax()}]{flag}")
    for bp in bern_pars:
        at_lo = abs(bp.getVal() - bp.getMin()) < 1e-6
        at_hi = abs(bp.getVal() - bp.getMax()) < 1e-6
        flag = " <-- AT LIMIT" if (at_lo or at_hi) else ""
        print(f"  [{tag}]   {bp.GetName()}: {bp.getVal():.4f} +- {bp.getError():.4f}{flag}")

def fit_with_sumw2_fallback(fit_model, data, base_args):
    """
    Try SumW2Error(True) first (correct treatment for weighted MC).
    If the weight^2 covariance correction is singular, fall back to
    SumW2Error(False) and flag the result so its errors can be treated
    as approximate.
    """
    ROOT.RooMsgService.instance().saveState()
    err_stream = ROOT.RooMsgService.instance().addStream(
        ROOT.RooFit.ERROR, ROOT.RooFit.Topic(ROOT.RooFit.Fitting)
    )
    # crude capture: just try it, then check status/covQual as a proxy,
    # since RooFit doesn't raise a catchable exception for this — it's a
    # printed ERROR while the fit result itself may still report status 0.

    fit_args_sw2 = base_args + [ROOT.RooFit.SumW2Error(True)]
    result = fit_model.fitTo(data, *fit_args_sw2)
    ROOT.RooMsgService.instance().restoreState()

    used_fallback = False
    if result is None or result.covQual() < 2:
        # retry without the weight^2 correction
        fit_args_no_sw2 = base_args + [ROOT.RooFit.SumW2Error(False)]
        result = fit_model.fitTo(data, *fit_args_no_sw2)
        used_fallback = True

    return result, used_fallback

def build_signal_pdf(mass, model_type='single_cb'):
    """Returns (pdf, shape_vars_dict, param_list_for_output)."""
    if model_type == 'single_cb':
        cb_mean   = ROOT.RooRealVar('cb_mean', 'cb_mean', 3.0969, 3.0, 3.15)
        cb_sigma  = ROOT.RooRealVar('cb_sigma', 'cb_sigma', 0.03, 0.01, 0.1)
        cb_alpha  = ROOT.RooRealVar('cb_alpha', 'cb_alpha', 0.5, 0.1, 30.)
        cb_n      = ROOT.RooRealVar('cb_n', 'cb_n', 10.0, 1.01, 40.)
        cb_alpha2 = ROOT.RooRealVar('cb_alpha2', 'cb_alpha2', 1.5, 0.1, 5.0)
        cb_n2     = ROOT.RooRealVar('cb_n2', 'cb_n2', 2.0, 1.0, 15.0)
        sig_pdf = ROOT.RooCrystalBall('sig_pdf', 'Signal Fit', mass,
                                       cb_mean, cb_sigma, cb_alpha, cb_n, cb_alpha2, cb_n2)
        shape_vars = {'mean': cb_mean, 'sigma': cb_sigma, 'alpha': cb_alpha,
                       'n': cb_n, 'alpha2': cb_alpha2, 'n2': cb_n2}
        keepalive = []

    elif model_type == 'double_cb':
        cb_mean = ROOT.RooRealVar('cb_mean', 'cb_mean', 3.0969, 3.0, 3.15)

        sigma1 = ROOT.RooRealVar('sigma1', 'sigma1', 0.02, 0.005, 0.05)
        alpha1 = ROOT.RooRealVar('alpha1', 'alpha1', 1.5, 0.3, 5.0)
        n1     = ROOT.RooRealVar('n1', 'n1', 3.0, 1.0, 15.0)
        alpha1b = ROOT.RooRealVar('alpha1b', 'alpha1b', 1.5, 0.3, 5.0)
        n1b     = ROOT.RooRealVar('n1b', 'n1b', 3.0, 1.0, 15.0)
        sig_narrow = ROOT.RooCrystalBall('sig_narrow', 'Narrow core', mass,
                                          cb_mean, sigma1, alpha1, n1, alpha1b, n1b)

        sigma2 = ROOT.RooRealVar('sigma2', 'sigma2', 0.08, 0.03, 0.2)
        alpha2 = ROOT.RooRealVar('alpha2', 'alpha2', 1.0, 0.1, 5.0)
        n2     = ROOT.RooRealVar('n2', 'n2', 3.0, 1.0, 15.0)
        sig_broad = ROOT.RooCrystalBall('sig_broad', 'Broad radiative tail', mass,
                                         cb_mean, sigma2, alpha2, n2)

        frac_narrow = ROOT.RooRealVar('frac_narrow', 'frac_narrow', 0.5, 0.0, 1.0)
        sig_pdf = ROOT.RooAddPdf('sig_pdf', 'Double CB signal',
                                  ROOT.RooArgList(sig_narrow, sig_broad),
                                  ROOT.RooArgList(frac_narrow))

        shape_vars = {'mean': cb_mean, 'sigma1': sigma1, 'alpha1': alpha1, 'n1': n1,
                       'alpha1b': alpha1b, 'n1b': n1b, 'sigma2': sigma2,
                       'alpha2': alpha2, 'n2': n2, 'frac_narrow': frac_narrow}
        keepalive = [sig_narrow, sig_broad]
    else:
        raise ValueError(f'Unknown model_type {model_type}')

    return sig_pdf, shape_vars, keepalive


def do_fit(h, signal_only=False, savename=None, get_params=False, signal_params=None, bkg_shape_deg=3, printlevel=ROOT.RooFit.PrintLevel(-1), chi2_threshold=15.0, max_tries=20, shape_constraint=None, free_fit=False, model_type='single_cb'):

    data_yield = h.GetSumOfWeights()
    data_min = h.GetXaxis().GetXmin()
    data_max = h.GetXaxis().GetXmax()

    mass = ROOT.RooRealVar('mass', 'mass', 2.5, 3.4)
    data = ROOT.RooDataHist('data', 'data', mass, h)

    sig_coeff =  ROOT.RooRealVar('sig_coeff', 'sig_coeff', data_yield*.05, 0, data_yield*1.5)
    # cb_mean =   ROOT.RooRealVar('cb_mean', 'cb_mean', 3.0969,  3.0,  3.15)
    # cb_sigma =  ROOT.RooRealVar('cb_sigma', 'cb_sigma', 0.03, 0.01,  0.1)
    # cb_alpha = ROOT.RooRealVar('cb_alpha', 'cb_alpha', 0.5, 0.1, 30.)
    # cb_n =     ROOT.RooRealVar('cb_n', 'cb_n', 10.0, 1.01, 40.)
    # cb_alpha2 = ROOT.RooRealVar('cb_alpha2', 'cb_alpha2', 1.5, 0.1, 5.0)
    # cb_n2 = ROOT.RooRealVar('cb_n2', 'cb_n2', 2.0, 1.0, 15.0)
    
    # sig_pdf = ROOT.RooCrystalBall('sig_pdf', 'Signal Fit', mass, cb_mean, cb_sigma, cb_alpha, cb_n)
    # sig_pdf = ROOT.RooCrystalBall('sig_pdf', 'Signal Fit', mass, cb_mean, cb_sigma, cb_alpha, cb_n, cb_alpha2, cb_n2)
    sig_pdf, shape_vars, sig_keepalive = build_signal_pdf(mass, model_type)

    bkg_coeff =  ROOT.RooRealVar('bkg_coeff', 'bkg_coeff', data_yield*.95, 0, data_yield)
    # bern_pars = [ROOT.RooRealVar(f'bern_p{i}', f'bern_p{i}', 1, 0.01, 10) for i in range(bkg_shape_deg)]
    # bkg_pdf = ROOT.RooBernstein('bkg_pdf', 'Background Fit', mass, ROOT.RooArgList(*bern_pars))
    bern_pars = [ROOT.RooRealVar(f'bern_p{i}', f'bern_p{i}', 0.1, -1.0, 1.0) for i in range(bkg_shape_deg)]
    bkg_pdf = ROOT.RooChebychev('bkg_pdf', 'Background Fit', mass, ROOT.RooArgList(*bern_pars))

    # shape_vars = {'mean': cb_mean, 'sigma': cb_sigma, 'alpha': cb_alpha, 'n': cb_n, 'alpha2': cb_alpha2, 'n2': cb_n2,}
    extra_constraints = []
    keepalive = list(sig_keepalive)

    # Default tail constraint (kept as fallback if not overridden by MC)
    if free_fit:
        # No constraints at all — used for MC shape extraction.
        # Widen ranges since we have no prior on where the true shape sits.
        # cb_mean.setRange(2.95, 3.15)
        # cb_sigma.setRange(0.01, 0.25)
        # cb_alpha.setVal(1.5)
        # cb_alpha.setRange(0.3, 5.0)
        # cb_n.setVal(2.0)
        # cb_n.setRange(1.0, 15.0)
        # cb_alpha2.setVal(1.5)
        # cb_alpha2.setRange(0.3, 5.0)
        # cb_n2.setVal(2.0)
        # cb_n2.setRange(1.0, 15.0)
        pass
    elif shape_constraint:
        for name, (central, width) in shape_constraint.items():
            if name not in shape_vars:
                continue
            var = shape_vars[name]
            var.setVal(central)
            if width is None:
                var.setConstant(True)
            else:
                c_val = ROOT.RooConstVar(f'{name}_mc_c', f'{name}_mc_c', central)
                c_wid = ROOT.RooConstVar(f'{name}_mc_s', f'{name}_mc_s', width)
                constraint = ROOT.RooGaussian(f'{name}_mc_constraint', f'Constraint on {name}', var, c_val, c_wid)
                extra_constraints.append(constraint)
                keepalive.extend([c_val, c_wid, constraint])
    else:
        if 'mean' in shape_vars:
            c_val = ROOT.RooConstVar("mean_c", "mean_c", 3.0969)
            c_wid = ROOT.RooConstVar("mean_s", "mean_s", 0.01)
            constraint = ROOT.RooGaussian(
                "mean_constraint_pdf", "Constraint on mean", shape_vars['mean'], c_val, c_wid
            )
            extra_constraints.append(constraint)
            keepalive.extend([c_val, c_wid, constraint])
        # c_val = ROOT.RooConstVar("mean_c", "mean_c", 3.0969)
        # c_wid = ROOT.RooConstVar("mean_s", "mean_s", 0.01)
        # constraint = ROOT.RooGaussian("mean_constraint_pdf", "Constraint on cb_mean", cb_mean, c_val, c_wid)
        # extra_constraints.append(constraint)
        # keepalive.extend([c_val, c_wid, constraint])

    # if free_fit or (shape_constraint and 'n' in shape_constraint):
    #     pass  # skip default n constraint, either unconstrained or MC-supplied
    # else:
    #     n_c = ROOT.RooConstVar("n_central_val", "n_central_val", 10.0)
    #     n_s = ROOT.RooConstVar("n_sigma_val", "n_sigma_val", 3.0)
    #     constraint = ROOT.RooGaussian("n_constraint_pdf", "Constraint on cb_n", cb_n, n_c, n_s)
    #     extra_constraints.append(constraint)
    #     keepalive.extend([n_c, n_s, constraint])

    if signal_only:
        fit_model = ROOT.RooAddPdf('fit_model', 'Signal Fit', ROOT.RooArgList(sig_pdf), ROOT.RooArgList(sig_coeff))
    else:
        fit_model = ROOT.RooAddPdf('fit_model', 'Signal + Background Fit', ROOT.RooArgList(sig_pdf, bkg_pdf), ROOT.RooArgList(sig_coeff, bkg_coeff))

    constraint_set = ROOT.RooArgSet(*extra_constraints) if extra_constraints else ROOT.RooArgSet()
    keepalive.append(constraint_set)

    staged_names = [n for n in shape_vars if n != 'mean']

    fit_result = None
    chi2 = float('inf')
    attempt = 0
    sig_unc_ratio = 1
    good_fit = False
    used_sumw2_fallback = False

    # fit_args = [
    #     data,
    #     ROOT.RooFit.Save(True),
    #     ROOT.RooFit.Offset(True),
    #     # ROOT.RooFit.Minos(True),
    #     printlevel
    # ]

    # if constraint_set.getSize() > 0:
    #     # Pass constraint_set directly, without recreating it
    #     fit_args.append(ROOT.RooFit.ExternalConstraints(constraint_set))

    base_args = [ROOT.RooFit.Save(True), ROOT.RooFit.IntegrateBins(1e-3), printlevel]

    while not good_fit and attempt < max_tries:
        if fit_result:
            fit_result.Delete()
    
        for name in staged_names:
            shape_vars[name].setConstant(True)
 
        if free_fit:
            stage1_result, sw2_fallback_1 = fit_with_sumw2_fallback(
                fit_model, data, [ROOT.RooFit.Save(True), printlevel]
            )
        else:
            fit_args = [data] + base_args
            stage1_result = fit_model.fitTo(*fit_args)
        
        # if free_fit:
        #     fit_result, used_fallback = fit_with_sumw2_fallback(
        #         fit_model, data,
        #         [ROOT.RooFit.Save(True), printlevel],
        #         printlevel,
        #     )
        #     if used_fallback:
        #         print(f"  [{savename.stem if savename else ''}] WARNING: SumW2 covariance "
        #               f"correction was singular ?~@~T used SumW2Error(False) fallback. "
        #               f"Treat shape parameter errors from this bin as approximate; "
        #               f"consider inflating them further before using as a data constraint.")
        # else:
        #     fit_result = fit_model.fitTo(*fit_args)

        for name in staged_names:
            var = shape_vars[name]
            var.setConstant(False)
            step = max(1e-4, (var.getMax() - var.getMin()) * 0.05)
            var.setError(step)
 
        if constraint_set.getSize() > 0:
            base_args.append(ROOT.RooFit.ExternalConstraints(constraint_set))
 
        if free_fit:
            fit_result, used_sumw2_fallback = fit_with_sumw2_fallback(fit_model, data, base_args)
            if used_sumw2_fallback and savename:
                print(f"  [{savename.stem}] WARNING: SumW2 covariance correction was "
                      f"singular -- used SumW2Error(False) fallback. Treat shape "
                      f"parameter errors from this bin as approximate.")
        else:
            fit_args = [data] + base_args
            fit_result = fit_model.fitTo(*fit_args)
 

        fit_status = fit_result.status()
        cov_qual = fit_result.covQual()
        if fit_status >= 0 and cov_qual >= 2:
            sig_unc_ratio = sig_coeff.getError() / sig_coeff.getVal() if sig_coeff.getVal() > 0 else float('inf')
            
            tmp_frame = mass.frame(ROOT.RooFit.Title('tmp'))
            data.plotOn(tmp_frame)
            if signal_only:
                fit_model.plotOn(tmp_frame, ROOT.RooFit.Normalization(sig_coeff.getVal(),ROOT.RooAbsReal.NumEvent))
            else:
                fit_model.plotOn(tmp_frame, ROOT.RooFit.Normalization(sig_coeff.getVal()+bkg_coeff.getVal(),ROOT.RooAbsReal.NumEvent))
            
            chi2 = tmp_frame.chiSquare('fit_model_Norm[mass]', 'h_data', len(fit_result.floatParsFinal()))
            tmp_frame.Delete()

            if not math.isnan(chi2) and chi2 < chi2_threshold and sig_unc_ratio < 1:
                good_fit = True
            
        if fit_result and not good_fit:
            params_to_randomize = [cb_sigma, cb_alpha, cb_n] + bern_pars 
            for param in params_to_randomize:
                current_val = param.getVal()
                min_val = param.getMin()
                max_val = param.getMax()
                
                perturbation_range = (max_val - min_val) * 0.1
                random_shift = np.random.uniform(-perturbation_range, perturbation_range)
                new_val = current_val + random_shift
                new_val = max(min_val, min(max_val, new_val))
                
                param.setVal(new_val)
                param.setError(0.0)
        

        if fit_result and not good_fit:
            params_to_randomize = [var for name, var in shape_vars.items()] + bern_pars 
            for param in params_to_randomize:
                current_val = param.getVal()
                min_val = param.getMin()
                max_val = param.getMax()
                
                perturbation_range = (max_val - min_val) * 0.1
                random_shift = np.random.uniform(-perturbation_range, perturbation_range)
                new_val = current_val + random_shift
                new_val = max(min_val, min(max_val, new_val))
                
                param.setVal(new_val)
                param.setError(0.0)
        
        attempt += 1


    if not good_fit:
        print(f"WARNING: Fit failed to converge for {savename.stem} after {max_tries} attempts.")
        print(f"  Last attempt: status={fit_status}, covQual={cov_qual}, chi2={chi2:.2f}, sig_unc_ratio={sig_unc_ratio:.3f}")

    if attempt >= max_tries and not good_fit:
        print(f"WARNING: Fit failed to converge for {savename.stem} after {max_tries} attempts.")
        if get_params:
           return [None, None, None, None, None] if not signal_only else [None, None, None]
    #     if get_params:
    #         return [None, None, None, None, None] if not signal_only else [None, None, None]
    #     return [None, None, None, None] if not signal_only else [None, None]

    
    check_parameter_limits(savename, fit_result)
    out = [sig_coeff.getVal(), sig_coeff.getError()]
    # params_output = {
    #     'cb_mean'   : (cb_mean.getVal(),  cb_mean.getError()),
    #     'cb_sigma'  : (cb_sigma.getVal(), cb_sigma.getError()),
    #     'cb_alpha'  : (cb_alpha.getVal(), cb_alpha.getError()),
    #     'cb_n'      : (cb_n.getVal(),     cb_n.getError()),
    #     'cb_alpha2' : (cb_alpha2.getVal(), cb_alpha2.getError()),
    #     'cb_n2'     : (cb_n2.getVal(),     cb_n2.getError()),
    # }
    params_output = {name: (var.getVal(), var.getError()) for name, var in shape_vars.items()}
    params_output['chi2_ndf'] = chi2
    params_output['model_type'] = model_type
    print(params_output)

    if not signal_only:
        out.extend([bkg_coeff.getVal(), bkg_coeff.getError()])
        params_output.update({
            # **{i.GetName() : i for i in poly_pars},
            **{i.GetName() : i for i in bern_pars},
        })

    if not signal_only:
        out.extend([bkg_coeff.getVal(), bkg_coeff.getError()])
        params_output.update({i.GetName(): i for i in bern_pars})

    out += [params_output] if get_params else []
    out.append(chi2)

    frame = mass.frame(ROOT.RooFit.Title(str(savename.stem)))
    data.plotOn(frame)

    if signal_only:
        # fit_model.plotOn(frame, ROOT.RooFit.Name(fit_model.GetName()), ROOT.RooFit.LineColor(38), ROOT.RooFit.Normalization(sig_coeff.getVal(),ROOT.RooAbsReal.NumEvent))
        fit_model.plotOn(frame, ROOT.RooFit.Name(fit_model.GetName()), ROOT.RooFit.LineColor(38))
        h_pull = frame.pullHist()
    else:
        # fit_model.plotOn(frame, ROOT.RooFit.Name(fit_model.GetName()), ROOT.RooFit.LineColor(38), ROOT.RooFit.Normalization(sig_coeff.getVal()+bkg_coeff.getVal(),ROOT.RooAbsReal.NumEvent))
        fit_model.plotOn(frame, ROOT.RooFit.Name(fit_model.GetName()), ROOT.RooFit.LineColor(38))
        h_pull = frame.pullHist()
        fit_model.plotOn(frame, ROOT.RooFit.Components('sig_pdf'), ROOT.RooFit.LineColor(32), ROOT.RooFit.LineStyle(ROOT.kDashed))
        fit_model.plotOn(frame, ROOT.RooFit.Components('bkg_pdf'), ROOT.RooFit.LineColor(46), ROOT.RooFit.LineStyle(ROOT.kDashed))

    # fit_model.paramOn(frame, ROOT.RooFit.Layout(0.12, 0.3, 0.88), ROOT.RooFit.Format('NE', ROOT.RooFit.FixedPrecision(3)))

    frame_pull = mass.frame(ROOT.RooFit.Title(' '))
    frame_pull.addPlotable(h_pull, 'P')

    n_params = len(fit_result.floatParsFinal()) if fit_result else 0
    chi2 = frame.chiSquare('fit_model', 'h_data', n_params) if not signal_only else frame.chiSquare('fit_model', 'h_data', n_params)
    # chi2 = frame.chiSquare()
    chi2_text = ROOT.TLatex(0.7, 0.8, '#chi^{{2}}/ndf = {}'.format(round(chi2,1)))
    chi2_text.SetTextSize(0.05)
    chi2_text.SetNDC(ROOT.kTRUE)

    text = ROOT.TLatex(0.7, 0.7, f'N_{{J/#psi}} = {round(sig_coeff.getVal())} #pm {round(sig_coeff.getError(),1)}')
    text.SetTextSize(0.05)
    text.SetNDC(ROOT.kTRUE)

    c = ROOT.TCanvas('c', ' ', 800, 600)
    pad1 = ROOT.TPad('pad1', 'pad1', 0, 0.3, 1, 1.0)
    pad1.SetBottomMargin(0.02)
    pad1.SetGridx()
    pad1.Draw()
    c.cd()
    pad2 = ROOT.TPad('pad2', 'pad2', 0, 0.05, 1, 0.3)
    pad2.SetTopMargin(0.02)
    pad2.SetBottomMargin(0.2)
    pad2.SetGridx()
    pad2.Draw()

    pad1.cd()
    frame.Draw()
    ax_y_main = frame.GetYaxis()
    ax_x_main = frame.GetXaxis()
    ax_x_main.SetLabelOffset(3.)

    chi2_text.Draw()
    text.Draw()

    pad2.cd()
    frame_pull.Draw()

    ax_y_pull = frame_pull.GetYaxis()
    ax_x_pull = frame_pull.GetXaxis()

    line = ROOT.TLine(ax_x_pull.GetXmin(), 0, ax_x_pull.GetXmax(), 0)
    line.SetLineStyle(7)
    line.Draw()

    ax_y_pull.SetTitle('#frac{y - y_{fit}}{#sigma_{y}}')
    ax_y_pull.SetTitleOffset(.35)
    ax_y_pull.SetNdivisions(8)

    ax_y_pull.SetTitleSize(2.8*ax_y_main.GetTitleSize())
    ax_y_pull.SetLabelSize(2.8*ax_y_main.GetLabelSize())
    ax_x_pull.SetTitleSize(2.8*ax_x_main.GetTitleSize())
    ax_x_pull.SetLabelSize(2.8*ax_x_main.GetLabelSize())


    c.SaveAs(str(savename))
    c.Close()

    return out

def fit_signal_shape_with_fallback(hist, savename, printlevel,
                                    chi2_flag_threshold=100.0):
    """
    Fit an MC (truth-matched, signal-only) histogram to extract shape
    parameters. Tries the simpler single_cb model first; if the resulting
    chi2/ndf indicates real shape mismodeling (as opposed to the expected
    stats-inflated chi2 of a high-stat template fit), retries with the
    double_cb model.
 
    Returns a dict: {'params': {...}, 'model_type': 'single_cb'|'double_cb'}
    or None if both fits fail.
    """
    out = do_fit(
        hist, signal_only=True, get_params=True, free_fit=True,
        chi2_threshold=1e6, model_type='single_cb',
        savename=savename, printlevel=printlevel
    )
    if out[0] is None:
        return None
 
    params = out[-2]
    chi2_ndf = out[-1]
 
    if chi2_ndf is not None and chi2_ndf > chi2_flag_threshold:
        print(f"  [{savename.stem if savename else ''}] single_cb chi2/ndf={chi2_ndf:.1f} "
              f"-- refitting with double_cb model")
        out2 = do_fit(
            hist, signal_only=True, get_params=True, free_fit=True,
            chi2_threshold=1e6, model_type='double_cb',
            savename=savename.with_stem(savename.stem + '_doublecb') if savename else None,
            printlevel=printlevel
        )
        if out2[0] is not None:
            params = out2[-2]
            chi2_ndf = out2[-1]
            print(f"  [{savename.stem if savename else ''}] double_cb chi2/ndf={chi2_ndf:.1f}")
 
    return {'params': params, 'model_type': params.get('model_type', 'single_cb')}

def fit_mc_shapes(plot_cfg, mc_file, fit_output_file, printlevel):
    """Returns {bin_index: {'num': {...}, 'denom': {...}}} of MC shape params + errors."""
    n_bins = len(plot_cfg.bins) - 1
    shapes = {}

    for ibin in range(1, n_bins + 1):
        combined_num, combined_denom = None, None

        for mc_trig in plot_cfg.mc_triggers:
            path = mc_trig['path'].strip('"')
            weight = mc_trig['weight']
            num_hist = mc_file.Get(f'hists/diel_m_{path}_num_{plot_cfg.var}binned')
            denom_hist = mc_file.Get(f'hists/diel_m_{path}_denom_{plot_cfg.var}binned')

            num_proj = num_hist.ProjectionX(f'num_bin_{ibin}_{path}', ibin, ibin)
            denom_proj = denom_hist.ProjectionX(f'denom_bin_{ibin}_{path}', ibin, ibin)
            num_proj.Scale(weight)
            denom_proj.Scale(weight)

            if combined_num is None:
                combined_num = num_proj.Clone(f'mc_shape_num_bin{ibin}')
                combined_denom = denom_proj.Clone(f'mc_shape_denom_bin{ibin}')
            else:
                combined_num.Add(num_proj)
                combined_denom.Add(denom_proj)

        # MC is typically truth-matched signal -> try signal_only first;
        # fall back to signal+bkg if there's real combinatorial background present.
        num_shape = fit_signal_shape_with_fallback(
            combined_num,
            fit_output_file.with_stem(f'mc_num_shape_{plot_cfg.name}_bin{ibin-1}'),
            printlevel
        )
        denom_shape = fit_signal_shape_with_fallback(
            combined_denom,
            fit_output_file.with_stem(f'mc_denom_shape_{plot_cfg.name}_bin{ibin-1}'),
            printlevel
        )
    
        shapes[ibin - 1] = {'num': num_shape, 'denom': denom_shape}

    return shapes

def build_constraint(mc_shape_entry, inflate=10.0, sigma_inflate=20.0,
                      tail_inflate=1.5, n_inflate=1.5, mean_width=0.015):
    """
    mc_shape_entry: {'params': {...}, 'model_type': 'single_cb'|'double_cb'}
    (as returned by fit_signal_shape_with_fallback / fit_mc_shapes)
 
    Always returns a constraint dict keyed for a SINGLE_CB data fit: data
    statistics are generally assumed insufficient to resolve genuine
    bimodal structure (see conversation re: bin0 MC needing double_cb).
    If the MC bin needed double_cb, this projects down to an effective
    single_cb constraint with substantially inflated widths, reflecting the
    extra shape complexity the single_cb model can't capture -- i.e. we
    tell the data fit "don't trust this constraint too tightly" rather than
    forcing 10 parameters onto a low-stat data bin (Option A from the
    discussion; see Option B in do_fit's model_type if a data bin's own
    pull plot later shows it needs the full double_cb treatment).
    """
    if mc_shape_entry is None:
        return None
 
    params = mc_shape_entry['params']
    model_type = mc_shape_entry.get('model_type', 'single_cb')
 
    if model_type == 'single_cb':
        return {
            'sigma': (params['sigma'][0], params['sigma'][1] * sigma_inflate),
            'alpha': (params['alpha'][0], params['alpha'][1] * inflate),
            'n': (params['n'][0], params['n'][1] * n_inflate),
            'alpha2': (params['alpha2'][0], params['alpha2'][1] * tail_inflate),
            'n2': (params['n2'][0], params['n2'][1] * tail_inflate),
            'mean': (params['mean'][0], mean_width),
        }
 
    elif model_type == 'double_cb':
        frac = params['frac_narrow'][0]
        eff_sigma = frac * params['sigma1'][0] + (1 - frac) * params['sigma2'][0]
        eff_alpha = frac * params['alpha1'][0] + (1 - frac) * params['alpha2'][0]
        eff_n = frac * params['n1'][0] + (1 - frac) * params['n2'][0]
        return {
            'sigma': (eff_sigma, params['sigma2'][1] * sigma_inflate * 3),
            'alpha': (eff_alpha, 0.3 * inflate),
            'n': (eff_n, 2.0 * inflate),
            'alpha2': (params['alpha1b'][0], 0.3 * tail_inflate * 3),
            'n2': (params['n1b'][0], 2.0 * tail_inflate * 3),
            'mean': (params['mean'][0], mean_width * 2),
        }
 
    raise ValueError(f'Unknown model_type in mc_shape_entry: {model_type}')

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, key):
        return self.get(key, False)

    def __setattr__(self, key, value):
        self[key] = DotDict(value) if isinstance(value, dict) else value

    def __delattr__(self, key):
        if key in self: del self[key]
        else: raise AttributeError(f"No attribute named {key}")




def make_plotlist(cfg):
    plots = []
    out_path = Path('./eff_plots/test') if cfg.test else Path(cfg.output.output_dir)
    data_in_path = Path(cfg.inputs.data_dir).parent / 'test' if cfg.test else Path(cfg.inputs.data_dir)
    mc_in_path = Path(cfg.inputs.mc_dir).parent / 'test' if cfg.test else Path(cfg.inputs.mc_dir)
    for plot_name, plot_dict in cfg.plots.items():
        # For new format: triggers is a dict with data_trigger
        data_trigger = plot_dict.triggers['data_trigger'] if isinstance(plot_dict.triggers, dict) else plot_dict.triggers[0]
        for var in plot_dict.variables:
            plots.append(DotDict({
                'name': '_'.join([plot_name, f'{var}binned']),
                'data_trigger': data_trigger,
                'data_file': data_in_path / plot_dict.files.data,
                'mc_file': mc_in_path / plot_dict.files.mc if hasattr(plot_dict.files, 'mc') else None,
                'output_file': out_path / Path('_'.join(['eff', data_trigger, f'{var}binned'])).with_suffix('.pdf'),
                'mc_triggers': plot_dict.mc_triggers if hasattr(plot_dict, 'mc_triggers') else [],
                'var': var,
                **assign_hist_format(f'{var}binned'),
            }))
    return plots

def get_hists(cfg):
    # Only for data, since MC is handled per-path
    hists = []
    hist_paths = [(
        cfg.data_file, 
        f'hists/diel_m_{cfg.data_trigger}_num_{cfg.var}binned', 
        f'hists/diel_m_{cfg.data_trigger}_denom_{cfg.var}binned',
    )]
    for fname, num_path, denom_path in hist_paths:
        subhists = []
        f = ROOT.TFile(str(fname))
        num_h = f.Get(num_path)
        denom_h = f.Get(denom_path)
        nbins = num_h.GetYaxis().GetNbins()
        for i in range(1, nbins + 1):
            try:
                num_bin = num_h.ProjectionX('num_bin', i, i)
                # print(f'bin {i} num: ',num_bin.GetEntries())
            except AttributeError:
                continue
            try:
                denom_bin = denom_h.ProjectionX('denom_bin', i, i)
                # print(f'bin {i} deonom: ',denom_bin.GetEntries())
            except AttributeError:
                continue
            subhists.append((copy.deepcopy(num_bin), copy.deepcopy(denom_bin)))
        hists.append(subhists)
    return hists

def make_eff_plot_dict(cfg):
    plot_list = make_plotlist(cfg)
    eff_plot_list = []
    for plot_cfg in plot_list:
        print(f'processing hists for {plot_cfg.name}')
        eff_dict = {
            'name': plot_cfg.name,
            'data_trigger': plot_cfg.data_trigger,
            'output_file': plot_cfg.output_file,
            'data_num_yields': [],
            'data_denom_yields': [],
            'mc_eff_mixture': [],
            'mc_triggers': plot_cfg.mc_triggers,
            'bins': plot_cfg.bins,
            'xlabel': plot_cfg.xlabel,
        }
        # Data
        data_hists = get_hists(plot_cfg)
        fit_output_file = plot_cfg.output_file.parent / 'fits' / plot_cfg.output_file.name

        mc_shapes = (
            fit_mc_shapes(plot_cfg, ROOT.TFile(str(plot_cfg.mc_file)), fit_output_file,
                          cfg.printlevel)
            if plot_cfg.mc_file and plot_cfg.mc_triggers else {}
        )
        for i, (data_num_hist, data_denom_hist) in enumerate(data_hists[0]):
            mc_num_shape = mc_shapes.get(i, {}).get('num')
            mc_denom_shape = mc_shapes.get(i, {}).get('denom')

            num_results = do_fit(
                data_num_hist,
                savename=fit_output_file.with_stem(f'data_num_fit_{plot_cfg.name}_bin{i}'),
                shape_constraint=build_constraint(mc_num_shape),
                printlevel=cfg.printlevel,
            )
            n_num_data, n_num_data_err = (num_results[0], num_results[1]) if num_results[0] is not None else (None, None)

            denom_results = do_fit(
                data_denom_hist,
                savename=fit_output_file.with_stem(f'data_denom_fit_{plot_cfg.name}_bin{i}'),
                shape_constraint=build_constraint(mc_denom_shape),
                printlevel=cfg.printlevel,
            )
            n_denom_data, n_denom_data_err = (denom_results[0], denom_results[1]) if denom_results[0] is not None else (None, None)

            eff_dict['data_num_yields'].append((n_num_data, n_num_data_err))
            eff_dict['data_denom_yields'].append((n_denom_data, n_denom_data_err))

        if plot_cfg.mc_file and plot_cfg.mc_triggers:
            mc_file = ROOT.TFile(str(plot_cfg.mc_file))
            n_bins = len(eff_dict['bins']) - 1
            mc_path_effs = []
            for mc_trig in plot_cfg.mc_triggers:
                path = mc_trig['path'].strip('"')
                weight = mc_trig['weight']
                num_hist_name = f'hists/diel_m_{path}_num_{plot_cfg.var}binned'
                denom_hist_name = f'hists/diel_m_{path}_denom_{plot_cfg.var}binned'
                num_hist = mc_file.Get(num_hist_name)
                denom_hist = mc_file.Get(denom_hist_name)
                effs = []
                for ibin in range(1, n_bins + 1):
                    num_proj = num_hist.ProjectionX('num_bin', ibin, ibin)
                    denom_proj = denom_hist.ProjectionX('denom_bin', ibin, ibin)


                    if denom_proj.GetSumOfWeights() > 0:
                        nNum, nNum_err = get_event_count(num_proj)
                        nDenom, nDenom_err = get_event_count(denom_proj)
                        eff = ufloat(nNum, nNum_err) / ufloat(nDenom, nDenom_err) 
                    else:
                        eff = ufloat(0, 0)
                    effs.append(eff * weight)
                mc_path_effs.append(effs)

            mc_eff_mixture = []
            for ibin in range(n_bins):
                eff_sum = sum(effs[ibin] for effs in mc_path_effs)
                mc_eff_mixture.append(eff_sum)
            eff_dict['mc_eff_mixture'] = mc_eff_mixture
        elif plot_cfg.mc_file:
            mc_file = ROOT.TFile(str(plot_cfg.mc_file))
            n_bins = len(eff_dict['bins']) - 1
            num_hist = mc_file.Get(f'hists/diel_m_{plot_cfg.data_trigger}_num_{plot_cfg.var}binned')
            denom_hist = mc_file.Get(f'hists/diel_m_{plot_cfg.data_trigger}_denom_{plot_cfg.var}binned')
            mc_effs = []
            for ibin in range(1, n_bins + 1):
                num_proj = num_hist.ProjectionX('num_bin', ibin, ibin)
                denom_proj = denom_hist.ProjectionX('denom_bin', ibin, ibin)
                if denom_proj.GetSumOfWeights() > 0:
                    nNum, nNum_err = get_event_count(num_proj)
                    nDenom, nDenom_err = get_event_count(denom_proj)
                    eff = ufloat(nNum, nNum_err) / ufloat(nDenom, nDenom_err)
                else:
                    eff = ufloat(0, 0)
                mc_effs.append(eff)
            eff_dict['mc_eff_mixture'] = mc_effs
        eff_plot_list.append(eff_dict)

    with open(Path(cfg.output.output_dir) / 'eff_plot_entries.pkl', 'wb') as f:
        pkl.dump(eff_plot_list, f)
    
    return eff_plot_list


def plot_efficiencies(eff_dicts, test=False):
    for d in eff_dicts:
        if test and ('test' not in d['name']):
            continue
        # elif not test and ('test' in d['name']):
        #     continue

        bins = d['bins']
        h_num_data = ROOT.TH1F('h_num_data', 'h_num_data', len(bins)-1, bins)
        h_denom_data = ROOT.TH1F('h_denom_data', 'h_denom_data', len(bins)-1, bins)
        h_num_mc = ROOT.TH1F('h_num_mc', 'h_num_mc', len(bins)-1, bins)
        h_denom_mc = ROOT.TH1F('h_denom_mc', 'h_denom_mc', len(bins)-1, bins)

        # Fill data as before
        for ibin, (data_num, data_denom) in enumerate(zip(d['data_num_yields'], d['data_denom_yields'])):
            if data_num[0] is not None and data_denom[0] is not None:
                h_num_data.SetBinContent(ibin+1, data_num[0])
                h_num_data.SetBinError(ibin+1, data_num[1])
                h_denom_data.SetBinContent(ibin+1, data_denom[0])
                h_denom_data.SetBinError(ibin+1, data_denom[1])


        # MC: use mixture if present
        if 'mc_eff_mixture' in d and d['mc_eff_mixture']:
            for ibin, eff in enumerate(d['mc_eff_mixture']):
                # For TEfficiency, set numerator = eff * denom, error propagation is handled by ufloat
                denom = 1.0  # or set to 1 for all bins, since TEfficiency cares about ratio
                num = eff.n
                err = eff.s
                h_num_mc.SetBinContent(ibin+1, num)
                h_num_mc.SetBinError(ibin+1, err)
                h_denom_mc.SetBinContent(ibin+1, denom)
                h_denom_mc.SetBinError(ibin+1, 0)
        else:
            for ibin, (mc_num, mc_denom) in enumerate(zip(d['mc_num_yields'], d['mc_denom_yields'])):
                h_num_mc.SetBinContent(ibin+1, mc_num[0])
                h_num_mc.SetBinError(ibin+1, mc_num[1])
                h_denom_mc.SetBinContent(ibin+1, mc_denom[0])
                h_denom_mc.SetBinError(ibin+1, mc_denom[1])

        eff_data = ROOT.TEfficiency(h_num_data, h_denom_data)
        eff_data.SetStatisticOption(ROOT.TEfficiency.kBBayesian)
        eff_mc = ROOT.TEfficiency(h_num_mc, h_denom_mc)
        eff_mc.SetStatisticOption(ROOT.TEfficiency.kFNormal)

        eff_plot = EfficiencyPlot(init_params={
            'title_string': ' ;' + d["xlabel"] + ';Efficiency',
            'xrange': (bins[0], bins[-1]),
            'yrange': (0., 1.1),
            'leg_scale': .65,
            'leg_header': '',
        })

        eff_plot.plotEfficiencies(
            eff_data,
            eff_mc,
            ratio=True,
            h1_title='ParkingDoubleMuonLowMass 2023 Data',
            h2_title='B^{+} #rightarrow J/#psi K^{+} MC',
            save=str(d['output_file']),
            addIntegral=False
        )


def make_sf_json(cfg, eff_dicts):
    outputs = {}
    for eff_dict in eff_dicts:
        matches = list(re.finditer(r'_([^_]+)binned(?=_|$)', eff_dict['name']))
        if not matches:
            raise ValueError(f"Could not find binvar in {eff_dict['name']}")

        binvar = matches[-1].group(1)
        # data_num = [ufloat(val, unc) for val, unc in eff_dict['data_num_yields']]
        # data_denom = [ufloat(val, unc) for val, unc in eff_dict['data_denom_yields']]
        # data_ratio = [n / d if d.n != 0 else ufloat(0, 0) for n, d in zip(data_num, data_denom)]
        # mc_ratio = eff_dict['mc_eff_mixture']
        # sfs = [d / m if m.n != 0 else ufloat(0, 0) for d, m in zip(data_ratio, mc_ratio)]

        # sfs = [(round(r.n,3), round(r.s,3)) for r in sfs]
        # data_ratio = [(round(r.n,3), round(r.s,3)) for r in data_ratio]
        # mc_ratio = [(round(r.n,3), round(r.s,3)) for r in mc_ratio]

        # output = {
        #     'binvar'    : binvar,
        #     'bins'      : list(eff_dict['bins'])[:-1],
        #     'sfs'       : sfs,
        #     'data_effs' : data_ratio,
        #     'mc_effs'   : mc_ratio,
        # }

        # outputs[eff_dict['name']] = outputs.get(eff_dict['name'], copy.deepcopy(output))

        data_num = [ufloat(val, unc) if val is not None else None for val, unc in eff_dict['data_num_yields']]
        data_denom = [ufloat(val, unc) if val is not None else None for val, unc in eff_dict['data_denom_yields']]

        # Propagate None through ratios
        data_ratio = [(n / d) if (n is not None and d is not None and d.n != 0) else None for n, d in zip(data_num, data_denom)]
        mc_ratio = eff_dict['mc_eff_mixture']
        sfs = [(d / m) if (d is not None and m.n != 0) else None for d, m in zip(data_ratio, mc_ratio)]

        # Format final output, preserving None
        sfs_out = [ (round(r.n, 3), round(r.s, 3)) if r is not None else None for r in sfs]
        data_ratio_out = [ (round(r.n, 3), round(r.s, 3)) if r is not None else None for r in data_ratio]
        mc_ratio_out = [ (round(r.n, 3), round(r.s, 3)) if r is not None else None for r in mc_ratio]

        output = {
            'binvar'    : binvar,
            'bins'      : list(eff_dict['bins'])[:-1],
            'sfs'       : sfs_out,
            'data_effs' : data_ratio_out,
            'mc_effs'   : mc_ratio_out,
        }

        outputs[eff_dict['name']] = outputs.get(eff_dict['name'], copy.deepcopy(output))

    out_path = Path('eff_hists') / 'test' if cfg.test else Path(cfg.output.output_dir)
    with open(out_path / 'sf_jsons' / 'trigger_sfs.json', 'w') as outfile:
        json.dump(outputs, outfile, indent=4)

def main(cfg):
    cfg = DotDict(cfg)

    if cfg.file is None:
        eff_dicts = make_eff_plot_dict(cfg)
    elif Path(cfg.file).is_file():
        with open(cfg.file,'rb') as f:
            eff_dicts = pkl.load(f)
    else:
        raise ValueError(f'Efficiency dict file {str(cfg.file)} is not readable')

    plot_efficiencies(eff_dicts, test=cfg.test)
    make_sf_json(cfg, eff_dicts)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='eff_plot_cfg_v2.yml', help='plot configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='only run test samples')
    parser.add_argument('-f', '--file', dest='file', default=None, help='make plots from pkl file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = DotDict(yaml.safe_load(f))

    os.makedirs(Path(cfg.output.output_dir), exist_ok=True)
    os.makedirs(Path(cfg.output.output_dir) / 'fits', exist_ok=True)
    os.makedirs(Path(cfg.output.output_dir) / 'sf_jsons', exist_ok=True)
    cfg.test = args.test
    cfg.verbose = args.verbose
    cfg.printlevel = set_verbosity(args.verbose)
    cfg.file = args.file

    main(cfg)

