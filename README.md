# Run 3 R(K) Trigger Efficiency Studies

This repository calculates the trigger efficiencies and Scale Factors (SFs) for the Run 3 R(K) analysis using an orthogonal dataset method (DoubleMuon).

## Setting Up

### Building Environment (First-Time Setup)
```bash
cmsrel CMSSW_14_0_6
cd CMSSW_14_0_6/src
cmsenv
git clone git@github.com:DiElectronX/r3k-trigeffs.git
cd $CMSSW_BASE/src/r3k-trigeffs

```

### Loading Environment (When Logging In)

```bash
cd .../r3k-trigeffs
cmsenv

```

## Procedure Overview

We utilize the 2022 DoubleMuon dataset (NanoAOD) as an orthogonal proxy to measure the efficiency of the di-electron triggers. The workflow consists of three main stages: Skimming, Histogramming, and Plotting/SF Calculation.

We employ a mixture of L1+HLT trigger paths. Scale Factors are derived by comparing Data (fitted J/psi yields) vs MC (cut-and-count weighted by luminosity fractions).

**Note:** Ensure you inspect all `.yml` config files before running to confirm dataset paths, cuts, and trigger lists are correct for your specific study.

---

### Step 1: Skimming

This step filters the NanoAOD events. It:

1. Uses a JSON filter to select events where the di-electron triggers were active.
2. Selects events firing a subset of DoubleMuon triggers that are mostly orthogonal to the electron component.
3. Applies rough cuts to emulate the di-electron phase space (>=2 opp. sign electrons, small dR/dz, pT/eta acceptance).

**Configuration:** `eff_skim_cfg.yml`

**Command:**

```bash
python3 skim_nanoaod.py -c eff_skim_cfg.yml

```

---

### Step 2: Histogramming

This step bins events for the numerator and denominator of the efficiency calculations.

* **Data:** Iterates through events, checks trigger flags, and fills num/denom histograms based on the subleading electron . This is performed for each trigger path in the mixture and the logical OR.
* **MC:** Similarly fills histograms but applies MC event weights based on FONLL corrections to the B meson  spectrum to correct for mismodeling.

**Configuration:** `eff_hist_cfg_v2.yml`

**Commands:**

### New workflow 
```bash
python3 data_trigger_eff_hists.py  -c eff_hist_cfg_v2.ym

python3 mc_trigger_eff_hists.py  -c eff_hist_cfg_v2.yml

```

*(Optional) PU reweighting*

```bash
# Run on produced data and MC hists 
python3 make_pu_weights.py --data eff_hists/{dir}/effs_data_trigger_OR.root --mc eff_hists/{dir}/effs_jpsi_resonant.root --trigger trigger_OR --out pu_weights.json

# Rerun MC hists (add pu json file to eff_hist_cfg_v2.yml)
python3 mc_trigger_eff_hists.py -c eff_hist_cfg_v2.yml
```


### Old workflow 

```bash
# Process Data
python3 make_data_efficiency_hists_v2.py -c eff_hist_cfg_v2.yml

# Process MC (with FONLL weights)
python3 make_mc_efficiency_hists_v2.py -c eff_hist_cfg_v2.yml

```

---

### Step 3: Plotting and Scale Factors

This step calculates efficiencies, plots them, and exports the Scale Factors (SFs).

* **Data:** Loads histograms bin-by-bin and fits a J/psi peak to extract signal yields, which are used to calculate efficiency per bin.
* **MC:** Performs a cut-and-count efficiency calculation for each trigger path. A weighted sum of path-by-path efficiencies is calculated based on the exclusive luminosity fractions (defined as where a specific path is the lowest unprescaled, operational trigger).
* **Output:** Plots of efficiencies, Data/MC ratios, and a JSON file containing the efficiencies and SFs.

**Configuration:** `eff_plot_cfg_v2.yml`

**Command:**

### New workflow

```bash
python3 eff_plot_fit.py -c eff_plot_cfg_v2.yml

```

### Old workflow 

```bash
python3 make_mc_efficiency_plots_v2.py -c eff_plot_cfg_v2.yml

```

---

### Analysis Application

For the general analysis using these corrections:

1. **Allocation:** Allocate MC samples to specific L1+HLT paths according to the exclusive luminosity fractions used in Step 3.
2. **Filtering:** Filter MC events based on whether they pass their allocated trigger path.
3. **Correction:** For passing events, apply the Scale Factor (SF) derived for the whole trigger mixture.

*Note: This mixture approach was adopted because statistics were insufficient to derive robust SFs binned in multiple dimensions (pT, dR) for individual trigger paths.*
