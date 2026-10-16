"""Single source of truth for histogram binning.

Imported by the histogram producers (data + MC) AND by the fit/plot script, so the
edges used to fill the histograms and the edges used to label/plot them cannot drift.
"""
import numpy as np

# Dielectron mass histogram: 0.01 GeV bins, edges land exactly on 2.5 and 3.4 (the fit range)
DIEL_M_BINS = np.linspace(2.0, 4.0, 201, dtype=np.double)
FIT_RANGE = (2.5, 3.4)

# Sublead-electron pT [GeV]. Last edge is a real edge (events above it go to overflow and
# are not used), so the plotted x-range is the filled x-range.
PT_BINS = np.array([4., 5., 6., 8., 12., 20.], dtype=np.double)

DR_BINS = np.array([0., 0.12, 0.2, 0.28, 0.44, 1.], dtype=np.double)
ETA_BINS = np.array([-1.22, -0.7, -0.2, 0.2, 0.7, 1.22], dtype=np.double)

# Number of reconstructed primary vertices: integer edges
NPV_BINS = np.linspace(0., 100., 101, dtype=np.double)

# Lead-electron pT x DeltaR, filled for the denominator so a data/MC kinematic reweighting
# (acceptance closure) can be derived without rerunning the skim.
LEADPT_BINS = np.array([4., 5., 6., 7., 8., 10., 12., 16., 20., 30., 100.], dtype=np.double)
DR_FINE_BINS = np.linspace(0., 1., 21, dtype=np.double)
