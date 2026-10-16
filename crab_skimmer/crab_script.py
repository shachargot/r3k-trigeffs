#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import *
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.utils.crabhelper import inputFiles, runsAndLumis


@dataclass
class ParamSet:
    outDir: str = '.'
    inputFiles: list[str] = field(default_factory=lambda: inputFiles())
    cut: str = 'nElectron > 1 && Electron_pt[0] > 4. && Electron_pt[1] > 4. && abs(Electron_eta[0]) < 1.22 && abs(Electron_eta[1]) < 1.22 && abs(Electron_dz[0] - Electron_dz[1]) <= 1.  && sqrt(2*Electron_pt[0]*Electron_pt[1]*(cosh(Electron_eta[0]-Electron_eta[1]) - cos(Electron_phi[0]-Electron_phi[1]))) < 5.  && sqrt(pow(Electron_eta[0]-Electron_eta[1],2) + pow(acos(cos(Electron_phi[0]-Electron_phi[1])),2)) > 0.03 && Electron_charge[0] + Electron_charge[1] == 0'
    json: str = None
    branchesIn: str = None
    branchesOut: str = None
    label: str = '_skim'
    friend: bool = False
    maxEntries: int = None
    haddFileName: str = 'output_skim.root'


class SkimEvents(Module):
    def __init__(self):
        self.writeHistFile = True
    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)
    def analyze(self, event):
        return True


params = ParamSet()
p = PostProcessor(
        params.outDir,
        params.inputFiles,
        cut=params.cut,
        postfix=params.label,
        branchsel=params.branchesIn,
        outputbranchsel=params.branchesOut,
        modules=[SkimEvents()],
        haddFileName=params.haddFileName,
        compression=('LZMA:9'),
        friend=params.friend,
        maxEntries=params.maxEntries,
        jsonInput=params.json,
        provenance=True,
        fwkJobReport=True,
)
p.run()

print("DONE")
