import os
import sys
import multiprocessing as mp
import argparse
import random
import string
import time
import json
import yaml
import glob
import array
import ROOT
from pathlib import Path
import copy
from pprint import pprint
from datetime import datetime
import shutil

sys.path.insert(0, os.path.abspath('crab_skimmer'))
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

from CRABAPI.RawCommand import crabCommand
from PhysicsTools.NanoAODTools.postprocessing.utils.crabhelper import inputFiles, runsAndLumis
from crab_skimmer.crab_cfg_template import config


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
        if key in self:
            del self[key]
        else:
            raise AttributeError(f'No attribute named {key}')

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self), memo))

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, DotDict) else v for k, v in self.items()}

class SkimEvents(Module):
    def __init__(self):
        self.writeHistFile = True
    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)
    def analyze(self, event):
        return True

class SkimEvents_wVertex(Module):
    def __init__(self):
        # --- Call the parent class's __init__ ---
        # This is CRITICAL. It initializes self.out, self.histFile, etc.
        Module.__init__(self)
        
        self.writeHistFile = False
        
        # --- Initialize buffers for new branches ---
        # 'f' is for float, 'i' is for integer
        self.dielectron_mass_kin = array.array('f', [0.])
        self.dielectron_vtxChi2 = array.array('f', [0.])
        self.dielectron_mass_sv = array.array('f', [0.])
        self.dielectron_svIdx = array.array('i', [0])
        
        # --- TLorentzVectors for kinematic calculation ---
        self.ele1_vec = ROOT.TLorentzVector()
        self.ele2_vec = ROOT.TLorentzVector()

    def beginJob(self, histFile=None, histDirName=None):
        # beginJob is called *before* the output tree is created.
        # Do not define branches here.
        Module.beginJob(self, histFile, histDirName)
        
    def initbranches(self, tree):
        """
        initbranches is called *after* the output tree is created.
        This is the correct place to define new branches.
        """
        # --- Define new branches in the output TTree ---
        # Use tree.Branch, which is the TTree method, as self.out may not be set yet.
        tree.Branch("Dielectron_mass_kin", self.dielectron_mass_kin, "Dielectron_mass_kin/F")
        tree.Branch("Dielectron_vtxChi2", self.dielectron_vtxChi2, "Dielectron_vtxChi2/F")
        tree.Branch("Dielectron_mass_SV", self.dielectron_mass_sv, "Dielectron_mass_SV/F")
        tree.Branch("Dielectron_svIdx", self.dielectron_svIdx, "Dielectron_svIdx/I")

    def analyze(self, event):
        """
        Processes events that pass the PostProcessor's preselection cut.
        Calculates dielectron mass and saves related electron variables.
        """
        
        # --- Default "not found" values ---
        kin_mass = -99.
        sv_chi2 = -99.
        sv_mass = -99.
        sv_idx = -1

        if event.nElectron >= 2:
            # --- 1. Calculate Kinematic Mass (always) ---
            self.ele1_vec.SetPtEtaPhiM(
                event.Electron_pt[0], event.Electron_eta[0],
                event.Electron_phi[0], event.Electron_mass[0]
            )
            self.ele2_vec.SetPtEtaPhiM(
                event.Electron_pt[1], event.Electron_eta[1],
                event.Electron_phi[1], event.Electron_mass[1]
            )
            kin_mass = (self.ele1_vec + self.ele2_vec).M()

            # --- 2. Check for a Common Secondary Vertex ---
            # This logic leverages the pre-computed links in NanoAOD.
            # It checks if both electrons point to the *same* valid SV.
            lead_svIdx = event.Electron_svIdx[0]
            sublead_svIdx = event.Electron_svIdx[1]

            if (lead_svIdx >= 0 and lead_svIdx == sublead_svIdx):
                # A common, valid SV is found. Save its properties.
                common_idx = lead_svIdx
                
                # Check bounds just in case, though this should be guaranteed
                if common_idx < event.nSV:
                    sv_chi2 = event.SV_chi2[common_idx]
                    sv_mass = event.SV_mass[common_idx]
                    sv_idx = common_idx
                else:
                    print(f"Warning: Electron_svIdx {common_idx} is out of bounds for nSV {event.nSV}.")

        # --- 3. Fill the new branches for every event ---
        # We must fill the buffers *before* calling tree.Fill()
        # The PostProcessor will call tree.Fill() implicitly.
        self.dielectron_mass_kin[0] = kin_mass
        self.dielectron_vtxChi2[0] = sv_chi2
        self.dielectron_mass_sv[0] = sv_mass
        self.dielectron_svIdx[0] = sv_idx

        # We don't call self.out.fillBranch here.
        # By defining the branches with array buffers,
        # the TTree.Fill() call (done by the PostProcessor)
        # will automatically read the current value from the buffer.
        
        return True # Keep the event


def worker(params):
    pf = f'_skim_{"".join(random.choices(string.ascii_lowercase + string.digits, k=8))}'
    p = PostProcessor(
            params['output_path'],
            params['files'],
            cut=params['preselection'],
            branchsel=None,
            modules=[SkimEvents()],
            jsonInput=params['json'],
            prefetch=True,
            longTermCache=True,
            postfix=pf
    )
    p.run()


def main(cfg):
    datasets = {k:v for k,v in cfg.datasets.items() if k in cfg.sel_datasets} if cfg.sel_datasets else cfg.datasets
    assert datasets, 'No valid dataset given!'
    job_configs = []
    for job_name, params in datasets.items():
        if cfg.test and ('test' not in job_name):
            continue
        elif not cfg.test and ('test' in job_name):
            continue
        dic = {
            'name' : job_name,
            'json' : params.json_path if params.json_path else None,
            'output_path' : params.output_path,
            'preselection' : params.preselection,
        }
        if 'private' not in job_name:
            dic['files'] = params.files
        else:
            dic['files'] = [f for path in params.files for f in glob.glob(path, recursive=True)]
        job_dict = DotDict(dic)
        job_configs.append(copy.deepcopy(job_dict))

    start_time = time.perf_counter()
    if 'mp' in cfg.run_strategy:
        n_cores = mp.cpu_count()
        if cfg.verbose:
            print(''.join(['Distributing ~',str(len(job_configs)),' jobs to ', str(n_cores), ' cores...']))

        with mp.Pool(processes=n_cores) as pool:
            pool.map(worker, [job.to_dict() for job in job_configs])

    elif 'crab' in cfg.run_strategy:
        for params in job_configs:
            config_values = {
                'CUT_TEMPLATE'  : f"'{params.preselection}'" if params.preselection is not None else "None",
                'JSON_TEMPLATE' : f"'{os.sep.join(params.json.split(os.sep)[1:])}'" if params.json is not None else "None",
            }

            with open('crab_skimmer/crab_script_template.py', 'r') as f:
                content = f.read().format(**config_values)

            with open('crab_skimmer/crab_script.py', 'w') as f:
                f.write(content)
            if 'private' not in params.name:
                for dataset in params.files:  # iterate over each DAS dataset path
                    # derive a short dataset tag for the job name, e.g. "ParkingDoubleMuonLowMass0_Run2023B-NanoAODv15-v1"
                    dataset_tag = '_'.join(dataset.strip('/').split('/')[:2])
                    if 'jpsi' in params.name:
                        process = dataset_tag.split('_')[0].lower().replace('to', '')
                        condition = "bpix" if "BPix" in dataset_tag else ""
                        year = "2023" if "2023" in dataset_tag else ""
                        
                        parts = [process, condition, year]
                        dataset_tag = "_".join(filter(None, parts))
                    
                    config.General.workArea = 'crab_skimmer/crab_jobs'
                    config.General.requestName = '_'.join([
                        params.name,
                        dataset_tag,
                        datetime.now().strftime("%m_%d_%y")
                    ])
                    
                    config.Data.inputDataset = dataset  # single string
                    config.Data.inputDBS = 'global'
                    config.Data.splitting = 'FileBased'  # or 'LumiBased' if using json
                    config.Data.unitsPerJob = 80
                    config.Data.outLFNDirBase = params.output_path[params.output_path.index('/store'):]
                    
                    if params.json:
                        config.Data.lumiMask = params.json
                    config.Site.whitelist = []  
                    config.Site.ignoreGlobalBlacklist = False
                    
                    res = crabCommand('submit', config=config)
            else:
                config.General.workArea = 'crab_skimmer/crab_jobs'
                config.General.requestName = '_'.join([params.name,datetime.now().strftime("%m_%d_%y")])
                
                config.Data.userInputFiles = [f.replace('/eos/cms/', 'root://cmsxrootd.fnal.gov//') for f in params.files]
                config.Data.unitsPerJob = 80
                config.Data.totalUnits = len(params.files)
                config.Data.outLFNDirBase = params.output_path[params.output_path.index('/store'):]

                request_path = Path(config.General.workArea) / ('crab_'+config.General.requestName)
                if request_path.exists():
                    if input(f"CRAB request '{request_path}' exists. Overwrite? (y/n): ").strip().lower() == 'y':
                        shutil.rmtree(request_path)
                    else:
                        raise FileExistsError('CRAB request already exists. Try a job with another name.')

                res = crabCommand('submit', config=config)
            print(f'Submitted job {params.name} to CRAB batch system\n',res)
    elif 'serial' in cfg.run_strategy:
        for params in job_configs:
            worker(params)
    else:
        raise ValueError(f'Invalid mode "{cfg.run_strategy}". Must be one of ["serial", "mp", "crab"]')


    else:
        raise ValueError(f'Invalid mode "{cfg.run_strategy}". Must be one of ["serial", "mp", "crab"]')

    if cfg.verbose:
        finish_time = time.perf_counter()
        print(''.join(['Finished in ', str(round(finish_time-start_time)), ' seconds']))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', type=str, default='eff_skim_cfg.yml', help='skim configuration file (.yml)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='printouts to stdout')
    parser.add_argument('-d', '--datasets', dest='datasets', nargs='+', help='target datasets to run (from cfg file)')
    parser.add_argument('-t', '--test', dest='test', action='store_true', help='only run test samples')
    parser.add_argument('-m', '--method', dest='method', choices=['serial', 'crab', 'mp'], help='execution mode')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = DotDict(yaml.safe_load(f))
    
    cfg.verbose = args.verbose
    cfg.sel_datasets = args.datasets
    cfg.test = args.test
    cfg.run_strategy = args.method if args.method else cfg['config']['run_strategy']
    
    main(cfg)
