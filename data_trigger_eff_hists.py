#!/usr/bin/env python
"""Data trigger-efficiency histograms (J/psi -> ee in ParkingDoubleMuonLowMass).
Fills num/denom (mass x sublead-pT) histograms per trigger path. Weight is 1 for every event."""
import argparse
import copy
import multiprocessing as mp
import os
import time
from pathlib import Path

import yaml

from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

from trigger_utils import DotDict, get_input_files
from producer_base import TriggerEfficiencyProducerBase


class DataTriggerEfficiencyProducer(TriggerEfficiencyProducerBase):
    def event_weight(self, event, electrons):
        return 1.0


def worker(params):
    p = PostProcessor(
        params['output_dir'],
        params['input_files'],
        cut=params['presel'],
        branchsel=None,
        modules=[DataTriggerEfficiencyProducer(params)],
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
    for job in cfg.data_samples:
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
        job.trigger_map = cfg.trigger_map
        job.hlt_filter_bits = cfg.hlt_filter_bits if cfg.hlt_filter_bits else {}
        job.trigobj_match = bool(global_cfg.trigobj_match)
        job.trigobj_dr = global_cfg.trigobj_dr if global_cfg.trigobj_dr else 0.1
        job_configs.append(copy.deepcopy(job))

    start_time = time.perf_counter()
    if 'mp' in global_cfg.run_strategy:
        n_cores = max(1, mp.cpu_count() // 2)
        with mp.Pool(processes=n_cores) as pool:
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
