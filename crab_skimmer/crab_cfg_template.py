#!/usr/bin/env python3

import CRABClient
from WMCore.Configuration import Configuration
from CRABClient.UserUtilities import config


config = Configuration()

config.section_('General')
config.General.workArea = 'Skim_Jobs'
config.General.requestName = ''
config.General.transferLogs = True

config.section_('JobType')
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'crab_skimmer/PSet.py'
config.JobType.scriptExe = 'crab_skimmer/crab_script.sh'
config.JobType.inputFiles = [
        'skim_nanoaod.py',
        'crab_skimmer/crab_script.py',
        'crab_skimmer/crab_script.py',
        'crab_skimmer/FrameworkJobReport.xml',
        'jsons/',
]
config.JobType.disableAutomaticOutputCollection = True
config.JobType.outputFiles = ['output_skim.root']

config.section_('Data')
#config.Data.inputDataset = ''
config.Data.userInputFiles = ''
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.totalUnits = 63
config.Data.outLFNDirBase = ''
config.Data.publication = False

config.section_('Site')
config.Site.storageSite = 'T2_CH_CERN'
config.Site.whitelist = ['T2_CH_CERN']
