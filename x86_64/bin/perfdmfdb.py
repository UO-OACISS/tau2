#!/usr/bin/env sh /home/scottb/tau2/x86_64/bin/ppscript

# ParaProf PyScript

from edu.uoregon.tau.paraprof.script import *
from edu.uoregon.tau.paraprof.treetable import *
from edu.uoregon.tau.paraprof.enums import *
from edu.uoregon.tau.paraprof import *
from edu.uoregon.tau.perfdmf import *
from javax import swing
import java
import edu
import sys
import os

def main():
  arg = sys.argv[1:] 
  
  if (arg[0] == 'download'):
    if (len(arg) > 5):
      downloaddb(arg[3], arg[4], arg[1], arg[2], arg[5])
    elif (len(arg) > 2):
      downloaddb(arg[3], "", arg[1], arg[2])
    else:  
      downloaddb(arg[1]) 
    print arg[1]
    #print filewriter
  elif (arg[0] == 'upload'):
    filelist = arg[3:]
    #print "file list ", filelist
    #print arg[1], arg[2]
    uploaddb(filelist, arg[1], arg[2])
  elif (arg[0] == 'list'):
    list = []
    if (not arg[1] is None):
      list = list_trials(arg[1], arg[2])
    else:
      list = list_trials()
    for l in list:
      print l
  elif (arg[0] == 'list_experiments'):
    list = []
    if (not arg[1] is None):
      list = list_experiments(arg[1])
    for l in list:
      print l

def downloaddb(trial, filename = "", appName = 'Portal', expName = 'Portal',
 perfdmf = java.lang.System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg"):
  print "download args: ", appName, expName, trial, filename, perfdmf
  dbApi = DatabaseAPI()
  dbApi.initialize(perfdmf, 0)

  #we what to add the trial to the application 'Portal' and the experiment
  #'Portal' we will need to create these if they do not exist.
  appID = 0
  expID = 0
  trialID = 0
  for app in dbApi.getApplicationList():
    if (app.getName() == appName):
      appID = app.getID()
      break

  if (appID == 0):
    print "Could not find Appplication named " + appName
    sys.exit(1)

  for exp in dbApi.getExperimentList():
    if (exp.getName() == expName and exp.getApplicationID() == appID):
      expID = exp.getID()
      break

  if (expID == 0):
    print "Could not find Experiment named " + expName
    sys.exit(1)
  #print expID, appID  
  for tr in dbApi.getTrialList():
    #print tr.getName()
    #print tr.getExperimentID()
    #print tr.getApplicationID()
    if (tr.getName() == trial and tr.getExperimentID() == expID and
    tr.getApplicationID() == appID):
      trialID = tr.getID()
      break
  
  if (trialID == 0):
    print "Could not find Trial named " + trial
    sys.exit(1)

  dbApi.setApplication(appID)
  dbApi.setExperiment(expID)
  dbApi.setTrial(trialID)
  dbDataSource = DBDataSource(dbApi)
  dbDataSource.load()
  if (filename == ""):
    filename = "/tmp/" + trial
  
  iostream = open(filename, "w")
  DataSourceExport.writePacked(dbDataSource,iostream)
  return iostream

def uploaddb(files, appName = "Portal", expName = "Portal"):

  perfdmf = java.lang.System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg";
  dbApi = DatabaseAPI()
  dbApi.initialize(perfdmf, 0)
 
  #we what to add the trial to the application 'Portal' and the experiment
  #'Portal' we will need to create these if they do not exist.
  appID = 0
  expID = 0
  for app in dbApi.getApplicationList():
    if (app.getName() == appName):
      appID = app.getID()
      break

  if (appID == 0):
    #Create App.
    newApp = Application()
    newApp.setName('Portal')
    dbApi.setApplication(newApp)
    appID = dbApi.saveApplication()
  
  for exp in dbApi.getExperimentList():
    if (exp.getName() == expName and exp.getApplicationID() == appID):
      expID = exp.getID()
      break

  if (expID == 0):
    #Create App.
    newExp = Experiment()
    newExp.setName('Portal')
    newExp.setApplicationID(appID)
    dbApi.setExperiment(newExp)
    expID = dbApi.saveExperiment()
  
  i = 0
  print files
  for file in files:
    dataSource = PackedProfileDataSource(java.io.File(file))
    dataSource.load()

    newTrial = Trial()
    newTrial.setDataSource(dataSource)
    newTrial.setName(file.split('/')[-1])
    newTrial.setExperimentID(expID)
    dbApi.uploadTrial(newTrial)


def list_trials(appName = 'Portal', expName = 'Portal'):
  perfdmf = java.lang.System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg";
  dbApi = DatabaseAPI()
  dbApi.initialize(perfdmf, 0)
  
  appID = 0
  expID = 0
  for app in dbApi.getApplicationList():
    if (app.getName() == appName):
      appID = app.getID()
      break

  if (appID == 0):
    print "Could not find Appplication named " + appName
    sys.exit(1)
  
  for exp in dbApi.getExperimentList():
    if (exp.getName() == expName and exp.getApplicationID() == appID):
      expID = exp.getID()
      break

  if (expID == 0):
    print "Could not find Experiment named " + expName
    sys.exit(1)
  
  list = []
  for trial in dbApi.getTrialList():
    if (trial.getExperimentID() == expID):
      list.append(trial.getName())
    
  return list

def list_experiments(appName = 'Portal'):

  perfdmf = java.lang.System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg";
  dbApi = DatabaseAPI()
  dbApi.initialize(perfdmf, 0)
  
  appID = 0
  expID = 0
  for app in dbApi.getApplicationList():
    if (app.getName() == appName):
      appID = app.getID()
      break

  if (appID == 0):
    print "Could not find Appplication named " + appName
    sys.exit(1)
  
  list = []
  for exp in dbApi.getExperimentList():
    if (exp.getApplicationID() == appID):
      list.append(exp.getName())
    
  return list
#print list_trials()
main()
