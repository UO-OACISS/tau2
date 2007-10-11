#!/usr/bin/env python
"""TAU Portal Python Module 
   the main defintation is for command line usage.
   the download method is used for importation (like in JPython)
"""

import httplib, urllib, fileinput, sys, getopt, getpass, platform, os, thread, threading, time
#for version 2.5 and greater use hashlib
version = platform.python_version().split('.')
if ((version[1] in ['0','1','2','3','4']) and (version[0] == "2")):
  import sha
else: 
  import hashlib


"""
  sync take a username, password (cleartext), workspace, and a list of trial
  names [name,...].

  return a string stating either a success or the cause of any errors.
"""
def sync(username, password, workspace, application, experiment,
transfer_to_perfdmf, transfer_to_portal, host = "tau.nic.uoregon.edu"):
  if (experiment is None):
    for exp in os.popen("perfdmfdb.py list_experiments '" + application + "'").readlines():
      sync_by_experiment(username, password, workspace, application, exp.strip(),
      transfer_to_perfdmf, transfer_to_portal, host) 
  else:
    sync_by_experiment(username, password, workspace, application, experiment,
    transfer_to_perfdmf, transfer_to_portal, host)     

def sync_by_experiment(username, password, workspace, application, experiment,
transfer_to_perfdmf, transfer_to_portal, host = "tau.nic.uoregon.edu"):  
  #global variables for thread communication.
  print "Transfering Experiment: ", experiment

  global threads_running, threads_running_lock
  threads_running_lock = thread.allocate_lock()

  def list_diff(list1, list2):
    diff = []
    for l in list1:
      if not l in list2:
        diff.append(l)
    return diff
  
  portal_trials = get_trials(username, password, workspace, experiment, host,
    True)
  if (portal_trials is None ):
    return "Error authenticating with TAU Portal, check username, password, and workspace."
  perfdmf_trials = []
  for trial in os.popen("perfdmfdb.py list '" + application + "' '" + experiment
  + "'").readlines():
    perfdmf_trials.append(trial.strip())
  #print perfdmf_trials
  #print "Portal:   ", portal_trials
  #print "Database:   ", perfdmf_trials
  to_perfdmf = list_diff(portal_trials, perfdmf_trials)
  to_portal = list_diff(perfdmf_trials, portal_trials)

  #print "to Portal: ", to_portal
  #print "to Database: ", to_perfdmf

  #move files from database to portal
  if (transfer_to_portal):
    def db2portal(username, password, workspace, application, experiment,
    to_portal, host):
      global threads_running, threads_running_lock
      #print "after: ", to_portal
      print len(to_portal), "trial(s) to be upload to the TAU Portal."
      to_portal_list = []
      #print to_portal
      for trial in to_portal:
        os.popen("perfdmfdb.py download '" + application + "' '" + experiment +
        "' " + trial)
        to_portal_list.append(open("/tmp/" + trial, 'r')) 
      upload(username, password, workspace, experiment, to_portal_list, host)

      threads_running_lock.acquire()
      threads_running = False
      threads_running_lock.release()
      #print "thread complete"

    #dispatch a thread to use the portal.
    #print "before:", to_portal
    threads_running_lock.acquire()
    threads_running = True
    threads_running_lock.release()
    thread.start_new_thread(db2portal, (username, password, workspace,
    application, experiment, to_portal, host))
    #db2portal(username, password, workspace, application, experiment, to_portal)
    
 
  #move files from portal to database
  if (transfer_to_perfdmf):
    print len(to_perfdmf), "trial(s) to be upload to the PerfDMF Database."
    to_perfdmf_list = ""
    for trial in to_perfdmf:
      file = download(username, password, workspace, experiment, trial, host)
      #print file
      name = "/tmp/" + trial
      filewriter = open(name, 'w')
      filewriter.write(file)
      to_perfdmf_list += " " + name
      filewriter.close()

    #print to_perfdmf_list
    #print "perfdmfdb.py upload " + application + " " + experiment + " " + to_perfdmf_list 
    os.popen("perfdmfdb.py upload '" + application + "' '" + experiment + "' " + to_perfdmf_list) 

  #wait for threads to finnish
  while threads_running: pass

def get_trials(username, password, workspace, experiment, 
  host = "tau.nic.uoregon.edu", ignore_unfound_experiment = False):

  params = {}
  #find all trials
  #hash password
  #for version geater than 2.5 use hashlib
  if ((version[1] in ['0','1','2','3','4']) and (version[0] == "2")):
    password = sha.new(password).hexdigest()
  else:
    password = hashlib.sha1(password).hexdigest()
    #print "using hashlib"
  
  
  #print "username: " + username 
  params['username'] = username
  #print "password: " + password 
  params['password'] = password
  #print "workspace: " + workspace
  params['workspace'] = workspace
  #print params
  if (not experiment is None):
    params['experiment'] = experiment  


  encoded_params = urllib.urlencode(params)

  #{'simple example': file2.read(), 'username': 'scottb',
  #'password': 'e548fdb1dded95c50e59b08106a5fe01397b4053', 'workspace': 'working database'})

  #form http request 
  if (host == "tau.nic.uoregon.edu"):
    connection = httplib.HTTPSConnection("tau.nic.uoregon.edu")
  else:
    connection = httplib.HTTPConnection(host)
  header = {"Content-type": "application/x-www-form-urlencoded", "Content-length":
  ("%d" % len(encoded_params)), 'Accept': 'text/plain', 'Host':
  host}
  connection.request("POST", "/trial/list_trials", encoded_params, header)
  response = connection.getresponse()
  #filewriter = open(trial + ".ppk", "w")
  #print response.status, response.reason
  #filename = response.readline()
  list = response.read()
  if (list == "PORTAL UPLOAD ERROR: Unknown experiment.\n" and
    ignore_unfound_experiment):
    return []
  elif (list.startswith("PORTAL UPLOAD")):
    print list
    return None
  list = list.split(',')
  final_list = []
  for l in list:
    final_list.append(l.strip())
  response.close()
  #filewriter.close()
  return final_list

def get_workspaces(username, password, host = "tau.nic.uoregon.edu"):

  params = {}
  #find all trials
  #hash password
  #for version geater than 2.5 use hashlib
  if ((version[1] in ['0','1','2','3','4']) and (version[0] == "2")):
    password = sha.new(password).hexdigest()
  else:
    password = hashlib.sha1(password).hexdigest()
    #print "using hashlib"
  
  
  #print "username: " + username 
  params['username'] = username
  #print "password: " + password 
  params['password'] = password

  encoded_params = urllib.urlencode(params)

  #{'simple example': file2.read(), 'username': 'scottb',
  #'password': 'e548fdb1dded95c50e59b08106a5fe01397b4053', 'workspace': 'working database'})

  #form http request 
  if (host == "tau.nic.uoregon.edu"):
    connection = httplib.HTTPSConnection("tau.nic.uoregon.edu")
  else:
    connection = httplib.HTTPConnection(host)
  header = {"Content-type": "application/x-www-form-urlencoded", "Content-length":
  ("%d" % len(encoded_params)), 'Accept': 'text/plain', 'Host':
  host}
  connection.request("POST", "/trial/list_workspaces", encoded_params, header)
  response = connection.getresponse()
  #filewriter = open(trial + ".ppk", "w")
  #print response.status, response.reason
  #filename = response.readline()
  list = response.read()
  list = list.split(',')[:-1]
  final_list = []
  for l in list:
    final_list.append(l.strip())
  response.close()
  #filewriter.close()
  return final_list

def upload(username, password, workspace, experiment, iostreams, host =
"tau.nic.uoregon.edu"):

  params = {}
  for io in iostreams:
    stream = io.read()
    params[io.name.split('/')[-1]] = stream

  #hash password
  #for version geater than 2.5 use hashlib
  if ((version[1] in ['0','1','2','3','4']) and (version[0] == "2")):
    password = sha.new(password).hexdigest()
  else:
    password = hashlib.sha1(password).hexdigest()
    #print "using hashlib"

  #print "username: " + username 
  params['username'] = username
  #print "password: " + password 
  params['password'] = password
  #print "workspace: " + workspace
  params['workspace'] = workspace
  #print params
  if (not experiment is None):
    params['experiment'] = experiment  
  encoded_params = urllib.urlencode(params)

  #{'simple example': file2.read(), 'username': 'scottb',
  #'password': 'e548fdb1dded95c50e59b08106a5fe01397b4053', 'workspace': 'working database'})

  #form http request 
  connection = httplib.HTTPConnection(host)
  header = {"Content-type": "application/x-www-form-urlencoded",
  'Accept': 'text/plain', 'Host':
  host}
  connection.request("POST", "/trial/batch_upload", encoded_params, header)
  response = connection.getresponse()
  return response.read()

"""
  upload take a username, password (cleartext), workspace, and a list of trial
  names [name,...].

  return a string stating either a success or the cause of any errors.
"""
def download(username, password, workspace, experiment, trial, 
  host = "tau.nic.uoregon.edu"):
  params = {}
  
  #files will form basis for the http parameters
  params[trial] = ''
  #hash password
  #for version geater than 2.5 use hashlib
  if ((version[1] in ['0','1','2','3','4']) and (version[0] == "2")):
    password = sha.new(password).hexdigest()
  else:
    password = hashlib.sha1(password).hexdigest()
    #print "using hashlib"
  
  
  #print "username: " + username 
  params['username'] = username
  #print "password: " + password 
  params['password'] = password
  #print "workspace: " + workspace
  params['workspace'] = workspace
  #print params
  if (not experiment is None):
    params['experiment'] = experiment  
  encoded_params = urllib.urlencode(params)

  #{'simple example': file2.read(), 'username': 'scottb',
  #'password': 'e548fdb1dded95c50e59b08106a5fe01397b4053', 'workspace': 'working database'})

  #form http request 
  if (host == "tau.nic.uoregon.edu"):
    connection = httplib.HTTPSConnection("tau.nic.uoregon.edu")
  else:
    connection = httplib.HTTPConnection(host)
  header = {"Content-type": "application/x-www-form-urlencoded", "Content-length":
  ("%d" % len(encoded_params)), 'Accept': 'text/plain', 'Host':
  host}
  connection.request("POST", "/trial/batch_download", encoded_params, header)
  response = connection.getresponse()
  filewriter = open(trial + ".ppk", "w")
  #print response.status, response.reason
  #filename = response.readline()
  file = response.read()
  if (file.startswith("PORTAL")):
    print file
    return file
  else:
    return file
  #filewriter.write(file)
  response.close()
  filewriter.close()

