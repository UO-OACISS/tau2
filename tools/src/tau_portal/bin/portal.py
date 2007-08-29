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
def sync(username, password, workspace, application, experiment, transfer_to_perfdmf, transfer_to_portal):
  #global variables for thread communication.

  global threads_running, threads_running_lock
  threads_running_lock = thread.allocate_lock()

  def list_diff(list1, list2):
    diff = []
    for l in list1:
      if not l in list2:
        diff.append(l)
    return diff
  
  portal_trials = get_trials(username, password, workspace)
  if (portal_trials is None ):
    return "Error authenticating with TAU Portal, check username, password, and workspace."
  perfdmf_trials = []
  for trial in os.popen("perfdmfdb.py list " + application + " " + experiment).readlines():
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
    def db2portal(username, password, workspace, application, experiment, to_portal):
      global threads_running, threads_running_lock
      #print "after: ", to_portal
      print len(to_portal), "trial(s) to be upload to the TAU Portal."
      to_portal_list = []
      for trial in to_portal:
        os.popen("perfdmfdb.py download " + application + " " + experiment + " " + trial)
        to_portal_list.append(open("/tmp/" + trial, 'r')) 
      upload(username, password, workspace, to_portal_list)

      threads_running_lock.acquire()
      threads_running = False
      threads_running_lock.release()
      #print "thread complete"

    #dispatch a thread to use the portal.
    #print "before:", to_portal
    threads_running_lock.acquire()
    threads_running = True
    threads_running_lock.release()
    thread.start_new_thread(db2portal, (username, password, workspace, application, experiment, to_portal))
    #db2portal(username, password, workspace, application, experiment, to_portal)
    
 
  #move files from portal to database
  if (transfer_to_perfdmf):
    print len(to_perfdmf), "trial(s) to be upload to the PerfDMF Database."
    to_perfdmf_list = ""
    for trial in to_perfdmf:
      file = download(username, password, workspace, trial)
      #print file
      name = "/tmp/" + trial
      filewriter = open(name, 'w')
      filewriter.write(file)
      to_perfdmf_list += " " + name
      filewriter.close()

    #print to_perfdmf_list
    #print "perfdmfdb.py upload " + application + " " + experiment + " " + to_perfdmf_list 
    os.popen("perfdmfdb.py upload " + application + " " + experiment + " " + to_perfdmf_list) 

  #wait for threads to finnish
  while threads_running: pass

def get_trials(username, password, workspace):

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


  encoded_params = urllib.urlencode(params)

  #{'simple example': file2.read(), 'username': 'scottb',
  #'password': 'e548fdb1dded95c50e59b08106a5fe01397b4053', 'workspace': 'working database'})

  #form http request 
  connection = httplib.HTTPSConnection("tau.nic.uoregon.edu")
  header = {"Content-type": "application/x-www-form-urlencoded", "Content-length":
  ("%d" % len(encoded_params)), 'Accept': 'text/plain', 'Host':
  'tau.nic.uoregon.edu'}
  connection.request("POST", "/trial/list_trials", encoded_params, header)
  response = connection.getresponse()
  #filewriter = open(trial + ".ppk", "w")
  #print response.status, response.reason
  #filename = response.readline()
  list = response.read()
  if (list.startswith("PORTAL UPLOAD")):
    print list
    return None
  list = list.split(',')[:-1]
  final_list = []
  for l in list:
    final_list.append(l.strip())
  response.close()
  #filewriter.close()
  return final_list

def get_workspaces(username, password):

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
  connection = httplib.HTTPSConnection("tau.nic.uoregon.edu")
  header = {"Content-type": "application/x-www-form-urlencoded", "Content-length":
  ("%d" % len(encoded_params)), 'Accept': 'text/plain', 'Host':
  'tau.nic.uoregon.edu'}
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

def upload(username, password, workspace, iostreams):

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
  encoded_params = urllib.urlencode(params)

  #{'simple example': file2.read(), 'username': 'scottb',
  #'password': 'e548fdb1dded95c50e59b08106a5fe01397b4053', 'workspace': 'working database'})

  #form http request 
  connection = httplib.HTTPConnection("tau.nic.uoregon.edu:80")
  header = {"Content-type": "application/x-www-form-urlencoded",
  'Accept': 'text/plain', 'Host':
  'tau.nic.uoregon.edu'}
  connection.request("POST", "/trial/batch_upload", encoded_params, header)
  response = connection.getresponse()
  return response.read()

"""
  upload take a username, password (cleartext), workspace, and a list of trial
  names [name,...].

  return a string stating either a success or the cause of any errors.
"""
def download(username, password, workspace, trial):
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


  encoded_params = urllib.urlencode(params)

  #{'simple example': file2.read(), 'username': 'scottb',
  #'password': 'e548fdb1dded95c50e59b08106a5fe01397b4053', 'workspace': 'working database'})

  #form http request 
  connection = httplib.HTTPSConnection("tau.nic.uoregon.edu")
  header = {"Content-type": "application/x-www-form-urlencoded", "Content-length":
  ("%d" % len(encoded_params)), 'Accept': 'text/plain', 'Host':
  'tau.nic.uoregon.edu'}
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

