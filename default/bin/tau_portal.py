#!/usr/bin/env python
import httplib, urllib, fileinput, sys, getpass, platform
from optparse import OptionParser
#import perfdmf and portal modules
import portal

#for version 2.5 and greater use hashlib
version = platform.python_version().split('.')
if ((version[1] in ['0','1','2','3','4']) and (version[0] == "2")):
  import sha
else: 
  import hashlib

def usage():
  return '''
  tau_portal [(-h, --help)] [Command] [Options] [Argument]
  
  Commands are:           Options are:                                    Arguments:
  upload (up)             -u,--username, -p,--password, -w,--workspace    ppk file, ...
                          -e,--experiment
  download (down)         -u,--username, -p,--password, -w,--workspace    trial name
                          -e,--experiment
  synchronize (sync)      -u,--username, -p,--password, -w,--workspace,  
                          -a,--application, -e,--experiment, 
                          --portal-only, --perfdmf-only
  list_workspaces (work)  -u,--username, -p,--password  
  list_trials (trial)     -u,--username, -p,--password, -w,--workspace  
                          -e,--experiment
  '''

def main():
#  try: 
#    opts, args = getopt.getopt(sys.argv[1:], "hu:p:w:a:e:", ["help", "user=",
#    "password=", "workspace=", "application=", "experiment="])
#
#  except getopt.GetoptError:
#    #print usage
#    usage()
#    sys.exit(2)

  if len(sys.argv[1:]) < 1:
    print usage()
    sys.exit(-1)
  

  parser = OptionParser(usage()) 
  parser.add_option('-u','--username', help="TAU portal username", default="")
  parser.add_option('-p','--password', help="TAU portal password", default="")
  parser.add_option('-w','--workspace', help="TAU portal workspace", default="")
  parser.add_option('-a','--application', help="PerfDMF application",  
    default=None) 
  parser.add_option('-e','--experiment', help="PerfDMF experiment", 
    default=None)
  parser.add_option('-d','--host', help="alternative host (debugging)",
  default="tau.nic.uoregon.edu")
  parser.add_option('--portal-only', action="store_false", dest="transfer_to_perfdmf", 
  default=True, help="only transfer trials to the TAU portal")
  parser.add_option('--perfdmf-only', action="store_false", dest="transfer_to_portal", 
  default=True, help="only transfer trials to the PerfDMF database")

  options, args = parser.parse_args(sys.argv[2:])
  #print sys.argv[1] 
  if ( sys.argv[1] in ['h', '-h', "--help", "-help"]):
    parser.parse_args(sys.argv[1:])

  if (options.username == ""):
    print "TAU Portal Username: ",
    options.username = sys.stdin.readline().strip()
  if (options.password == ""):
    options.password = getpass.getpass("TAU Portal Password: ")
  if (options.workspace == "" and not sys.argv[1] in ["list_workspaces", "work"]):
    print "TAU Portal Workspace: ",
    options.workspace = sys.stdin.readline().strip()
  
  #print options, args
  
  if (sys.argv[1] in ["upload", "up"]):
    trial_list = []
    for trial in args:
      trial_list.append(open(trial, 'r'))
    print portal.upload(options.username, options.password, options.workspace,
    options.experiment, trial_list, options.host)
    #print "upload"
  elif (sys.argv[1] in ["download", "down"]):
    file = portal.download(options.username, options.password,
    options.workspace, options.experiment, args[0], options.host)
    if (file.startswith("TAU Portal")):
      print file
    else:  
      print "TAU PORTAL download is a success.\n"
      name = args[0] + ".ppk"
      filewriter = open(name, 'w')
      filewriter.write(file)
      filewriter.close()
    #print "download"
  elif (sys.argv[1] in ["synchronize", "sync"]):
    if (options.application is None):
      options.application = "Portal"
    portal.sync(options.username, options.password, options.workspace, 
    options.application, options.experiment, options.transfer_to_perfdmf,
    options.transfer_to_portal, options.host)
    #print "sync"
  elif (sys.argv[1] in ["list_workspaces", "work"]):
    for workspace in portal.get_workspaces(options.username, options.password):
      print workspace + ",",
  elif (sys.argv[1] in ["list_trials", "trial"]):
    trials = portal.get_trials(options.username, options.password,
    options.workspace, options.experiment, options.host)
    if (not trials is None):
      print ", ".join(trials)
    #print portal.get_trials(options.username, options.password,
    #options.workspace, options.host) 
  else:
    print "Command : " + sys.argv[1] + " unknown."

main()
