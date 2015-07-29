#! /usr/bin/env python
import subprocess
import os,sys,getopt,argparse


def shell_command(command,errorMessage):
#command initiated where this script is ran
  try:
    print command
    subprocess.check_call(command, stderr=subprocess.STDOUT, shell=True)
    errorStatus=0
  except :
    print errorMessage
    pass
    errorStatus=1
  return errorStatus

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def dumpclean(obj):
    if type(obj) == dict:
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                print k
                dumpclean(v)
            else:
                print '%s : %s' % (k, v)
    elif type(obj) == list:
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                print v
    else:
        print obj

def main():
    parser = argparse.ArgumentParser(description='Run all makefiles in each directory')
 # 20   parser.add_argument('-f','--figurename', default=None,
 # 21                       help='Name of histogram figure.')
 # 22
 # 23   parser.add_argument('-t','--figuretitle', default="NE=3, 1 mpi rank at full device thread use" ,
 # 24                       help='Title on histogram figure.')
 # 25
 # 26   parser.add_argument('-g','--grouptime', default="prim_run",
 # 27                       help='Group timing desired.')
 # 28
 # 29   parser.add_argument('-l','--listRundir', dest='listRundir', default=None,
 # 30                       help='list of Rundirs to use for a line plot. Comma seperated')
    args = parser.parse_args()
    parentDir=os.getcwd()

#get list of directories

    directories = filter(os.path.isdir, os.listdir(os.getcwd()))
    makeResults=dict((dirs,[{"make":0},{"build.sh":0}]) for dirs in directories)
      # 0,1 normal, -1 if doesn't exist
    for currentDir in directories:
#cd into dirs and run makefile
        os.chdir(currentDir)
        isBuild = os.path.isfile("build.sh")
        isMake = os.path.isfile("Makefile")
        if isMake :
          makeError=shell_command("make","make not working")
          makeResults[currentDir]["make"] = makeError
        else:
            makeResults[currentDir]["make"] = -1
        if isbuild :
          buildError=shell_command("build","build not working")
          makeResults[currentDir]["build"] = buildError
        else:
            makeResults[currentDir]["build"] = -1

        os.chdir(parentDir)

    dumpclean(makeResults)

if __name__ == "__main__":
   main()
