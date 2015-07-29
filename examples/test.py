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

def resultMeaning(result):
    if result == 0 :
        return "PASS"
    elif  result == -1:
        return "NONE"
    else:
        return "FAIL"

def main():
    parser = argparse.ArgumentParser(description='Run all makefiles in each directory')
    parser.add_argument('-f','--filename', default='testResults.txt',
                        help='Name of column format results file.')
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


    directories = filter(os.path.isdir, os.listdir(os.getcwd()))
    makeResults=dict((dirs,{"make":0,"build.sh":0}) for dirs in directories)
      # 0,1 normal, -1 if doesn't exist
    for currentDir in directories:
        os.chdir(currentDir)
        print "*** CD into ", currentDir
        isBuild = os.path.isfile("build.sh")
        isMake = os.path.isfile("Makefile")
        if isMake :
          shell_command("make clean", "make clean not working")
          makeError=shell_command("make","make not working")
          makeResults[currentDir]["make"] = makeError
        else:
            makeResults[currentDir]["make"] = -1
        if isBuild :
          buildError=shell_command("./build.sh","build not working")
          makeResults[currentDir]["build.sh"] = buildError
        else:
            makeResults[currentDir]["build.sh"] = -1

        os.chdir(parentDir)

    # for dir,resultDict in makeResults.iteritems():
    #     for type,result in resultDict.iteritems():
    #         print '%s: %s: %s' % (dir,type,resultMeaning(result))

    target = open(args.filename,'w')

    target.write("#dir,   make,    build.sh \n")
    target.write("\n")

    for dir,resultDict in makeResults.iteritems():
        result=[]
        result.append(dir + " , ")
        for type,code in resultDict.iteritems():
            result.append(resultMeaning(code)+ " , ")
            #print '%s: %s: %s' % (dir,type,resultMeaning(result))}
        result.append("\n")
        target.writelines("%s" % item for item in result)
    target.close()

if __name__ == "__main__":
   main()
