from glue import *
from client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfdmf import Experiment
from java.util import *
import sys
import time

True = 1
False = 0
config = "cqos"
inApp = "./ex27_2"
inExp = "0x0"
inTrial = ""
parameterMap = None
fileName = "/tmp/classifier.serialized"

def getParameters():
	global parameterMap
	global config
	global inApp
	global inExp
	global fileName
	print "getting parameters..."
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	for key in keys:
		print key, parameterMap.get(key)
	config = parameterMap.get("config")
	inApp = parameterMap.get("app")
	inExp = parameterMap.get("exp")
	fileName = parameterMap.get("fileName")
	print "...done."

def testClassifier(classifier):
	bcgs = 0
	fgmres = 0
	gmres = 0
	tfqmr = 0
	cg = 0

    # {lidvelocity=100, snesrtol=1.0e-3, gridsize=32x32, cflini=1.0e-1, grashof=1.e5}:

	# test the classifier
	inputFields = HashMap()
	inputFields.put("snes", "ls")
	for lidvelocity in ['10', '50', '100']:
		for grashof in ['100', '1000', '100000']:
			for cflini in ['0.1', '10', '20']:
				for gridsize in ['32x32', '64x64', '16x16']:
					for snesrtol in ['1.000000e-08', '1.000000e-03']:
						inputFields.put("gridsize", gridsize)
						inputFields.put("snesrtol", snesrtol)
						inputFields.put("lidvelocity", lidvelocity)
						inputFields.put("grashof", grashof)
						inputFields.put("cflini", cflini)
						className =  classifier.getClass(inputFields)
						confidence = classifier.getConfidence()
						if className == "bcgs":
							print inputFields
							bcgs+=1
						if className == "fgmres":
							fgmres+=1
						if className == "gmres":
							gmres+=1
						if className == "tfqmr":
							tfqmr+=1

	print "bcgs", bcgs
	print "fgmres", fgmres
	print "gmres", gmres
	print "tfqmr", tfqmr

print "--------------- JPython test script start ------------"

getParameters()

print "TESTING J48"
classifier = CQoSClassifierOperation.readClassifier(fileName + ".j48")
testClassifier(classifier)

print "TESTING MP"
classifier = CQoSClassifierOperation.readClassifier(fileName + ".mp")
testClassifier(classifier)

print "TESTING NB"
classifier = CQoSClassifierOperation.readClassifier(fileName + ".nb")
testClassifier(classifier)

print "TESTING SVM"
classifier = CQoSClassifierOperation.readClassifier(fileName + ".svm")
testClassifier(classifier)

print "---------------- JPython test script end -------------"
