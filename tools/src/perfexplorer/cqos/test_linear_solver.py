from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
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

	# test the classifier
	inputFields = HashMap()
	inputFields.put("snes", "ls")
	#inputFields.put("procs", "2")
	#inputFields.put("cflini", "0.1")
	#inputFields.put("prandtl", "1")
	#inputFields.put("lidvelocity", "100")
	#inputFields.put("grashof", "100000")
	#inputFields.put("fnorm", "3.266912311248e+03")
	#inputFields.put("snes_fnorm", "3562.78")
	#inputFields.put("cfl", "0.100865")
	for lidvelocity in ['10', '50', '100']:
		for grashof in ['100', '1000', '100000']:
			for cflini in ['0.1', '10', '20']:
				for kspmaxit in ['200', '400', '600']:
					for gridsize in ['32x32', '64x64', '16x16']:
						for pc in ['jacobi', 'ilu', 'bjacobi', 'none', 'sor', 'asm', 'cholesky']:
							for matrixsize in ['64516x64516', '3844x3844', '15876x15876']:
								for ksprtol in ['1.000000e-04', '1.000000e-05']:
									for snesrtol in ['1.000000e-08', '1.000000e-03']:
										inputFields.put("gridsize", gridsize)
										inputFields.put("pc", pc)
										inputFields.put("matrixsize", matrixsize)
										inputFields.put("snesrtol", snesrtol)
										inputFields.put("ksprtol", ksprtol)
										inputFields.put("lidvelocity", lidvelocity)
										inputFields.put("grashof", grashof)
										inputFields.put("cflini", cflini)
										inputFields.put("kspmaxit", kspmaxit)
										className =  classifier.getClass(inputFields)
										confidence = classifier.getConfidence()
										if className == "bcgs":
											bcgs+=1
										if className == "fgmres":
											fgmres+=1
										if className == "gmres":
											gmres+=1
										if className == "tfqmr":
											tfqmr+=1
										if className == "cg":
											cg+=1
	print "bcgs", bcgs
	print "fgmres", fgmres
	print "gmres", gmres
	print "tfqmr", tfqmr
	print "cg", cg



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
