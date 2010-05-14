from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.cqos import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from java.util import *
import sys
import time

classifierFilename = "classifier.class"
inputData = "trainingData.csv"
mode = "build"

def getParameters():
	global inputData
	global classifierFilename
	global mode
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	for key in keys:
		print key, parameterMap.get(key)
	mode = parameterMap.get("mode")
	inputData = parameterMap.get("inputData")
	classifierFilename = parameterMap.get("classifierFilename")

def buildClassifier():
	global inputData
	global classifierFilename
	start = time.clock()
	wrapper = WekaClassifierWrapper (inputData, "benchmark")
	#wrapper.setClassifierType(WekaClassifierWrapper.MULTILAYER_PERCEPTRON)
	#wrapper.setClassifierType(WekaClassifierWrapper.SUPPORT_VECTOR_MACHINE)
	wrapper.setClassifierType(WekaClassifierWrapper.J48)
	wrapper.buildClassifier()
	end = time.clock()
	print end - start, " seconds to build classifier"
	start = time.clock()
	print wrapper.crossValidateModel(10);
	end = time.clock()
	print end - start, " seconds to validate classifier"
	WekaClassifierWrapper.writeClassifier(classifierFilename, wrapper)
	print classifierFilename, "created."

def testClassifier():
	global inputData
	global classifierFilename
	classifier = WekaClassifierWrapper.readClassifier(classifierFilename)
	classifier.testClassifier(inputData)
	#for className in classifier.testClassifier(inputData):
	#	print className

def main(argv):
	print "--------------- JPython test script start ------------"

	getParameters()

	if mode == "build":
		print "building classifier"
		buildClassifier()
	else:
		print "using classifier"
		testClassifier()

	print "...done."
	
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main(sys.argv[1:])

