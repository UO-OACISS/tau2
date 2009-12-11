from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

parameterMap = None
tauData = ""
threshold = 10
functions = "function-list.txt"
gprof = False
parent = None

def getParameters():
	global parameterMap
	global tauData
	global functions
	global threshold
	global gprof
	global parent
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	#for key in keys:
		#print key, parameterMap.get(key)
	tmp = parameterMap.get("tauData")
	if tmp != None:
		tauData = tmp
		print "Performance data: " + tauData
	else:
		print "TAU profile data path not specified... using current directory of profile.x.x.x files."
	tmp = parameterMap.get("fileName")
	if tmp != None:
		functions = tmp
		print "Output filename: " + functions
	else:
		print "Output filename not specified... using " + functions
	tmp = parameterMap.get("threshold")
	if tmp != None:
		threshold = int(tmp)
		print "Threshold: " + str(threshold)
	else:
		print "Threshold not specified... using " + str(threshold)
	tmp = parameterMap.get("parent")
	if tmp != None:
		parent = tmp
		print "Parent function: " + functions
	else:
		print "Parent not specified... exiting. "
		System.exit(1)

def loadFile(fileName):
	global gprof
	# load the trial
	files = []
	files.append(fileName)
	input = None
	if fileName.endswith("gprof.out"):
		input = DataSourceResult(DataSourceResult.GPROF, files, True)
		gprof = True
	elif fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PPK, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

def main():
	global tauData
	global functions
	global threshold
	global gprof

	print "--------------- JPython test script start ------------"

	# get the parameters
	getParameters()

	# load the data
	input = loadFile(tauData)

	# extract the non-callpath data
	print "Extracting children of", parent, "..."
	input.setIgnoreWarnings(True)
	extractor = ExtractChildrenOperation(input, parent)
	extracted = extractor.processData().get(0)

	# extract computation code (remove MPI)
	myEvents = ArrayList()
	print "Filtering out MPI calls..."
	for event in extracted.getEvents():
		if not event.startswith("MPI_"):
			myEvents.add(event)
	extractor = ExtractEventOperation(extracted, myEvents)
	extracted = extractor.processData().get(0)

	# generate statistics
	print "Generating stats..."
	doStats = BasicStatisticsOperation(extracted, False) 
	mean = doStats.processData().get(BasicStatisticsOperation.MEAN)
	doStats = BasicStatisticsOperation(input, False) 
	fullMean = doStats.processData().get(BasicStatisticsOperation.MEAN)
	meanTotal = fullMean.getInclusive(0,fullMean.getMainEvent(),fullMean.getTimeMetric())

	# get the top X events
	print "Extracting top events by INCLUSIVE value..."
	mean.setIgnoreWarnings(True)
	topper = TopXEvents(mean, mean.getTimeMetric(), AbstractResult.INCLUSIVE, threshold) 
	topped = topper.processData().get(0)

	# put the top X events names in a file
	myFile = open(functions, 'w')
	for event in topped.getEvents():
		shortEvent = event
		# fix gprof names
		if gprof:
			shortEvent = shortEvent.upper()
			if shortEvent.startswith("__MODULE"):
				shortEvent = shortEvent.replace("__MODULE","MODULE")
				shortEvent = shortEvent.replace("_NMOD_","::")
				shortEvent = shortEvent.replace("_STUB_IN_" + parent,"")
				shortEvent = shortEvent.replace("_IN_" + parent,"")
			if shortEvent.startswith("*__MODULE"):
				shortEvent = shortEvent.replace("*__MODULE","MODULE")
				shortEvent = shortEvent.replace("_NMOD_","::")
				shortEvent = shortEvent.replace("_STUB_IN_" + parent,"")
				shortEvent = shortEvent.replace("_IN_" + parent,"")
		# fix TAU names
		else:
			shortEvent = Utilities.shortenEventName(event)
		print "%00.2f%%\t %s" % (topped.getInclusive(0,event,topped.getTimeMetric()) / meanTotal * 100.0, event)
		myFile.write(shortEvent + "\n")
	myFile.close()

	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
