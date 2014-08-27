from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *
import json
import gzip

True = 1
False = 0

def loadFile(fileName):
	# load the trial
	files = []
	files.append(fileName)
	if fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PPK, files, False)
	elif fileName.endswith("gprof"):
		input = DataSourceResult(DataSourceResult.GPROF, files, False)
	elif fileName.endswith("xml") or fileName.endswith("xml.gz"):
		input = DataSourceResult(DataSourceResult.SNAP, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

def loadFromFiles():
	inputs = ArrayList()
	inputs.add(loadFile("tauprofile.xml"))
	return inputs.get(0)

def dumpNode(myfile,node,parent,parentPath,result,metric):
	comma = False
	for key, value in node.iteritems():
		currentPath = key
		if parentPath != "":
			currentPath = parentPath + " => " + key
		if comma:
			myfile.write(",")
		"""
		if "[CONTEXT]" in key:
			dumpNode(myfile,value,key,currentPath,result,metric)
			return
		"""
		myfile.write("{\"name\":")
		myfile.write("\"" + key + "\", \"size\":")
		#if value == {}:
		#	myfile.write("\"" + key + "\", \"size\":")
		#else:
		#	myfile.write("\"" + key[0:12] + "\", \"size\":")
		myfile.write(str(result.getInclusive(0, currentPath, metric)))
		if value != {}:
			myfile.write(", \"children\": [")
			dumpNode(myfile,value,key,currentPath,result,metric)
			if parent != "":
				myfile.write(",")
				myfile.write("{\"name\":")
				myfile.write("\"" + key + "\", \"size\":")
				myfile.write(str(result.getExclusive(0, currentPath, metric)))
				myfile.write("}\n")
			myfile.write("]")
		myfile.write("}")
		comma = True

def dumpIcicleNode(myfile,node,parent,parentPath,result,metric):
	comma = False
	for key, value in node.iteritems():
		currentPath = key
		if parentPath != "":
			currentPath = parentPath + " => " + key
		if comma:
			myfile.write(",")
		myfile.write("{\"" + key + "\":")
		if value == {}:
			myfile.write(str(result.getInclusive(0, currentPath, metric)))
		else:
			dumpNode(myfile,value,key,currentPath,result,metric)
			if parent != "":
				myfile.write(",")
				myfile.write("{\"" + key + "\":")
				myfile.write(str(result.getExclusive(0, currentPath, metric)))
				myfile.write("}\n")
		myfile.write("}")
		comma = True

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	t = 0
	raw = loadFromFiles()
	raw.setIgnoreWarnings(True)
	statmaker = BasicStatisticsOperation(raw, False)
	result = statmaker.processData().get(BasicStatisticsOperation.MEAN)
	mainEvent = result.getMainEvent()
	metric = result.getTimeMetric()

	# build the callpath tree
	callpathOperation = ExtractCallpathEventOperation(result)
	callpath = callpathOperation.processData().get(0)
	tree = {}
	for e in callpath.getEvents():
		if "[SAMPLE]" in e:
			continue
		nodes = e.split(" => ")
		current = tree
		for n in nodes:
			if not n in current:
				current[str(n)] = {}
			current = current[str(n)]

	mydata = open("profile.json",'w')
	dumpNode(mydata,tree,"","",result,metric)
	mydata.close()

	mydata2 = open("profile2.json",'w')
	dumpIcicleNode(mydata2,tree,"","",result,metric)
	mydata2.close()

	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
