from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.client import PerfExplorerModel
from java.util import *
import sys
import operator

# This flag determines whether or not to include UNWIND events.
# This can be helpful when unresolved samples come from unwound functions.
doInclusive = True
tauData = "tauprofile.xml.gz"

def matchBrackets(tmp, startchar, endchar):
	left = tmp
	right = ""
	tmp = tmp.rstrip()
	if tmp[len(tmp)-1] == endchar:
		e = len(tmp)-1
		s = e-1
		found = False
		stack = 1
		while True:
			if tmp[s] == endchar:
				stack = stack + 1
			elif tmp[s] == startchar:
				stack = stack - 1
			if stack == 0:
				found = True
				break
			if s == 0:
				break
			s = s-1
		if found:
			left = tmp[:s]
			right = tmp[s:]
	return left, right

def stripIntro(ebd):
	tmp = ebd.fullName
	if "[UNWIND]" in tmp:
		ebd.type = "UNWIND"
		# remove [UNWIND] from start
		tmp = tmp[9:]
	if "[SAMPLE]" in tmp:
		ebd.type = "SAMPLE"
		# remove [SAMPLE] from start
		tmp = tmp[9:]
	index = tmp.find("[@] ")
	if index > -1:
		tmp = tmp[index+4:]
	return tmp

def stripSource(tmp, ebd):
	# get the source info
	s = tmp.rfind(" [{")
	m = tmp.rfind("} {")
	e = tmp.rfind("}]")
	if e > -1 and s > -1 and m > -1:
		ebd.file = tmp[s+3:m]
		ebd.line = tmp[m+3:e]
		tmp = tmp[:s]
		#ebd.file, ebd.line
	# check for const at the end
	index = tmp.rfind(" const")
	if index == len(tmp)-6:
		tmp = tmp[:len(tmp)-6]
		ebd.const = True
	return tmp

def parseFullName(ebd):
	tmp = stripIntro(ebd)
	tmp = stripSource(tmp, ebd)

	# check for argument list
	tmp, ebd.arguments = matchBrackets(tmp, "(", ")")

	# check for method templates
	tmp, ebd.methodTemplates = matchBrackets(tmp, "<", ">")

	# get the method
	e = tmp.rfind("::")
	e2 = tmp.rfind(" ")
	if e > -1 and e > e2:
		ebd.method = tmp[e+2:]
		tmp = tmp[:e] # strip the delimiter
	# plain old method, has a return type (can be a class or primitive)
	elif e2 > -1 and e2 > e:
		ebd.method = tmp[e2+1:]
		ebd.returnType = tmp[:e2+1]
		return
	# plain old method, no namespaces or classes
	else:
		# check for destructor
		if tmp[0] == "~":
			ebd.className = tmp[1:]
			ebd.method = tmp
		else:
			ebd.method = tmp
		return
		
	# could be namespace1::...::namespaceN::class1<T1>::...::classN<Tn>
	while True:
		# check for a reference character on return type
		e = tmp.rfind("&")
		if e == len(tmp)-1:
			tmp = tmp[:e]
			ebd.referfence = True
		# check for template on class 
		if len(ebd.classTemplates) > 0:
			tmp, ct = matchBrackets(tmp, "<", ">")
			ebd.classTemplates = ct + ebd.classTemplates
		else:
			tmp, ebd.classTemplates = matchBrackets(tmp, "<", ">")

		# get the class
		e = tmp.rfind("::")
		e2 = tmp.rfind(" ")
		# if we found a :: after a space
		if e > -1 and e > e2:
			ebd.className = tmp[e+2:] + "::" + ebd.className
			ebd.className = ebd.className.rstrip(':')
			ebd.className = ebd.className.lstrip('*')
			ebd.className = ebd.className.lstrip('&')
			tmp = tmp[:e] # strip the delimiter
		# if a space is before the :: (if one exists)
		# plain old method, has a return type (can be a class or primitive)
		elif e2 > -1 and e2 > e:
			ebd.nameSpace = tmp[e2+1:]
			ebd.className = tmp[e2+1:] + "::" + ebd.className
			ebd.className = ebd.className.rstrip(':')
			ebd.className = ebd.className.lstrip('*')
			ebd.className = ebd.className.lstrip('&')
			ebd.returnType = tmp[:e2]
			return
		# nothing left but a namespace/class name
		elif e2 == -1 and e == -1:
			ebd.nameSpace = tmp
			ebd.className = tmp + "::" + ebd.className
			ebd.className = ebd.className.rstrip(':')
			ebd.className = ebd.className.lstrip('*')
			ebd.className = ebd.className.lstrip('&')
			return
		
class EventBreakdown:
	fullName = "" # done
	type = "Function" # done
	nameSpace = ""
	className = ""
	classTemplates = ""
	returnType = "int" #default
	const = False
	reference = False
	method = ""
	methodTemplates = ""
	arguments = "" # done
	file = "" # done
	line = "0" # done
	inclusive = 0
	exclusive = 0
	def __init__(self, fullName):
		self.fullName = fullName
		parseFullName(self)

def getParameters():
	global tauData
	global threshold
	global callsCutoff
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	tmp = parameterMap.get("tauData")
	if tmp != None:
		tauData = tmp
		print "Performance data: " + tauData
	else:
		print "TAU profile data path not specified... using current directory of profile.x.x.x files."

def loadFile(fileName):
	# load the trial
	files = []
	files.append(fileName)
	input = None
	if fileName.endswith("ppk"):
		input = DataSourceResult(DataSourceResult.PPK, files, False)
	elif fileName.endswith("gprof"):
		input = DataSourceResult(DataSourceResult.GPROF, files, False)
	elif fileName.endswith("xml") or fileName.endswith("xml.gz"):
		input = DataSourceResult(DataSourceResult.SNAP, files, False)
	else:
		input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False)
	return input

def dumpResult(ebd):
	print ebd.fullName
	print "type: " + ebd.type
	print "returnType: " + ebd.returnType
	if ebd.reference:
		print "reference: True"
	else:
		print "reference: False"
	print "namespace: " + ebd.nameSpace
	print "className: " + ebd.className
	print "classTemplates: " + ebd.classTemplates
	print "method: " + ebd.method
	print "methodTemplate: " + ebd.methodTemplates
	print "arguments: " + ebd.arguments
	if ebd.const:
		print "const: True"
	else:
		print "const: False"
	print "file: " + ebd.file
	print "line: " + ebd.line
	print "inclusive: ", ebd.inclusive
	print "exclusive: ", ebd.exclusive

def checkParents(ebd, full, ebds):
	# iterate over the callpath events
	for e in full.getEvents():
		# if this event ends with the current event as a leaf
		if e.endswith(" => " + ebd.fullName):
			# get all the parents in this callpath
			for p in e.split(" => "):
				if p in ebds:
					pebd = ebds[p]
					# if this event is not the same, but the parent has the same class,
					# don't include the value for this event
					if p != ebd.fullName and pebd.className == ebd.className:
						return 0
	return ebd.inclusive

def showChildren(ebds, className, full):

	# build a dictionary of values. We need to do this, because 
	# methods can have multiple values that need to be aggregated.
	# For example, samples can be at multiple lines of a method.
	# Also, Multiple methods could have different templated instances.

	methods = dict()
	for event,ebd in ebds.items():
		if ebd.className == className:
			value = 0.0
			if ebd.type == "UNWIND":
				value = checkParents(ebd,full,ebds)
			elif ebd.type == "SAMPLE":
				value = ebd.inclusive # should be the same as exclusive, but...
			else:
				value = ebd.exclusive
			if ebd.method in methods:
				methods[ebd.method] = methods[ebd.method] + value
			else:
				methods[ebd.method] = value
	
	# Iterate over the methods in this class, sorted by value

	othervalue = 0
	showmax=5 # set to 0 to show all methods
	for m in sorted(methods, key=methods.get, reverse=True):
		if showmax > 0:
			print "\tMethod '%s' : %f" % (m,methods[m]/1000000)
		else:
			othervalue = othervalue + methods[m]
		showmax = showmax - 1
	if showmax < 0:
		print "\tAll other methods : %f" % (othervalue/1000000)

def main():
	global filename
	global tauData
	global doInclusive
	print "--------------- JPython test script start ------------"
	# get the parameters
	getParameters()
	# load the data
	result = loadFile(tauData)
	result.setIgnoreWarnings(True)

	# set the metric, type we are interested in
	metric = result.getTimeMetric()
	if metric == None:
		metrics = result.getMetrics().toArray()
		metric = metrics[0]
	type = result.EXCLUSIVE
	mainEvent = result.getMainEvent()
	
	# then, extract those events from the actual data
	print "Extracting non-callpath data..."
	flatten = ExtractNonCallpathEventOperation(result)
	flat = flatten.processData().get(0)

	print "Computing statistics..."
	statmaker = BasicStatisticsOperation(flat, False)
	statmaker.setIncludeNull(False)
	stats = statmaker.processData().get(BasicStatisticsOperation.MEAN)

	# get the callpath events
	print "Extracting callpath data..."
	fullen = ExtractCallpathEventOperation(result)
	full = fullen.processData().get(0)

	ebds = dict()
	for event in flat.getEvents():
		if event.startswith("[SAMPLE] "):
			if "UNRESOLVED" not in event:
				ebd = EventBreakdown(event)
				ebd.inclusive = stats.getInclusive(0,event,metric)
				ebd.exclusive = stats.getExclusive(0,event,metric)
				ebds[event] = ebd
		elif event.startswith("[UNWIND] "):
			if doInclusive and "UNRESOLVED" not in event:
				ebd = EventBreakdown(event)
				ebd.inclusive = stats.getInclusive(0,event,metric)
				ebd.exclusive = stats.getExclusive(0,event,metric)
				ebds[event] = ebd
		elif event != ".TAU application":
			ebd = EventBreakdown(event)
			ebd.inclusive = stats.getInclusive(0,event,metric)
			ebd.exclusive = stats.getExclusive(0,event,metric)
			ebds[event] = ebd

	classes = dict()
	for event,ebd in ebds.items():
		value = 0
		if ebd.type == "UNWIND":
			value = checkParents(ebd,full,ebds)
		else:
			value = ebd.exclusive
		if ebd.className in classes:
			classes[ebd.className] = classes[ebd.className] + value
		else:
			classes[ebd.className] = value

	othervalue = 0
	showmax=10 # set to 0 to show all classes
	for c in sorted(classes, key=classes.get, reverse=True):
		#if len(c) > 0:
		if showmax > 0:
			print "\nClass '%s' : %f" % (c,classes[c]/1000000)
			showmax = showmax - 1
			showChildren(ebds,c,full)
		else:
			othervalue = othervalue + classes[c]
	# get the application total from the original profile, thread 0. It is the true application main.
	print "\nAll other classes : %f, application total : %f" % (othervalue/1000000, result.getInclusive(0,mainEvent,metric)/1000000)
	print "(inclusive aggregation of unwound samples and means without NULLs can add up to more than application total)"
	print "\nMetric:", metric, "/ 1,000,000"

	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()

