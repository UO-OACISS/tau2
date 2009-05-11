from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.common import TransformationType
from edu.uoregon.tau.perfexplorer.common import AnalysisType
from edu.uoregon.tau.perfexplorer.glue import Utilities

def findMetric(metrics, findme):
	i = 0;
	for metric in metrics:
		name = metric.getName().upper()
		# print name
		if name.find(findme) > -1:
			return i
		i += 1
	return -1

def getEvents(pe, trial, metricIndex):
	# create a dictionary of events and their performance measurements
	returnEvents = {}
	events = pe.getEventList(trial, metricIndex)
	while events.hasNext():
		event = events.next()
		# print event.getName()
		returnEvents[event.getName()] = event.getMeanSummary()
	return returnEvents

def findMain(events, metric):
	inclusive = 0.0
	main = {}
	for key in events.keys():
		data = events[key]
		if data.getInclusive() > inclusive:
			inclusive = data.getInclusive()
			main["name"] = key
	main["inclusive"] = inclusive
	return main

def mapMetrics(baseMetrics, otherMetrics):
	metricMap = {}
	i = 0
	for metric in baseMetrics:
		baseName = metric.getName().upper()
		j = 0
		for metric in otherMetrics:
			otherName = metric.getName().upper()
			if baseName == otherName:
				metricMap[i] = j
				# print i, " = ", j
				break
			j += 1
		i += 1
	return metricMap

def pairwiseEvent(baseEvents, otherEvents, i, j, filter):
	faster = {}
	slower = {}
	for event in baseEvents.keys():
		# print event
		baseValues = baseEvents[event]
		otherValues = otherEvents[event]
		# print "base: ", baseValues.getExclusive(i), " other: ", otherValues.getExclusive(j)
		diff = baseValues.getExclusive(i) - otherValues.getExclusive(j)
		if diff > 0:
			faster[event] = abs(diff)
		else:
			slower[event] = abs(diff)
	results = {}
	items = faster.items()
	items.sort()
	results["faster"] = items
	items = slower.items()
	items.sort()
	results["slower"] = items
	return results

def mainReport(baseMain, otherMain, baseName, otherName):
	if baseMain["inclusive"] > otherMain["inclusive"]:
		print "\nBaseline trial is relatively slower than second trial.\n"
		percentage = (baseMain["inclusive"] - otherMain["inclusive"]) / otherMain["inclusive"]
		fasterSlower = -1
	elif baseMain["inclusive"] < otherMain["inclusive"]:
		print "\nBaseline trial is relatively faster than second trial.\n"
		percentage = (otherMain["inclusive"] - baseMain["inclusive"]) / baseMain["inclusive"]
		fasterSlower = 1
	else:
		print "\nBaseline trial and second trial have the same execution time."
		fasterSlower = 0
		percentage = 0.0
	# print "\t", baseName, baseMain["name"], ":", baseMain["inclusive"], "seconds\n", 
	# print "\t", otherName, otherMain["name"], ":", otherMain["inclusive"], "seconds\n", 
	print "\t", baseName, ":", baseMain["inclusive"]/1000000, "seconds\n", 
	print "\t", otherName, ":", otherMain["inclusive"]/1000000, "seconds\n", 
	if fasterSlower < 0:
		print "\t Relative Difference: ", percentage*100, "% slower\n"
	elif fasterSlower > 0:
		print "\t Relative Difference: ", percentage*100, "% faster\n"
	else:
		print "\t Relative Difference: ", percentage*100, "%\n"
	return fasterSlower

def showSignificantTimeEvents(diffs, type, significant):
	events = diffs[type]
	x = 0
	shown = 0
	for event in events:
		# don't show more than 10 differences
		if x >= 10:
			break
		# don't show insignificant differences
		if event[1] > 1000000:
			print "\t", event[0], ":", event[1]/1000000, "seconds", type, "than baseline"
			significant.append(event[0])
			shown += 1
		x += 1
	return shown
		

def showSignificantEvents(diffs, type):
	events = diffs[type]
	if type == "faster":
		type = "fewer"
	else:
		type = "more"

	x = 0
	shown = 0
	for event in events:
		# don't show more than 10 differences
		if x >= 10:
			break
		# don't show insignificant differences
		if event[1] > 1000000:
			print "\t", event[0], ":", event[1]/1000000, "million", type, "than baseline"
			shown += 1
		x += 1
	return shown

def DoAnalysis(pe):
	# set the application, experiment, trial
	Utilities.setSession("perfdmf_test")
	pe.setApplication("simple_papi-DSTATIC_MATRIX")
	pe.setExperiment("-O0")
	baseTrial = pe.setTrial("regular")

	# baseMetrics is a List
	baseMetrics = baseTrial.getMetrics().toArray()

	# find the time metric
	baseTime = findMetric(baseMetrics, "TIME")

	# get all the data for each event
	baseEvents = getEvents(pe, baseTrial, baseTime)
	
	# find the main event
	baseMain = findMain(baseEvents, baseTime)
	# print baseMain
	
	# get the other trial
	otherTrial = pe.setTrial("strided")

	# otherMetrics is a List
	otherMetrics = otherTrial.getMetrics().toArray()

	# find the time metric
	metricMap = mapMetrics(baseMetrics, otherMetrics)
	otherTime = metricMap[baseTime]

	# get all the data for each event
	otherEvents = getEvents(pe, otherTrial, otherTime)

	# find the main event
	otherMain = findMain(otherEvents, otherTime)
	# print otherMain
	
	# compare the differences in main
	if baseMain["name"] == otherMain["name"]:
		fasterSlower = mainReport(baseMain, otherMain, baseTrial.getName(), otherTrial.getName())
	else:
		print "Main events do not match: ", baseMain["name"], ", ", otherMain["name"]
		return

	# compare the events for metric i
	diffs = pairwiseEvent(baseEvents, otherEvents, baseTime, otherTime, None)

	# tell the user the significant differences
	significant = []
	print "Significant", baseMetrics[baseTime], "differences between trials:\n"
	shown = showSignificantTimeEvents(diffs, "faster", significant)
	shown += showSignificantTimeEvents(diffs, "slower", significant)
	if shown == 0:
		print "\t None.\n"

	# significant is now populated with the names of events that are significant.
	# iterate through the metrics to locate possible causes for the time difference.

	x = 0
	for metric in baseMetrics:
		if x != baseTime:
			y = metricMap[x]
			diffs = pairwiseEvent(baseEvents, otherEvents, x, y, significant)
			print "\nSignificant", baseMetrics[x], "differences between trials:\n"
			shown = showSignificantEvents(diffs, "faster")
			shown += showSignificantEvents(diffs, "slower")
			if shown == 0:
				print "\t None.\n"
		x += 1
	
print "--------------- JPython test script start ------------"

pe = ScriptFacade()
DoAnalysis(pe)

print "\n"
print "---------------- JPython test script end -------------"

pe.exit()

