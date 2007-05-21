from client import ScriptFacade
from common import TransformationType
from common import AnalysisType
from common import EngineType

False = 0
True = 1
million = 1000000
billion = 1000000000

def findMetric(metrics, findme):
	i = 0;
	if findme == "TIME":
		# look for the usual "Time" metric from TAU
		for metric in metrics:
			name = metric.getName().upper()
			# print name
			if name == findme:
				return i
			i += 1
		i = 0
		# look for the usual "WALL_CLOCK_TIME" from PAPI/TAU
		for metric in metrics:
			name = metric.getName().upper()
			# print name
			if name.find("WALL_CLOCK_TIME") > -1:
				return i
			i += 1
		i = 0
		# look for the usual "GET_TIME_OF_DAY" from PAPI/TAU
		for metric in metrics:
			name = metric.getName().upper()
			# print name
			if name.find("GET_TIME_OF_DAY") > -1:
				return i
			i += 1
		i = 0

	for metric in metrics:
		name = metric.getName().upper()
		# print name
		if name.find(findme) > -1:
			return i
		i += 1
	return -1

def getEvents(pe, trial, metricIndex, groupName, contains):
	# create a dictionary of events and their performance measurements
	returnEvents = {}
	events = pe.getEventList(trial, metricIndex)
	while events.hasNext():
		event = events.next()
		if (groupName == "") or \
		(contains == True and event.getGroup().find(groupName) > -1) or \
		(contains == False and event.getGroup().find(groupName) < 0):
			returnEvents[event.getName()] = event.getMeanSummary()
	return returnEvents

def findMain(events, metric):
	inclusive = 0.0
	main = {}
	for key in events.keys():
		data = events[key]
		if data.getInclusive(metric) > inclusive:
			inclusive = data.getInclusive(metric)
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

def sort_by_value(d):
    """ Returns the keys of dictionary d sorted by their values """
    items=d.items()
    backitems=[ [v[1],v[0]] for v in items]
    backitems.sort()
    return [ backitems[i][1] for i in range(0,len(backitems))]

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
	results["faster"] = faster
	results["slower"] = slower
	return results

def pairwiseEventDerived(baseEvents, otherEvents, i, j, m, n, filter):
	faster = {}
	slower = {}
	for event in baseEvents.keys():
		if filter[event] == 1:
			baseValues = baseEvents[event]
			otherValues = otherEvents[event]
			baseSeconds = baseValues.getExclusive(m) / million
			otherSeconds = otherValues.getExclusive(n) / million
			baseGFLOP = baseValues.getExclusive(i) / billion
			otherGFLOP = otherValues.getExclusive(j) / billion
			baseGFLOPS = baseGFLOP / baseSeconds
			otherGFLOPS = otherGFLOP / otherSeconds
			diff = (baseGFLOPS - otherGFLOPS)
			if diff > 0:
				faster[event] = abs(diff)
			else:
				slower[event] = abs(diff)
	results = {}
	results["faster"] = faster
	results["slower"] = slower
	return results

def mainReport(baseMain, otherMain, baseName, otherName):
	if baseMain["inclusive"] > otherMain["inclusive"]:
		tmp = "\nSelected trial (" + otherName + ") is relatively faster than baseline trial (" + baseName + ").\n"
		print tmp
		percentage = (baseMain["inclusive"] - otherMain["inclusive"]) / otherMain["inclusive"]
		fasterSlower = -1
	elif baseMain["inclusive"] < otherMain["inclusive"]:
		tmp = "\nSelected trial (" + otherName + ") is relatively slower than baseline trial (" + baseName + ").\n"
		print tmp
		percentage = (otherMain["inclusive"] - baseMain["inclusive"]) / baseMain["inclusive"]
		fasterSlower = 1
	else:
		print "\nBaseline trial and second trial have the same execution time."
		fasterSlower = 0
		percentage = 0.0
	# print "\t", baseName, baseMain["name"], ":", baseMain["inclusive"], "seconds\n", 
	# print "\t", otherName, otherMain["name"], ":", otherMain["inclusive"], "seconds\n", 
	print "\t", baseName, ":", baseMain["inclusive"]/million, "seconds\n", 
	print "\t", otherName, ":", otherMain["inclusive"]/million, "seconds\n", 
	if fasterSlower > 0:
		print "\t Relative Difference: ", percentage*100, "% slower\n"
	elif fasterSlower < 0:
		print "\t Relative Difference: ", percentage*100, "% faster\n"
	else:
		print "\t Relative Difference: ", percentage*100, "%\n"
	return fasterSlower

def showSignificantTimeEvents(diffs, type, totalRuntime, significant, baseEvents, x):
	events = diffs[type]
	shown = 0
	orderedKeys = sort_by_value(events)
	orderedKeys.reverse()
	for key in orderedKeys:
		# don't show more than 10 differences
		if shown < 10:
			# don't show insignificant differences
			if events[key] / totalRuntime > .01:
				if baseEvents[key].getExclusive(x) > 0:
					percent = ( events[key]/baseEvents[key].getExclusive(x) ) * 100.0
				else:
					percent = 0.0
				print "\t", key, ":", events[key]/million, "seconds", type.upper(), "than baseline (", percent, "% )"

				significant[key] = 1
				shown += 1
			else:
				significant[key] = 0
		else:
			significant[key] = 0
	return shown
		

def showSignificantEvents(diffs, type, significant, baseEvents, x):
	events = diffs[type]
	if type == "faster":
		type = "LESS"
	else:
		type = "MORE"

	shown = 0
	orderedKeys = sort_by_value(events)
	orderedKeys.reverse()
	for key in orderedKeys:
		# don't show insignificant differences
		if significant[key] == 1:
			if baseEvents[key].getExclusive(x) > 0:
				percent = ( events[key]/baseEvents[key].getExclusive(x) ) * 100.0
			else:
				percent = 0.0
			print "\t", key, ":", events[key]/million, "million", type, "than baseline (", percent, "% )"
			shown += 1
	return shown

def showSignificantEventsDerived(diffs, type, significant, baseEvents, x, m):
	events = diffs[type]
	if type == "faster":
		type = "LESS"
	else:
		type = "MORE"

	shown = 0
	orderedKeys = sort_by_value(events)
	orderedKeys.reverse()
	for key in orderedKeys:
		# don't show insignificant differences
		if significant[key] == 1:
			if baseEvents[key].getExclusive(x) > 0:
				baseSeconds = baseEvents[key].getExclusive(x) / million
				baseGFLOP = baseEvents[key].getExclusive(x) / billion
				percent = ( events[key]/(baseGFLOP/baseSeconds) ) * 100.0
			else:
				percent = 0.0
			print "\t", key, ":", events[key], type, "than baseline (", percent, "% )"
			shown += 1
	return shown

def DoAnalysis(pe):
	# set the application, experiment, trial
	# pe.setApplication("NPB2.3")
	# pe.setExperiment("garuda")
	# baseTrialName = "results.8"
	# otherTrialName = "results.16"
	pe.setApplication("mm2")
	pe.setExperiment("problem size")
	baseTrialName = "1000"
	otherTrialName = "2000"
	baseTrial = pe.setTrial(baseTrialName)

	# baseMetrics is a List
	baseMetrics = baseTrial.getMetrics().toArray()

	# find the time metric
	baseTime = findMetric(baseMetrics, "TIME")
	baseFlops = findMetric(baseMetrics, "PAPI_FP_INS")

	# get all the data for each event
	baseEvents = getEvents(pe, baseTrial, baseTime, "TAU_CALLPATH", False)
	
	# find the main event
	baseMain = findMain(baseEvents, baseTime)
	# print baseMain
	
	# get the other trial
	# pe.setExperiment("garuda")
	otherTrial = pe.setTrial(otherTrialName)

	# otherMetrics is a List
	otherMetrics = otherTrial.getMetrics().toArray()

	# find the time metric
	metricMap = mapMetrics(baseMetrics, otherMetrics)
	otherTime = metricMap[baseTime]

	# get all the data for each event
	otherEvents = getEvents(pe, otherTrial, otherTime, "TAU_CALLPATH", False)

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
	significant = {}
	print "Significant", baseMetrics[baseTime], "differences between trials:\n"
	shown = showSignificantTimeEvents(diffs, "faster", baseMain["inclusive"], significant, baseEvents, baseTime)
	if shown > 0:
		print ""
	shown += showSignificantTimeEvents(diffs, "slower", baseMain["inclusive"], significant, baseEvents, baseTime)
	if shown == 0:
		print "\t None.\n"

	# significant is now populated with the names of events that are significant.
	# iterate through the metrics to locate possible causes for the time difference.

	x = 0
	for metric in baseMetrics:
		if x != baseTime:
			try:
				y = metricMap[x]
			except KeyError:
				pass
			else:
				diffs = pairwiseEvent(baseEvents, otherEvents, x, y, significant)
				print "\nSignificant", baseMetrics[x], "differences between trials:\n"
				shown = showSignificantEvents(diffs, "faster", significant, baseEvents, x)
				if shown > 0:
					print ""
				shown += showSignificantEvents(diffs, "slower", significant, baseEvents, x)
				if shown == 0:
					print "\t None.\n"
				if x == baseFlops:
				# also do GFLOP/Second per processor
					diffs = pairwiseEventDerived(baseEvents, otherEvents, x, y, baseTime, otherTime, significant)
					print "\nSignificant GFLOP/sec per processor differences between trials:\n"
					shown = showSignificantEventsDerived(diffs, "faster", significant, baseEvents, x, baseTime)
					if shown > 0:
						print ""
					shown += showSignificantEventsDerived(diffs, "slower", significant, baseEvents, x, baseTime)
					if shown == 0:
						print "\t None.\n"
		x += 1
	
print "--------------- JPython test script start ------------"

pe = ScriptFacade()
DoAnalysis(pe)

print "\n"
print "---------------- JPython test script end -------------"

# pe.exit()

