from edu.uoregon.tau.perfexplorer.glue import *
from java.util import *
from java.lang import *
import sys
import re
import time

#########################################################################################

def preProcessSamples(inTrial):
	# keep the main timer
	mainEvent = inTrial.getMainEvent()
	print "Pre-processing Samples..."
	haveSamples = False
	aggregators = ArrayList()
	newEvents = DefaultResult(inTrial)
	newEvents.setIgnoreWarnings(True)
	# match something like '[SAMPLE] search_tr2_ [{/global/u2/k/khuck/src/XGC-1_CPU/./search.F95} {829}]'
	pattern = re.compile('\s*\{[\d]+\}\s*\]$', re.VERBOSE)
	# iterate over the events, and find those with [SAMPLE] in the name
	for event in inTrial.getEvents():
		if "[SAMPLE]" in event:
			haveSamples = True
	if haveSamples:
		myEvents = ArrayList()
		myEvents.add(mainEvent)
		events = inTrial.getEvents()
		for event in events:
			if "[SAMPLE]" in event:
				match = pattern.sub("]", event)
				#print match
				myEvents.add(match)
				for metric in inTrial.getMetrics():
					tmp = newEvents.getExclusive(0,match,metric) 
					tmp = tmp + inTrial.getExclusive(0,event,metric)
					newEvents.putExclusive(0,match,metric,tmp)
					tmp = newEvents.getInclusive(0,match,metric) 
					tmp = tmp + inTrial.getInclusive(0,event,metric)
					newEvents.putInclusive(0,match,metric,tmp)
				tmp = newEvents.getCalls(0,match) 
				tmp = tmp + inTrial.getCalls(0,event)
				newEvents.putCalls(0,match,tmp)
				tmp = newEvents.getSubroutines(0,match) 
				tmp = tmp + inTrial.getSubroutines(0,event)
				newEvents.putSubroutines(0,match,tmp)
		extractor = ExtractEventOperation(newEvents, myEvents)
		extracted = extractor.processData().get(0)
		return extracted
	return inTrial

#########################################################################################

