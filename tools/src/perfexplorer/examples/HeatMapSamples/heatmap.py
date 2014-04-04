###############################################################################
#
# processCPI.py 
# PerfExplorer Script for processing OpenUH metrics
#
# usage: 
#	perfexplorer \
#	   -c <TAUdb configuration> \
#	   -n \
#	   -i <full-path>/process.py \
#	   -p "inputData=<data.ppk>"
#
###############################################################################

from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.client import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.util import *
from java.lang import *
import sys
import re
import time
from os import system

#########################################################################################

True = 1
False = 0
inputData = "matmult.ppk"
cpuMHz = 2100.0 # this should get overridden by what is in the metadata
mainEvent = ""
cpiThreshold = 0.80			 # ignore clusters with CPI smaller than this
								# Why: no need to examine good CPI clusters

#########################################################################################

def parsetrial(inputDataIn):
	global mainEvent
	global inputData
	Utilities.setSession("local")
	files = []
	files.append(inputData)
	print "Parsing files:", files, "..."
	trial = DataSourceResult(DataSourceResult.PPK, files, False)
	mainEvent = trial.getMainEvent()
	trial.setIgnoreWarnings(True)
	return trial

#########################################################################################

def getTopX(inTrial, threshold, timerType, metric=None, filterMPI=True):
	global mainEvent
	inTrial.setIgnoreWarnings(True)

	extracted = inTrial
	# extract computation code (remove MPI)
	if filterMPI:
		myEvents = ArrayList()
		print "Filtering out MPI calls..."
		for event in inTrial.getEvents():
			if not event.startswith("MPI_"):
				myEvents.add(event)
		extractor = ExtractEventOperation(inTrial, myEvents)
		extracted = extractor.processData().get(0)

	# put the top X events names in a list
	myEvents = ArrayList()

	# get the top X events
	print "Extracting top events..."
	extracted.setIgnoreWarnings(True)
	if metric is None:
		metric = extracted.getTimeMetric()
	topper = TopXEvents(extracted, metric, timerType, threshold) 
	topped = topper.processData().get(0)

	for event in topped.getEvents():
		shortEvent = Utilities.shortenEventName(event)
		exclusivePercent = topped.getDataPoint(0,event,metric, timerType) / extracted.getInclusive(0,extracted.getMainEvent(),metric) * 100.0
		if (exclusivePercent > 1.0):
			print "%00.2f%%\t %d\t %s" % (exclusivePercent, extracted.getCalls(0,event), shortEvent)
			myEvents.add(event)
	return myEvents
#########################################################################################

def getParameters():
	global inputData
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	for key in keys:
		print key, parameterMap.get(key)
	inputData = parameterMap.get("inputData")

#########################################################################################

def initialize(inputData):
	global cpuMHz
	global mainEvent
	print "Parsing: ", inputData
	inTrial = parsetrial(inputData)
	inTrial.setIgnoreWarnings(True)

	#mainEvent = inTrial.getMainEvent()
	print "Main Event:", mainEvent

	for metric in inTrial.getMetrics():
		print "Found Metric:", metric

	return inTrial

def computeDerivedMetrics(inTrial):
	global cpuMHz
	global mainEvent
	# derive the Cycles 
	print "Computing Cycles from TIME..."
	deriver = ScaleMetricOperation(inTrial, cpuMHz, "TIME", DeriveMetricOperation.MULTIPLY)
	deriver.setNewName("PAPI_TOT_CYC")
	tmp = deriver.processData();
	tmp.add(inTrial)
	merger = MergeTrialsOperation(tmp)
	merged = merger.processData().get(0)

	print "Computing Cycles per Instruction..."
	deriver  = DeriveMetricOperation(merged, "PAPI_TOT_CYC", "PAPI_TOT_INS", DeriveMetricOperation.DIVIDE)
	deriver.setNewName("TOT_CYC/TOT_INS")
	tmp = deriver.processData()

	print "Computing FLOPs per Instruction..."
	deriver  = DeriveMetricOperation(merged, "PAPI_FP_INS", "PAPI_TOT_INS", DeriveMetricOperation.DIVIDE)
	deriver.setNewName("FP_INS/TOT_INS")
	tmp2 = deriver.processData().get(0)
	deriver = ScaleMetricOperation(tmp2, 100.0, "FP_INS/TOT_INS", DeriveMetricOperation.MULTIPLY)
	deriver.setNewName("% FP_INS ")
	tmp.add(deriver.processData().get(0))

	#print "Computing Branch Mispredictions per Instruction..."
	#deriver  = DeriveMetricOperation(merged, "PAPI_BR_MSP", "PAPI_TOT_INS", DeriveMetricOperation.DIVIDE)
	#deriver.setNewName("BR_MSP/TOT_INS")
	#tmp.add(deriver.processData().get(0))

	print "Computing Cache Misses per Instruction..."
	deriver  = DeriveMetricOperation(merged, "PAPI_L1_TCM", "PAPI_TOT_INS", DeriveMetricOperation.DIVIDE)
	deriver.setNewName("L1_TCM/TOT_INS")
	tmp.add(deriver.processData().get(0))
	merger = MergeTrialsOperation(tmp)
	merged = merger.processData().get(0)

	for metric in merged.getMetrics():
		print "Found Metric:", metric, merged.getInclusive(0,mainEvent,metric)

	return merged

#########################################################################################

def checkRatios(derived, events):
	global mainEvent
	for metric in derived.getMetrics():
		mainValue = derived.getInclusive(0,mainEvent,metric)
		print "===", metric, "( ", mainValue, ") ==="
		for event in events:
			shortName = Utilities.shortenEventName(event)
			eventValue = derived.getExclusive(0,event,metric)
			print "%s\t%0.3f\t%00.2f%%" % (shortName, eventValue, ((eventValue / mainValue) * 100.0))

#########################################################################################

def dump(raw, derived, events):
	global mainEvent
	index = 1
	for event in events:
		print index, Utilities.shortenEventName(event)
		index = index + 1
	print ""

	index = 1
	print "Timer",
	for metric in raw.getMetrics():
		print "\t", metric,
	print ""
	for event in events:
		print index, "\t",
		for metric in raw.getMetrics():
			value = raw.getExclusive(0,event,metric) / raw.getInclusive(0,mainEvent,metric)
			if value < 0.1:
				print " %0.2f %%\t\t" % ((value * 100.0)),
			else:
				print "%0.2f %%\t\t" % ((value * 100.0)),
		index = index + 1
		print ""
	print ""

	print "Timer",
	for metric in derived.getMetrics():
		print "\t", metric,
	print ""

	index = 1
	for event in events:
		print index, "\t",
		for metric in derived.getMetrics():
			value = derived.getExclusive(0,event,metric)
			if "%" in metric:
				if value < 10.0:
					print " %0.2f %%\t\t" % (value),
				else:
					print "%0.2f %%\t\t" % (value),
			else:
				print "%0.5f\t\t" % (value),
		index = index + 1
		print ""

	print "Avg.\t",
	for metric in derived.getMetrics():
		value = derived.getInclusive(0,mainEvent,metric)
		if "%" in metric:
			if value < 10.0:
				print " %0.2f %%\t\t" % (value),
			else:
				print "%0.2f %%\t\t" % (value),
		else:
			print "%0.5f\t\t" % (value),
	print "\n"

#########################################################################################

def main(argv):
	print "--------------- JPython test script start ------------"
	getParameters()
	global fractionThreshold
	global inputData
	inTrial = initialize(inputData)
	print "Making basic stats..."
	statmaker = BasicStatisticsOperation(inTrial)
	stats = statmaker.processData().get(BasicStatisticsOperation.MEAN)
	print "Extracting Flat Profile..."
	extractor = ExtractNonCallpathEventOperation(stats)
	flat = extractor.processData().get(0)
	print "Extracting Callpath Profile..."
	extractor = ExtractCallpathEventOperation(stats)
	callpath = extractor.processData().get(0)
	print "Finding CONTEXT events..."
	contexts = ArrayList()
	samples = ArrayList()
	for event in flat.getEvents():
		if event.startswith("[CONTEXT]"):
			contexts.add(event)
		if event.startswith("[SAMPLE]"):
			samples.add(event)
	# get top 10 contexts
	extractor = ExtractEventOperation(stats,contexts)
	contextsOnly = extractor.processData().get(0)
	contexts = getTopX(contextsOnly, 10, AbstractResult.INCLUSIVE, "TIME", False)
	# get top 10 samples
	extractor = ExtractEventOperation(stats,samples)
	samplesOnly = extractor.processData().get(0)
	samples = getTopX(samplesOnly, 10, AbstractResult.INCLUSIVE, "TIME", False)
	gp = open('heatmap.gp','w')
	gp.write('set term png size 1024,768\n')
	gp.write('set output "%s-heatmap.png"\n' % inputData)
	gp.write('set title "Samples in Contexts"\n')
	gp.write('set xlabel "CONTEXT"\n')
	gp.write('set ylabel "SAMPLES"\n')
	gp.write('set tic scale 0\n')
	gp.write('set palette rgbformulae 22,13,10\n')
	print "contexts:", contexts.size()
	gp.write("set xtics (")
	for i in range(contexts.size()):
		if i > 0:
			gp.write(",")
		a = Utilities.shortenEventName(contexts[i]).replace("[CONTEXT] ","")
		gp.write("\"%s\" %d" % (a, i))
	gp.write(") rotate by 45 right\n\n")
	print "samples:", samples.size()
	gp.write("set ytics (")
	for i in range(samples.size()):
		if i > 0:
			gp.write(",")
		a = Utilities.shortenEventName(samples[i]).replace("[SAMPLE] ","")
		gp.write("\"%s\" %d" % (a, i))
		#gp.write(a)
		#gp.write(" ")
		#gp.write(i)
	gp.write(")\n\n")
	gp.write("plot '-' with image\n")
	Collections.sort(contexts)
	Collections.sort(samples)
	for i in range(contexts.size()):
		for j in range(samples.size()):
			count = 0.0
			for event in callpath.getEvents():
				#if samples[j] in event and contexts[i] in event and "[SAMPLE]" not in event:
				if samples[j] in event and contexts[i] in event:
					count = stats.getCalls(0, event)
			gp.write("%d %d %d\n" % (i, j, count))
		gp.write('\n')
	gp.close()
	system('gnuplot -persist heatmap.gp')
	System.exit(0)

#########################################################################################

if __name__ == "__main__":
	main(sys.argv[1:])

#########################################################################################

