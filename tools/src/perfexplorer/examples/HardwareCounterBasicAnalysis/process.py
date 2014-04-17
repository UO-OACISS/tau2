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
#	   -p "inputData=<data.ppk>,rules=<full-path>/OpenUH.drl"
#
###############################################################################

from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.client import *
from java.util import *
from java.lang import *
import sys
import re
import time
import parsetrial
import samples
import getTopX

#########################################################################################

True = 1
False = 0
inputData = "cpi.csv"
rules = "./OpenUH.drl"
verbose = "no"
cpuMHz = 2100.003 # this should get overridden by what is in the metadata
mainEvent = ""
cpiThreshold = 0.80			 # ignore clusters with CPI smaller than this
								# Why: no need to examine good CPI clusters
cacheMissMetric = "L1_DCM"

#########################################################################################

def getParameters():
	global inputData
	global rules
	global verbose
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	#for key in keys:
	#	print key, parameterMap.get(key)
	inputData = parameterMap.get("inputData")
	rules = parameterMap.get("rules")
	verbose = parameterMap.get("verbose")

#########################################################################################

def initialize(inputData):
	global cpuMHz
	global mainEvent
	inTrial = parsetrial.parsetrial(inputData)
	inTrial.setIgnoreWarnings(True)

	mainEvent = inTrial.getMainEvent()
	print "Main Event:", mainEvent

	#metadata = TrialThreadMetadata(inTrial)
	#for name in metadata.getFields():
	#	print name, ":", metadata.getNameValue(0,name)

	# cpuMHz = metadata.getNameValue(0,"CPU MHz")
	print "Using CPU MHz", cpuMHz

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

	print "Computing Branch Mispredictions per Instruction..."
	deriver  = DeriveMetricOperation(merged, "PAPI_BR_MSP", "PAPI_TOT_INS", DeriveMetricOperation.DIVIDE)
	deriver.setNewName("BR_MSP/TOT_INS")
	tmp.add(deriver.processData().get(0))

	print "Computing Cache Misses per Instruction..."
	deriver  = DeriveMetricOperation(merged, "PAPI_" + cacheMissMetric, "PAPI_TOT_INS", DeriveMetricOperation.DIVIDE)
	deriver.setNewName(cacheMissMetric + "/TOT_INS")
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
		print "\n===", metric, "( ", mainValue, ") ==="
		for event in events:
			shortName = Utilities.shortenEventName(event)
			eventValue = derived.getExclusive(0,event,metric)
			tmp = 1.0
			if mainValue > 0:
				tmp = eventValue/mainValue
			print "%s\t%0.3f\t%00.2f%%" % (shortName, eventValue, (tmp * 100.0))

#########################################################################################

def processRules(cpiStack):
	global rules
	global verbose
	ruleHarness = RuleHarness.useGlobalRules(rules)
	# have to use assertObject2, for some reason - why is the rule harness instance null?
	ruleHarness.setGlobal("cpiThreshold",Double(cpiThreshold))
	fact = FactWrapper("Overall", "CPI Stack", cpiStack)
	handle = ruleHarness.assertObject(fact)
	fact.setFactHandle(handle)
	print verbose
	if verbose == "yes":
		factDebug = FactWrapper("Dump CPI", "CPI Stack", cpiStack)
		handleDebug = ruleHarness.assertObject(factDebug)
		factDebug.setFactHandle(handleDebug)
	ruleHarness.processRules()
	return

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
	inTrial = initialize(inputData)
	inTrial = samples.preProcessSamples(inTrial)
	topXevents = getTopX.getTopX(inTrial, 10, AbstractResult.EXCLUSIVE)
	derived = computeDerivedMetrics(inTrial)
	dump(inTrial, derived, topXevents)
	checkRatios(derived, topXevents)
	print "\n--- Examining Time Top Events ---"
	topXevents = getTopX.getTopX(inTrial, 10, AbstractResult.EXCLUSIVE, "TIME", False)
	print "\n--- Examining", cacheMissMetric, "Top Events ---"
	topXevents = getTopX.getTopX(inTrial, 10, AbstractResult.EXCLUSIVE, "PAPI_" + cacheMissMetric)
	print "\n--- Examining BR MSP Top Events ---"
	topXevents = getTopX.getTopX(inTrial, 10, AbstractResult.EXCLUSIVE, "PAPI_BR_MSP")
	print "\n--- Examining FP INS Top Events ---"
	topXevents = getTopX.getTopX(inTrial, 10, AbstractResult.EXCLUSIVE, "PAPI_FP_INS")
	print "\n---------------- JPython test script end -------------"

#########################################################################################

if __name__ == "__main__":
	main(sys.argv[1:])

#########################################################################################

