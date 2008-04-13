import sys
from edu.uoregon.tau.perfdmf import Trial
from rules import RuleHarness
from glue import PerformanceResult
from glue import Utilities
from glue import TrialResult
from glue import BasicStatisticsOperation
from glue import AbstractResult
from glue import ExtractNonCallpathEventOperation
from glue import ExtractEventOperation
from glue import DeriveMetricOperation
from glue import ScaleMetricOperation
from glue import MergeTrialsOperation
from glue import DerivedMetrics
from glue import TopXEvents
from glue import MeanEventFact
from glue import DifferenceOperation

###################################################################

True = 1
False = 0
mainEvent = ""
Joules = ""
PPJ = ""

###################################################################

def deriveMetric(input, first, second, oper):
	# derive the metric
	derivor = DeriveMetricOperation(input, first, second, oper)
	derived = derivor.processData().get(0)
	newName = derived.getMetrics().toArray()[0]
	# merge new metric with the trial
	merger = MergeTrialsOperation(input)
	merger.addInput(derived)
	merged = merger.processData().get(0)
	#print "new metric: " + newName
	return merged, newName

def scaleMetric(input, metric, value, oper):
	# derive the metric
	scaler = ScaleMetricOperation(input, metric, value, oper)
	scaled = scaler.processData().get(0)
	newName = scaled.getMetrics().toArray()[0]
	# merge new metric with the trial
	merger = MergeTrialsOperation(input)
	merger.addInput(scaled)
	merged = merger.processData().get(0)
	#print "new metric: " + newName
	return merged, newName

###################################################################

def getPowerModel(input):
	# set some metric names
	totalCycles = "PAPI_TOT_CYC"
	totalInstructions = "PAPI_TOT_INS"
	L1references = "PAPI_L1_TCA"
	L2references = "PAPI_L2_TCA"
	L3references = "PAPI_L3_TCA"
	global mainEvent

	# derive the CPU Power term
	input, CPUPower = deriveMetric(input, totalInstructions, totalCycles, DeriveMetricOperation.DIVIDE)
	input, CPUPower = scaleMetric(input, CPUPower, (0.0459 * 122), ScaleMetricOperation.MULTIPLY)
	# derive the L1 Power term
	input, L1Power = deriveMetric(input, L1references, totalCycles, DeriveMetricOperation.DIVIDE)
	input, L1Power = scaleMetric(input, L1Power, (0.0017 * 122), ScaleMetricOperation.MULTIPLY)
	# derive the L2 Power term
	input, L2Power = deriveMetric(input, L2references, totalCycles, DeriveMetricOperation.DIVIDE)
	input, L2Power = scaleMetric(input, L2Power, (0.0171 * 122), ScaleMetricOperation.MULTIPLY)
	# derive the L3 Power term
	input, L3Power = deriveMetric(input, L3references, totalCycles, DeriveMetricOperation.DIVIDE)
	input, L3Power = scaleMetric(input, L3Power, (0.935 * 122), ScaleMetricOperation.MULTIPLY)

	# sum them all up!
	input, PowerPerProc = deriveMetric(input, CPUPower, L1Power, DeriveMetricOperation.ADD)
	input, PowerPerProc = deriveMetric(input, PowerPerProc, L2Power, DeriveMetricOperation.ADD)
	input, PowerPerProc = deriveMetric(input, PowerPerProc, L3Power, DeriveMetricOperation.ADD)
	input, PowerPerProc = scaleMetric(input, PowerPerProc, 97.66, ScaleMetricOperation.ADD)

	# return the trial, and the new derived metric name
	return input, PowerPerProc

def getEnergy(input, PowerPerProc):
	global mainEvent
	time = "LINUX_TIMERS"

	# convert time to seconds
	input, seconds = scaleMetric(input, time, (1/1000000.0), ScaleMetricOperation.MULTIPLY)
	input, joules = deriveMetric(input, PowerPerProc, seconds, DeriveMetricOperation.MULTIPLY)

	# return the trial, and the new derived metric name
	return input, joules

def getFlopsPerJoule(input, EnergyPerProc):
	global mainEvent
	fpOps = "PAPI_FP_OPS"

	input, flopsPerJoule = deriveMetric(input, fpOps, EnergyPerProc, DeriveMetricOperation.DIVIDE)

	# return the trial, and the new derived metric name
	return input, flopsPerJoule

def getIPC(input):
	global mainEvent
	totalInstructions = "PAPI_TOT_INS"
	totalCycles = "PAPI_TOT_CYC"

	input, IPC = deriveMetric(input, totalInstructions, totalCycles, DeriveMetricOperation.DIVIDE)

	# return the trial, and the new derived metric name
	return input, IPC

def getIssuedPerCycle(input):
	global mainEvent
	instructionsIssued = "PAPI_TOT_IIS"
	totalCycles = "PAPI_TOT_CYC"

	input, issuedPerCycle = deriveMetric(input, instructionsIssued, totalCycles, DeriveMetricOperation.DIVIDE)

	# return the trial, and the new derived metric name
	return input, issuedPerCycle

###################################################################

def main(trialName):
	global mainEvent
	global True
	global False
	global Joules
	global PPJ
	
	print "--------------- JPython test script start ------------"
	print "--- Calculating Power Models --- "

	# create a rulebase for processing
	#print "Loading Rules..."
	#ruleHarness = RuleHarness.useGlobalRules("openuh/OpenUHRules.drl")

	# load the trial
	print "loading the data..."

	# check to see if the user has selected a trial
	Utilities.setSession("openuh")
	trial = TrialResult(Utilities.getTrial("Fluid Dynamic Energy/Power", trialName, "1_1"))

	# extract the non-callpath events from the trial
	print "extracting non-callpath..."
	extractor = ExtractNonCallpathEventOperation(trial)
	extracted = extractor.processData().get(0)

	# get basic statistics
	print "computing mean..."
	statMaker = BasicStatisticsOperation(extracted, False)
	stats = statMaker.processData()
	means = stats.get(BasicStatisticsOperation.MEAN)

	# get main event
	mainEvent = means.getMainEvent()
	print "Main Event: ", mainEvent

	# calculate all derived metrics
	print
	print "Deriving power metric..."
	derived, PowerPerProc = getPowerModel(means)

	# get the top 10 events
	top10er = TopXEvents(derived, derived.getTimeMetric(), AbstractResult.EXCLUSIVE, 10)
	top10 = top10er.processData().get(0)

	# just one thread
	thread = 0

	# iterate over events, output inefficiency derived metric
	print
	#print "Top 10 Average", PowerPerProc, "values per thread for this trial:"
	#for event in top10er.getSortedEventNames():
		#print event, derived.getExclusive(thread, event, PowerPerProc)
	print
	#print mainEvent, "INCLUSIVE: ", derived.getInclusive(thread, mainEvent, PowerPerProc)
	print

	# compute the energy consumed by each event
	print "Computing joules consumed..."
	derived, EnergyPerProc = getEnergy(derived, PowerPerProc)
	Joules = EnergyPerProc

	# iterate over events, output inefficiency derived metric
	print
	#print "Top 10 Average", EnergyPerProc, "values per thread for this trial:"
	#for event in top10er.getSortedEventNames():
		#print event, derived.getExclusive(thread, event, EnergyPerProc)
	print
	#print mainEvent, "INCLUSIVE: ", derived.getInclusive(thread, mainEvent, EnergyPerProc)
	print

	# compute the floating point operations per joule per event
	print "Computing FP_OPS/joule..."
	derived, FlopsPerJoule = getFlopsPerJoule(derived, EnergyPerProc)
	PPJ = FlopsPerJoule

	# iterate over events, output inefficiency derived metric
	print
	#print "Top 10 Average", FlopsPerJoule, "values per thread for this trial:"
	#for event in top10er.getSortedEventNames():
		#print event, derived.getExclusive(thread, event, FlopsPerJoule)
	print
	#print mainEvent, "INCLUSIVE: ", derived.getInclusive(thread, mainEvent, FlopsPerJoule)
	print

	# compute the floating point operations per joule per event
	print "Computing Instructions Per Cycle..."
	derived, IPC = getIPC(derived)

	# iterate over events, output inefficiency derived metric
	print
	#print "Top 10 Average", IPC, "values per thread for this trial:"
	#for event in top10er.getSortedEventNames():
		#print event, derived.getExclusive(thread, event, IPC)
	print
	#print mainEvent, "INCLUSIVE: ", derived.getInclusive(thread, mainEvent, IPC)
	print

	# compute the floating point operations per joule per event
	print "Computing Issued Per Cycle..."
	derived, issuedPerCycle = getIssuedPerCycle(derived)

	# iterate over events, output inefficiency derived metric
	print
	#print "Top 10 Average", issuedPerCycle, "values per thread for this trial:"
	#for event in top10er.getSortedEventNames():
		#print event, derived.getExclusive(thread, event, issuedPerCycle)
	print

	#print mainEvent, "INCLUSIVE: ", derived.getInclusive(thread, mainEvent, issuedPerCycle)
	print

	#print "Time to completion..."
	print
	#for event in top10er.getSortedEventNames():
		#print event, derived.getExclusive(thread, event, derived.getTimeMetric())/1000000
	print

	#print mainEvent, "INCLUSIVE: ", derived.getInclusive(thread, mainEvent, derived.getTimeMetric())/1000000

	# process the rules
	#RuleHarness.getInstance().processRules()

	print "---------------- JPython test script end -------------"
	return derived

if __name__ == "__main__":
	first = main("O2 Optimization")
	second = main("O3 Optimization")
	differ = DifferenceOperation(first)
	differ.addInput(second)
	difference = differ.processData().get(0)

	top20er = TopXEvents(first, first.getTimeMetric(), AbstractResult.EXCLUSIVE, 20)
	top20 = top20er.processData().get(0)

	print "Joules"
	for event in top20.getEvents():
		if difference.getExclusive(0, event, Joules) < 0.0:
			print event, difference.getExclusive(0, event, Joules), first.getExclusive(0, event, Joules), second.getExclusive(0, event, Joules)

	print 
	print "Performance Per Joule"
	for event in top20.getEvents():
		if difference.getExclusive(0, event, PPJ) > 0.0:
			print event, difference.getExclusive(0, event, PPJ) , first.getExclusive(0, event, PPJ) , second.getExclusive(0, event, PPJ)
