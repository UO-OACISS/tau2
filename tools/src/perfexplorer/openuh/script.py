from edu.uoregon.tau.perfdmf import Trial
from rules import RuleHarness
from glue import PerformanceResult
from glue import Utilities
from glue import TopXEvents
from glue import TrialMeanResult
from glue import AbstractResult
from glue import DataNeeded
from glue import DeriveMetricOperation
from glue import ExtractEventOperation
from glue import ExtractNonCallpathEventOperation
from glue import ExtractCallpathEventOperation
from glue import DeriveMetricOperation
from glue import DeriveAllMetricsOperation
from glue import MergeTrialsOperation
from glue import MeanEventFact
from glue import DerivedMetrics

True = 1
False = 0
unstalledCyclesPerUsefulInstruction = ""
L1DataMisses  = ""
instructionMisses  = ""
fpStalls  = ""
flushStalls  = ""
L2DataHits  = ""
L3DataHits = ""
L3DataAccesses  = ""
branchPredictionStalls  = ""
allStalls  = ""

def loadRules():
	global ruleHarness
	print "Loading Rules..."
	ruleHarness = RuleHarness.useGlobalRules("rules/GeneralRules.drl")
	ruleHarness.addRules("rules/ApplicationRules.drl")
	ruleHarness.addRules("rules/MachineRules.drl")
	return 

def loaddata():
	print "loading the data..."
	Utilities.setSession("openuh")
	#Utilities.setSession("perfdmf")
	#trial = TrialMeanResult(Utilities.getTrial("fortran", "test", "O3-2048-real.8-bounds"))
	#trial = TrialMeanResult(Utilities.getTrial("fortran", "test", "O3-2048-real.8-options"))
	#trial = TrialMeanResult(Utilities.getTrial("msap_parametric.static", "size.400", "1.threads"))
	trial = TrialMeanResult(Utilities.getTrial("msap_parametric.optix.static", "size.400", "16.threads"))
	return trial

def extractNonCallpath(input):
	# extract the non-callpath events from the trial
	extractor = ExtractNonCallpathEventOperation(input)
	#extractor = ExtractCallpathEventOperation(input)
	return extractor.processData().get(0)

def deriveMetric(input, first, second, oper):
	# derive the metric
	derivor = DeriveMetricOperation(input, first, second, oper)
	derived = derivor.processData().get(0)
	newName = derived.getMetrics().toArray()[0]
	# merge the trials
	merger = MergeTrialsOperation(input)
	merger.addInput(derived)
	merged = merger.processData().get(0)
	#print "new metric: " + newName
	return merged, newName

def deriveMetrics(input):
	global unstalledCyclesPerUsefulInstruction
	global L1DataMisses 
	global instructionMisses 
	global fpStalls 
	global flushStalls 
	global L2DataHits 
	global L3DataHits
	global L3DataAccesses 
	global branchPredictionStalls 
	global allStalls 

	newMetrics = []

	# derive the unstalled cycles per useful instruction.
	first = "CPU_CYCLES"
	second = "BACK_END_BUBBLE_ALL"
	third = "IA64_INST_RETIRED_THIS"
	fourth = "NOPS_RETIRED"
	input, stallCycleRatio = deriveMetric(input, second, first, DeriveMetricOperation.DIVIDE)
	newMetrics.append(stallCycleRatio)

	input, newName = deriveMetric(input, first, second, DeriveMetricOperation.SUBTRACT)
	input, newName2 = deriveMetric(input, third, fourth, DeriveMetricOperation.SUBTRACT)
	input, unstalledCyclesPerUsefulInstruction = deriveMetric(input, newName, newName2, DeriveMetricOperation.DIVIDE)
	newMetrics.append(unstalledCyclesPerUsefulInstruction)

	# derive the L1 Data misses.
	first = "BE_L1D_FPU_BUBBLE_L1D"
	second = "BE_EXE_BUBBLE_GRALL"
	third = "BE_EXE_BUBBLE_GRGR"
	input, newName = deriveMetric(input, first, second, DeriveMetricOperation.ADD)
	input, L1DataMisses = deriveMetric(input, newName, third, DeriveMetricOperation.SUBTRACT)
	newMetrics.append(L1DataMisses)

	# derive the instruction misses
	first = "FE_BUBBLE_IMISS"
	second = "BACK_END_BUBBLE_FE"
	third = "FE_BUBBLE_ALLBUT_IBFULL"
	input, newName = deriveMetric(input, second, third, DeriveMetricOperation.DIVIDE)
	# save this new name, we will use it later
	savedTerm = newName
	input, instructionMisses = deriveMetric(input, first, newName, DeriveMetricOperation.MULTIPLY)
	newMetrics.append(instructionMisses)

	# derive the floating point stalls
	first = "BE_EXE_BUBBLE_FRALL"
	second = "BE_L1D_FPU_BUBBLE_FPU"
	input, fpStalls = deriveMetric(input, first, second, DeriveMetricOperation.ADD)
	newMetrics.append(fpStalls)

	# derive the flush stalls
	first = "FE_BUBBLE_FEFLUSH"
	input, flushStalls = deriveMetric(input, first, savedTerm, DeriveMetricOperation.MULTIPLY)
	newMetrics.append(flushStalls)

	# derive the L2 data hits
	first = "L2_DATA_REFERENCES_L2_ALL"
	second = "L2_MISSES"
	input, L2DataHits = deriveMetric(input, first, second, DeriveMetricOperation.SUBTRACT)
	newMetrics.append(L2DataHits)

	# derive the L3 data hits
	first = "L2_MISSES"
	second = "L3_MISSES"
	input, L3DataHits = deriveMetric(input, first, second, DeriveMetricOperation.SUBTRACT)
	newMetrics.append(L3DataHits)

	# derive the L3 data accesses
	first = "DATA_EAR_CACHE_LAT_128"
	second = "L3_MISSES"
	input, L3DataAccesses = deriveMetric(input, first, second, DeriveMetricOperation.DIVIDE)
	newMetrics.append(L3DataAccesses)

	# derive the branch prediction stalls
	first = "BE_FLUSH_BUBBLE_BRU"
	second = "FE_BUBBLE_BRANCH"
	third = "FE_BUBBLE_BUBBLE"
	input, fourth = deriveMetric(input, second, savedTerm, DeriveMetricOperation.MULTIPLY)
	input, fifth = deriveMetric(input, third, savedTerm, DeriveMetricOperation.MULTIPLY)
	input, newName = deriveMetric(input, first, fourth, DeriveMetricOperation.ADD)
	input, branchPredictionStalls = deriveMetric(input, newName, fifth, DeriveMetricOperation.ADD)
	newMetrics.append(branchPredictionStalls)

	# add other metrics of interest
	newMetrics.append("BACK_END_BUBBLE_ALL")
	newMetrics.append("BE_RSE_BUBBLE_ALL")
	newMetrics.append("BE_EXE_BUBBLE_GRGR")
	newMetrics.append("PAPI_L1_DCH")
	newMetrics.append("PAPI_L3_DCM")

	# add all stalls together
	input, allStalls = deriveMetric(input, L1DataMisses, instructionMisses, DeriveMetricOperation.ADD)
	input, allStalls = deriveMetric(input, allStalls, fpStalls, DeriveMetricOperation.ADD)
	input, allStalls = deriveMetric(input, allStalls, flushStalls, DeriveMetricOperation.ADD)
	input, allStalls = deriveMetric(input, allStalls, branchPredictionStalls, DeriveMetricOperation.ADD)
	input, allStalls = deriveMetric(input, allStalls, "BE_RSE_BUBBLE_ALL", DeriveMetricOperation.ADD)
	input, allStalls = deriveMetric(input, allStalls, "BE_EXE_BUBBLE_GRGR", DeriveMetricOperation.ADD)
	newMetrics.append(allStalls)
	return input, newMetrics

def stallPercentages(input):
	global L1DataMisses 
	global instructionMisses 
	global fpStalls 
	global flushStalls 
	global branchPredictionStalls 
	newMetrics = []

	# get the percentages of total stalls for each stall source
	total = "BACK_END_BUBBLE_ALL"
	input, newName = deriveMetric(input, L1DataMisses, total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, instructionMisses, total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, fpStalls, total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, flushStalls, total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, branchPredictionStalls, total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, "BE_RSE_BUBBLE_ALL", total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, tmpName = deriveMetric(input, "BE_EXE_BUBBLE_GRGR", total, DeriveMetricOperation.DIVIDE)
	waitName = tmpName

	for name in newMetrics:
		input, tmpName = deriveMetric(input, tmpName, name, DeriveMetricOperation.ADD)
	newMetrics.append(waitName)
	newMetrics.append(tmpName)

	return input, newMetrics

def stallHPPercentages(input):
	newMetrics = []
	total = "BACK_END_BUBBLE_ALL"

	input, newName = deriveMetric(input, "BE_FLUSH_BUBBLE_ALL", total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, "BE_L1D_FPU_BUBBLE_ALL", total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, "BE_EXE_BUBBLE_ALL", total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, "BE_RSE_BUBBLE_ALL", total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, tmpName = deriveMetric(input, "BACK_END_BUBBLE_FE", total, DeriveMetricOperation.DIVIDE)
	waitName = tmpName

	for name in newMetrics:
		input, tmpName = deriveMetric(input, tmpName, name, DeriveMetricOperation.ADD)
	newMetrics.append(waitName)
	newMetrics.append(tmpName)

	input, newName = deriveMetric(input, total, "CPU_CYCLES", DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)

	return input, newMetrics

def stallL1BreakdownPercentages(input):
	newMetrics = []
	total = "BE_L1D_FPU_BUBBLE_L1D"

	input, newName = deriveMetric(input, "BE_L1D_FPU_BUBBLE_L1D_DCURECIR", total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, "BE_L1D_FPU_BUBBLE_L1D_TLB", total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, "BE_L1D_FPU_BUBBLE_L1D_STBUFRECIR", total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, newName = deriveMetric(input, "BE_L1D_FPU_BUBBLE_L1D_FULLSTBUF", total, DeriveMetricOperation.DIVIDE)
	newMetrics.append(newName)
	input, tmpName = deriveMetric(input, "BE_L1D_FPU_BUBBLE_L1D_L2BPRESS", total, DeriveMetricOperation.DIVIDE)
	waitName = tmpName

	for name in newMetrics:
		input, tmpName = deriveMetric(input, tmpName, name, DeriveMetricOperation.ADD)
	newMetrics.append(waitName)
	newMetrics.append(tmpName)

	return input, newMetrics


print "--------------- JPython test script start ------------"
print "doing single trial analysis for mm on gomez"

# create a rulebase for processing
#loadRules()

# load the trial
trial = loaddata()
#event = trial.getMainEvent()
#event = "LOOP #0 [file:/home1/khuck/src/fortran/mm.f <12, 16>]"
#event = "LOOP #3 [file:/home1/khuck/src/fortran/mm-real.f <31, 37>]"
# TAU instrumentation
#event = "LOOP 0"
#event = "LOOP 3"
# BOUNDS
#event = "sub_"
#event = "LOOP #2 [file:/home1/khuck/src/fpga/msap.c <65, 158>]"
event = "LOOP #2 [file:/mnt/netapp/home1/khuck/openuh/src/fpga/msap.c <65, 158>]"

# extract the non-callpath events
#extracted = extractNonCallpath(trial)
extracted = trial

# calculate all derived metrics
derived, newMetrics = deriveMetrics(extracted)
for thread in derived.getThreads():
	for metric in newMetrics:
		print event, metric, derived.getInclusive(thread, event, metric)

print

# get the HP stall percentages
percentages, newMetrics = stallHPPercentages(derived)
for thread in percentages.getThreads():
	for metric in newMetrics:
		print event, metric, "%.2f%%" % (percentages.getInclusive(thread, event, metric)*100.0)

print

# get the stall percentages
percentages, newMetrics = stallPercentages(derived)
for thread in percentages.getThreads():
	for metric in newMetrics:
		print event, metric, "%.2f%%" % (percentages.getInclusive(thread, event, metric)*100.0)

print

# get the HP stall percentages, breakdown of fpstalls
percentages, newMetrics = stallL1BreakdownPercentages(derived)
for thread in percentages.getThreads():
	for metric in newMetrics:
		print event, metric, "%.2f%%" % (percentages.getInclusive(thread, event, metric)*100.0)

print

#RuleHarness.getInstance().processRules()

print "---------------- JPython test script end -------------"
