from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.glue import *
from java.util import *

###################################################################

import re

def sort_nicely( l ): 
  """ Sort the given list in the way that humans expect. """ 
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  l.sort( key=alphanum_key ) 
  return l

###################################################################

nonMPI     = "Computation"
MPI        = "MPI"
kernNonMPI = "Kernel Computation"
kernMPI    = "Kernel MPI"
mapping = {kernNonMPI:"Communication Efficiency", nonMPI:"Communication Efficiency"}
instructionsMetric = "PAPI_TOT_INS"
cyclesMetric = "PAPI_TOT_CYC"
instructions = 0
cycles = 0
kernInstructions = 0
kernCycles = 0

###################################################################

print "--------------- JPython test script start ------------"
print "--- Looking for load imbalances --- "

# load the trial
print "loading the data..."

Utilities.setSession("local")
files = []
files.append("justtime.ppk")
trial = DataSourceResult(DataSourceResult.PPK, files, False)

print "\nProcs\t Type\t\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"

trial.setIgnoreWarnings(True)

# extract the non-callpath events from the trial
extractor = ExtractNonCallpathEventOperation(trial)
extracted = extractor.processData().get(0)
mainEvent = extracted.getMainEvent()
#print "Main Event: ", mainEvent

split = DefaultResult(trial, False)
init = "MPI_Init"
final = "MPI_Finalize"

# extract the non-MPI events from the trial
for thread in extracted.getThreads():
	# initialize
	split.putCalls(thread, nonMPI, 1)
	split.putSubroutines(thread, nonMPI, 0)
	split.putCalls(thread, MPI, 1)
	split.putSubroutines(thread, MPI, 0)
	split.putCalls(thread, kernNonMPI, 1)
	split.putSubroutines(thread, kernNonMPI, 0)
	split.putCalls(thread, kernMPI, 1)
	split.putSubroutines(thread, kernMPI, 0)
	for metric in extracted.getMetrics():
		# initialize
		split.putExclusive(thread, nonMPI, metric, 0.0)
		split.putInclusive(thread, nonMPI, metric, 0.0)
		split.putInclusive(thread, MPI, metric, 0.0)
		split.putExclusive(thread, MPI, metric, 0.0)
		split.putExclusive(thread, kernNonMPI, metric, 0.0)
		split.putInclusive(thread, kernNonMPI, metric, 0.0)
		split.putInclusive(thread, kernMPI, metric, 0.0)
		split.putExclusive(thread, kernMPI, metric, 0.0)
		# get the total runtime for this thread
		total = extracted.getInclusive(thread, mainEvent, metric)
		kernTotal = total
		for event in extracted.getEvents():
			# get the exclusive time for this event
			value = extracted.getExclusive(thread, event, metric)
			if event.startswith(MPI):
				# if MPI, add to MPI running total
				current = split.getExclusive(thread, MPI, metric)
				split.putExclusive(thread, MPI, metric, value + current)
				split.putInclusive(thread, MPI, metric, value + current)
				if event.startswith(init) or event.startswith(final):
					kernTotal = kernTotal - value
				else:
					current = split.getExclusive(thread, kernMPI, metric)
					split.putExclusive(thread, kernMPI, metric, value + current)
					split.putInclusive(thread, kernMPI, metric, value + current)
			#else:
				#current = split.getInclusive(thread, nonMPI, metric)
				#split.putExclusive(thread, nonMPI, metric, value + current)
				#split.putInclusive(thread, nonMPI, metric, value + current)

		# save the values which include all fuctions
		communication = split.getExclusive(thread, MPI, metric)
		computation = total - communication
		split.putInclusive(thread, nonMPI, metric, computation / total)
		split.putExclusive(thread, nonMPI, metric, computation / total )
		split.putInclusive(thread, MPI, metric, communication / total )
		split.putExclusive(thread, MPI, metric, communication / total )
		#print thread, split.getExclusive(thread, nonMPI, metric)

		# save the values which ignore init, finalize
		communication = split.getExclusive(thread, kernMPI, metric)
		computation = kernTotal - communication
		split.putInclusive(thread, kernNonMPI, metric, computation / kernTotal)
		split.putExclusive(thread, kernNonMPI, metric, computation / kernTotal )
		split.putInclusive(thread, kernMPI, metric, communication / kernTotal )
		split.putExclusive(thread, kernMPI, metric, communication / kernTotal )
		#print thread, split.getExclusive(thread, kernNonMPI, metric)
		if metric == instructionsMetric:
			kernInstructions = kernInstructions + kernTotal
			instructions = instructions + total
		if metric == cyclesMetric:
			kernCycles = kernCycles + kernTotal
			cycles = cycles + total
#avgIPC = (instructions / cycles)
#kernAvgIPC = (kernInstructions / kernCycles)
avgInstructions = instructions / extracted.getThreads().size()
kernAvgInstructions = kernInstructions / extracted.getThreads().size()
				
# get basic statistics
event = nonMPI
metric = "TIME"
statMaker = BasicStatisticsOperation(split, False)
stats = statMaker.processData()
stddevs = stats.get(BasicStatisticsOperation.STDDEV)
means = stats.get(BasicStatisticsOperation.MEAN)
totals = stats.get(BasicStatisticsOperation.TOTAL)
maxs = stats.get(BasicStatisticsOperation.MAX)
mins = stats.get(BasicStatisticsOperation.MIN)

# get the ratio between stddev and total
ratioMaker = RatioOperation(means, maxs)
ratios = ratioMaker.processData().get(0)

thread = 0
metric = "TIME"
event = nonMPI
mean = means.getExclusive(thread, event, metric)
max = maxs.getExclusive(thread, event, metric)
min = mins.getExclusive(thread, event, metric)
stddev = stddevs.getExclusive(thread, event, metric)
ratio = ratios.getExclusive(thread, event, metric)
print "%d\t %s\t\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%" % (trial.getThreads().size(), event, mean*100, max*100, min*100, stddev*100, ratio*100)
event = kernNonMPI
mean = means.getExclusive(thread, event, metric)
max = maxs.getExclusive(thread, event, metric)
min = mins.getExclusive(thread, event, metric)
stddev = stddevs.getExclusive(thread, event, metric)
ratio = ratios.getExclusive(thread, event, metric)
print "%d\t %s\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\n" % (trial.getThreads().size(), event, mean*100, max*100, min*100, stddev*100, ratio*100)
print "Communication Efficiency (kernel only):\t%.3f" % max
print "Load Balance (kernel only):\t\t%.3f" % ratio
#print "Average IPC (kernel only):\t\t%.10f" % kernAvgIPC
print "Total Instructions(kernel only):\t%.3f" % kernAvgInstructions

print "\nNext Step: Computing Micro Load Imbalance.\n"


print "Searching for loop events..."
# get a list of the loop names
metric = "TIME"
loopPrefix = "loop ["
loopNames = set()
index = 0
for event in extracted.getEvents():
	if event.find(loopPrefix) > -1:
		loopNames.add(event)

print "Extracting callpath events..."
# extract the callpath events
extractor = ExtractCallpathEventOperation(trial)
extracted = extractor.processData().get(0)

print "Generating Statistics..."
statMaker = BasicStatisticsOperation(extracted, False)
stats = statMaker.processData()
stddevs = stats.get(BasicStatisticsOperation.STDDEV)
means = stats.get(BasicStatisticsOperation.MEAN)
totals = stats.get(BasicStatisticsOperation.TOTAL)
maxs = stats.get(BasicStatisticsOperation.MAX)
mins = stats.get(BasicStatisticsOperation.MIN)

print "Iterating over main loop..."
print "Loop ID:\t RealCommEff\t uLB\t\t CommEff"
# iterate over the iterations
totalLoopTimeIdeal = 0
totalLoopTime = 0
totalMaxTi = 0
loopSet = {}
for loopName in sort_nicely(list(loopNames)):
	# for each iteration, find all of the subroutines (path contains loop name)
	communicationTime = 0
	for event in extracted.getEvents():
		if event.find(loopName) > -1:
			tokens = event.split("=>")
			# for each MPI routine in the iteration, compute the T_ideal
			if len(tokens) > 1 and tokens[len(tokens)-1].find("MPI_") > -1:
				communicationTime = communicationTime + mins.getExclusive(0, event, metric)
				#print event, mins.getExclusive(0, event, metric)
			elif len(tokens) > 1 and tokens[len(tokens)-1].strip().find(loopName) > -1:
				loopTime = means.getInclusive(0, event, metric)
				maxTi = maxs.getInclusive(0, event, metric)
	loopTimeIdeal = loopTime - communicationTime
	#print loopName, loopTime, communicationTime, loopTimeIdeal
	realCommEff = loopTimeIdeal / loopTime
	uLB = loopTimeIdeal / maxTi
	commEff = realCommEff * uLB
	print "%s:\t %.5f\t %.5f\t %.5f" % (loopName, realCommEff, uLB, commEff)
	totalLoopTimeIdeal = totalLoopTimeIdeal + loopTimeIdeal
	totalLoopTime = totalLoopTime + loopTime
	totalMaxTi = totalMaxTi + maxTi
	#loopSet[loopName] = realCommEff, uLB, commEff

realCommEff = totalLoopTimeIdeal / totalLoopTime
uLB = totalLoopTimeIdeal / totalMaxTi
#uLB = totalMaxTi / totalLoopTimeIdeal
commEff = realCommEff * uLB
print "\n\nLoop ID:\t RealCommEff\t uLB\t\t CommEff"
print "Total: \t\t %.5f\t %.5f\t %.5f\n" % (realCommEff, uLB, commEff)
			
print "---------------- JPython test script end -------------"
