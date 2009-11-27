from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.glue import *
from java.util import *

###################################################################

nonMPI = "Computation"
MPI = "MPI"
kernNonMPI = "Kernel Computation"
kernMPI = "Kernel MPI"

###################################################################

def drawit(list, title, ylabel):
	# graph the ratios, showing the inefficiency as the app scales
	global nonMPI
	global MPI
	global kernNonMPI
	global kernMPI

	events = HashSet()
	events.add(nonMPI)
	events.add(kernNonMPI)
	events.add(MPI)
	#events.add(kernMPI)

	grapher = DrawGraph(list)
	grapher.setEvents(events)
	grapher.setLogYAxis(False)
	grapher.setShowZero(True)
	grapher.setTitle(title)
	grapher.setSeriesType(DrawGraph.EVENTNAME)
	grapher.setUnits(DrawGraph.MICROSECONDS)
	grapher.setCategoryType(DrawGraph.PROCESSORCOUNT)
	grapher.setXAxisLabel("Processor Count")
	grapher.setValueType(AbstractResult.EXCLUSIVE)
	grapher.setYAxisLabel(ylabel)
	grapher.processData()

###################################################################

print "--------------- JPython test script start ------------"
print "--- Looking for load imbalances --- "

# load the trial
print "loading the data..."

Utilities.setSession("local")
trials = ArrayList()
for i in [2,4,8,16,32,64,128,256]:
	#trials.add(TrialResult(Utilities.getTrial("GROMACS", "MareNostrum", str(i))))
	files = []
	files.append(str(i) + ".ppk")
	trials.add(DataSourceResult(DataSourceResult.PPK, files, False))

print "Procs\t Type\t\t AVG\t MAX\t MIN\t STDEV\t AVG/MAX"

ratioList = ArrayList()
meanList = ArrayList()
maxList = ArrayList()
minList = ArrayList()
stddevList = ArrayList()

for trial in trials:

	# extract the non-callpath events from the trial
	trial.setIgnoreWarnings(True)
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
				
	# get basic statistics
	event = nonMPI
	metric = "TIME"
	statMaker = BasicStatisticsOperation(split, False)
	stats = statMaker.processData()
	stddevs = stats.get(BasicStatisticsOperation.STDDEV)
	means = stats.get(BasicStatisticsOperation.MEAN)
	#totals = stats.get(BasicStatisticsOperation.TOTAL)
	maxs = stats.get(BasicStatisticsOperation.MAX)
	mins = stats.get(BasicStatisticsOperation.MIN)

	# get the ratio between stddev and total
	ratioMaker = RatioOperation(means, maxs)
	ratios = ratioMaker.processData().get(0)

	thread = 0
	metric = "TIME"
	event = kernNonMPI
	mean = means.getExclusive(thread, event, metric)
	max = maxs.getExclusive(thread, event, metric)
	min = mins.getExclusive(thread, event, metric)
	stddev = stddevs.getExclusive(thread, event, metric)
	ratio = ratios.getExclusive(thread, event, metric)
	#print trial.getThreads().size(), event, mean, max, ratio
	print "%d\t %s\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t %.2f%%\t" % (trial.getThreads().size(), event, mean*100, max*100, min*100, stddev*100, ratio*100)
	ratioList.add(ratios)
	meanList.add(means)
	maxList.add(maxs)
	minList.add(mins)
	stddevList.add(stddevs)

print

# graph the ratios, showing the inefficiency as the app scales

drawit(stddevList, "Standard Deviation, 0.0 is better", "Standard Deviation")
drawit(minList, "Min, 1.0 is better", "Min")
drawit(maxList, "Max (MAX), 1.0 is better", "Max")
drawit(meanList, "Mean (AVG), 1.0 is better", "Mean")
drawit(ratioList, "Load Balance (AVG/MAX), 1.0 is better", "Load Balance")

print "---------------- JPython test script end -------------"
