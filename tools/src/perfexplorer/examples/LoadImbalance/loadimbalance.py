from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.glue import *
from java.util import *

###################################################################

def drawit(list, title, ylabel):
	# graph the ratios, showing the inefficiency as the app scales

	events = HashSet()
	events.add(LoadImbalanceOperation.COMPUTATION)
	events.add(LoadImbalanceOperation.KERNEL_COMPUTATION)
	events.add(LoadImbalanceOperation.COMMUNICATION)
	events.add(LoadImbalanceOperation.KERNEL_COMMUNICATION)

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

	# compute the load imbalance
	splitter = LoadImbalanceOperation(extracted)
	loadBalance = splitter.processData()
				
	thread = 0
	metric = trial.getTimeMetric()
	event = LoadImbalanceOperation.KERNEL_COMPUTATION

	means = loadBalance.get(LoadImbalanceOperation.MEAN)
	maxs = loadBalance.get(LoadImbalanceOperation.MAX)
	mins = loadBalance.get(LoadImbalanceOperation.MIN)
	stddevs = loadBalance.get(LoadImbalanceOperation.STDDEV)
	ratios = loadBalance.get(LoadImbalanceOperation.LOAD_BALANCE)

	mean = means.getExclusive(thread, event, metric)
	max = maxs.getExclusive(thread, event, metric)
	min = mins.getExclusive(thread, event, metric)
	stddev = stddevs.getExclusive(thread, event, metric)
	ratio = ratios.getExclusive(thread, event, metric)

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
