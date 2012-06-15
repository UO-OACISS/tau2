from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0

def loadFile(fileName):
	# load the trial
	files = []
	files.append(fileName)
	input = DataSourceResult(DataSourceResult.PPK, files, False)
	return input

def loadFromFiles():
	inputs = ArrayList()
	inputs.add(loadFile("2.ppk"))
	inputs.add(loadFile("4.ppk"))
	inputs.add(loadFile("6.ppk"))
	inputs.add(loadFile("8.ppk"))
	return inputs

def loadDB(app, exp, trial):
	trial = Utilities.getTrial(app, exp, trial)
	input = TrialMeanResult(trial)
	return input

def loadFromDB():
	Utilities.setSession("alcf")
	inputs = ArrayList()
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2", "Au_bulk2x4x4_16_vn_ZYXT"))
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2", "Au_bulk2x4x4_32_vn_ZYXT"))
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2", "Au_bulk2x4x4_64_vn_ZYXT"))
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2", "Au_bulk2x4x4_128_vn_ZYXT"))
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2", "Au_bulk2x4x4_256_vn_ZYXT"))
	return inputs

def loadFromDB2():
	inputs = ArrayList()
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2_blocksize10", "Au_bulk2x4x4_16_vn_ZYXT_blocksize10"))
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2_blocksize10", "Au_bulk2x4x4_32_vn_ZYXT_blocksize10"))
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2_blocksize10", "Au_bulk2x4x4_64_vn_ZYXT_blocksize10"))
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2_blocksize10", "Au_bulk2x4x4_128_vn_ZYXT_blocksize10"))
	inputs.add(loadDB("GPAW r8581 (paper)", "Au_bulk4x2x2_blocksize10", "Au_bulk2x4x4_256_vn_ZYXT_blocksize10"))
	return inputs

def drawGraph(results):
	# change this as necessary
	metric = "BGP_TIMERS"
	grapher = DrawGraph(results)
	metrics = HashSet()
	metrics.add(metric)
	grapher.setMetrics(metrics)
	grapher.setLogYAxis(False)
	grapher.setShowZero(True)
	grapher.setTitle("Graph of Multiple Trials: " + metric)
	grapher.setSeriesType(DrawGraph.EVENTNAME)
	grapher.setUnits(DrawGraph.SECONDS)

	# for processor count X axis
	grapher.setCategoryType(DrawGraph.PROCESSORCOUNT)

	# for trial name X axis
	#grapher.setCategoryType(DrawGraph.TRIALNAME)

	# for metadata field on X axis
	#grapher.setCategoryType(DrawGraph.METADATA)
	#grapher.setMetadataField("pid")

	grapher.setXAxisLabel("Processor Count")
	grapher.setValueType(AbstractResult.INCLUSIVE)
	grapher.setYAxisLabel("Inclusive " + metric + " (seconds)")
	grapher.processData()

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	#inputs = loadFromFiles()
	inputs = loadFromDB()

	# extract the event of interest
	events = ArrayList()
	# change this to zoneLimitedGradient(PileOfScalars) as necessary
	events.add("GPAW.calculator")
	events.add("SCF-cycle")
	events.add("RMM-DIIS")
	extractor = ExtractEventOperation(inputs, events)
	extracted = extractor.processData()

	inputs = loadFromDB2()
	extractor = ExtractEventOperation(inputs, events)
	extracted2 = extractor.processData()

	for x in range(extracted.size()):
		before = extracted.get(x)
		after = extracted2.get(x)
		for thread in after.getThreads():
                	for event in after.getEvents():
        			before.putCalls(thread, event + " : blocksize 10", after.getCalls(thread, event))
        			before.putSubroutines(thread, event + " : blocksize 10", after.getSubroutines(thread, event))
        			for metric in after.getMetrics():
                			before.putExclusive(thread, event + " : blocksize 10", metric, after.getExclusive(thread, event, metric))
                			before.putInclusive(thread, event + " : blocksize 10", metric, after.getInclusive(thread, event, metric))

	drawGraph(extracted)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
