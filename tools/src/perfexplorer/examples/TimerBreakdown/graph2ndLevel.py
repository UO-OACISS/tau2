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
	Utilities.setSession("alcf_working")
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

def drawAreaGraph(results):
	# change this to P_WALL_CLOCK_TIME as necessary
	metric = "BGP_TIMERS"
	grapher = DrawGraph(results)
	metrics = HashSet()
	metrics.add(metric)
	grapher.setMetrics(metrics)
	grapher.setLogYAxis(False)
	grapher.setShowZero(True)
	grapher.setTitle("Runtime Breakdown of SCF-cycle: " + metric)
	grapher.setSeriesType(DrawGraph.EVENTNAME)
	grapher.setType(DrawGraph.STACKEDAREACHART);

	# for processor count X axis
	grapher.setCategoryType(DrawGraph.PROCESSORCOUNT)

	# for trial name X axis
	#grapher.setCategoryType(DrawGraph.TRIALNAME)

	# for metadata field on X axis
	#grapher.setCategoryType(DrawGraph.METADATA)
	#grapher.setMetadataField("pid")

	grapher.setXAxisLabel("Processor Count")
	grapher.setValueType(AbstractResult.INCLUSIVE)
	grapher.setYAxisLabel("Fraction of " + metric)
	grapher.processData()

def drawBarGraph(results, percent):
        # change this to P_WALL_CLOCK_TIME as necessary
        metric = "BGP_TIMERS"
        grapher = DrawGraph(results)
        metrics = HashSet()
        metrics.add(metric)
        grapher.setMetrics(metrics)
        grapher.setLogYAxis(False)
        grapher.setShowZero(True)
        grapher.setTitle("Runtime Breakdown of SCF-cycle: " + metric)
        grapher.setSeriesType(DrawGraph.EVENTNAME)
        grapher.setType(DrawGraph.STACKEDBARCHART);
	if percent == False:
        	grapher.setUnits(DrawGraph.SECONDS);

        # for processor count X axis
        grapher.setCategoryType(DrawGraph.PROCESSORCOUNT)

        # for trial name X axis
        #grapher.setCategoryType(DrawGraph.TRIALNAME)

        # for metadata field on X axis
        #grapher.setCategoryType(DrawGraph.METADATA)
        #grapher.setMetadataField("pid")

        grapher.setXAxisLabel("Processor Count")
        grapher.setValueType(AbstractResult.INCLUSIVE)
        if percent == True:
                grapher.setYAxisLabel("Fraction of " + metric)
        else:
                grapher.setYAxisLabel(metric + ", seconds")
        grapher.processData()

"""
============================================================
Timing:                               incl.     excl.
============================================================
SCF-cycle:                          533.921     2.114   0.3% |
Density:                             2.180     0.005   0.0% |
Hamiltonian:                        42.296     0.009   0.0% |
Orthonormalize:                    154.735     0.008   0.0% |
RMM-DIIS:                          143.417    85.774  11.6% |----|
Subspace diag:                     189.179     0.007   0.0% |
Other:                                1.732     1.732   0.2% |
============================================================
Total:                                        739.086 100.0%
============================================================
"""

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	#inputs = loadFromFiles()
	inputs = loadFromDB()

	# extract the events of interest
	events = ArrayList()
	events.add("Density")
	events.add("Hamiltonian")
	events.add("Orthonormalize")
	events.add("RMM-DIIS")
	events.add("SCF-cycle")
	events.add("Subspace diag")
	extractor = ExtractEventOperation(inputs, events)
	extracted = extractor.processData()
	# Put the exclusive value for SCF-cycle in its inclusive spot.
	# that is so it shows up on the graph, but doesn't confuse things.
	metric = "BGP_TIMERS"
	for trial in extracted:
		for thread in trial.getThreads():
			tmp = trial.getExclusive(thread, "SCF-cycle", metric)
			trial.putInclusive(thread, "SCF-cycle", metric, tmp)

	drawBarGraph(extracted, False)
	# extract a second time, this time with the outer function
	extractor = ExtractEventOperation(inputs, events)
	extracted = extractor.processData()
	index = 0
	while index < extracted.size():
		trial = extracted.get(index)
		master = inputs.get(index)
		for thread in trial.getThreads():
			outer = master.getInclusive(thread, "SCF-cycle", metric)
			total = 0.0
			for event in events:
				if event != "SCF-cycle":
					inner = trial.getInclusive(thread, event, metric) 
					trial.putInclusive(thread, event, metric, (inner/outer))
					total = total + (inner/outer)
			if total > 1.0:
				total = 1.0
			trial.putInclusive(thread, "SCF-cycle", metric, 1.0 - total)
		index = index + 1
	drawAreaGraph(extracted)
	drawBarGraph(extracted, True)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
