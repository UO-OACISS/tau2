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
	grapher.setValueType(AbstractResult.EXCLUSIVE)
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
	grapher.setValueType(AbstractResult.EXCLUSIVE)
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
 Atomic density matrices:            0.050     0.050   0.0% |
 Mix:                                1.624     1.624   0.2% |
 Multipole moments:                  0.107     0.107   0.0% |
 Pseudo density:                     0.394     0.392   0.1% |
  Symmetrize density:                0.002     0.002   0.0% |
Hamiltonian:                        42.296     0.009   0.0% |
 Atomic:                            10.729     1.399   0.2% |
  XC Correction:                     9.330     9.330   1.3% ||
 Communicate energies:              18.739    18.739   2.5% ||
 Hartree integrate/restrict:         0.303     0.303   0.0% |
 Poisson:                            7.760     7.760   1.0% |
 XC 3D grid:                         4.724     4.724   0.6% |
 vbar:                               0.032     0.032   0.0% |
Orthonormalize:                    154.735     0.008   0.0% |
 Blacs Band Layouts:                10.489     0.003   0.0% |
  Inverse Cholesky:                 10.486    10.486   1.4% ||
 calc_s_matrix:                     54.647    54.647   7.4% |--|
 projections:                       31.311    31.311   4.2% |-|
 rotate_psi:                        58.280    58.280   7.9% |--|
RMM-DIIS:                          143.417    85.774  11.6% |----|
 Apply hamiltonian:                 11.242    11.242   1.5% ||
 precondition:                      17.481    17.481   2.4% ||
 projections:                       28.919    28.919   3.9% |-|
Subspace diag:                     189.179     0.007   0.0% |
 Blacs Band Layouts:                83.293     0.004   0.0% |
  Diagonalize:                      83.282    83.282  11.3% |----|
  Distribute results:                0.007     0.007   0.0% |
 calc_h_matrix:                     53.117    47.269   6.4% |--|
  Apply hamiltonian:                 5.848     5.848   0.8% |
 rotate_psi:                        52.762    52.762   7.1% |--|
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

	# extract the event of interest
	events = ArrayList()
	events.add("SCF-cycle")
	events.add("Density")
	events.add("Atomic density matrices")
	events.add("Mix")
	events.add("Multipole moments")
	events.add("Pseudo density")
	events.add("Symmetrize density")
	events.add("Hamiltonian")
	events.add("Atomic")
	events.add("XC Correction")
	events.add("Communicate energies")
	events.add("Hartree integrate/restrict")
	events.add("Poisson")
	events.add("XC 3D grid")
	events.add("vbar")
	events.add("Orthonormalize")
	events.add("Blacs Band Layouts")
	events.add("Inverse Cholesky")
	events.add("calc_s_matrix")
	events.add("projections")
	events.add("rotate_psi")
	events.add("RMM-DIIS")
	events.add("Apply hamiltonian")
	events.add("precondition")
	events.add("Subspace diag")
	events.add("Diagonalize")
	events.add("Distribute results")
	events.add("calc_h_matrix")
	extractor = ExtractEventOperation(inputs, events)
	extracted = extractor.processData()
	metric = "BGP_TIMERS"
	drawBarGraph(extracted, False)
	for trial in extracted:
		for thread in trial.getThreads():
			outer = trial.getInclusive(thread, "SCF-cycle", metric)
			total = 0.0
			for event in events:
				inner = trial.getExclusive(thread, event, metric) 
				# Using the inclusive as our place holder
				trial.putExclusive(thread, event, metric, (inner/outer))
				total = total + (inner/outer)
			if total > 1.0:
				total = 1.0
			trial.putExclusive(thread, "Other", metric, 1.0 - total)

	drawAreaGraph(extracted)
	drawBarGraph(extracted, True)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
