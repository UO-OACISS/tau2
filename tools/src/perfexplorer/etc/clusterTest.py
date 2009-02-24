from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import Trial
from java.util import *

True = 1

def glue():
	print "doing cluster test"
	# load the trial
	Utilities.setSession("peri_s3d")
	trial = Utilities.getTrial("S3D", "hybrid-study", "hybrid")
	result = TrialResult(trial)

	reducer = TopXEvents(result1, 10)
	reduced = reducer.processData().get(0)

	for metric in reduced.getMetrics():
		k = 2
		while k<= 10:
			kmeans = KMeansOperation(reduced, metric, AbstractResult.EXCLUSIVE, k)
			kmeans.processData()

print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
