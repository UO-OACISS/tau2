from client import ScriptFacade
from glue import PerformanceResult
from glue import PerformanceAnalysisOperation
from glue import Utilities
from glue import TrialResult
from glue import AbstractResult
from glue import KMeansOperation
from glue import TopXEvents
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

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
