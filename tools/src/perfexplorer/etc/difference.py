from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1
False = 0

def diffs():
	print "doing phase test for gtc on jaguar"
	# load the trials
	Utilities.setSession("PERI_DB_production")
	baseline = Utilities.getTrial("gtc", "jaguar", "64")
	comparison = Utilities.getTrial("gtc", "thunder", "64")

	diff = DifferenceOperation(baseline)
	diff.addInput(comparison)
	diff.processData()
	metaDiff = DifferenceMetadataOperation(baseline, comparison)
	print metaDiff.differencesAsString()

	return

print "--------------- JPython test script start ------------"

diffs()

# pe.exit()

print "---------------- JPython test script end -------------"
