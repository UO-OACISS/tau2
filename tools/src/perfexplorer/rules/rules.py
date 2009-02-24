from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1
False = 0

def rules():
	print "doing phase test for gtc on jaguar"
	# load the trial
	Utilities.setSession("perfdmf.test")
	baseline = Utilities.getTrial("gtc_bench", "superscaling.jaguar", "64")
	comparison = Utilities.getTrial("gtc_bench", "superscaling.jaguar", "128")

	diff = DifferenceOperation(baseline)
	diff.addInput(comparison)
	diff.processData()
	metaDiff = DifferenceMetadataOperation(baseline, comparison)
	print metaDiff.differencesAsString()
	print "****** Processing Super Duper Rules! ******"
	ruleHarness = RuleHarness("rules/GeneralRules.drl")
	ruleHarness.addRules("rules/ApplicationRules.drl")
	ruleHarness.addRules("rules/MachineRules.drl")
	ruleHarness.assertObject(metaDiff)
	ruleHarness.assertObject(diff)
	ruleHarness.processRules()

	print "got the data"

	return

print "--------------- JPython test script start ------------"

rules()

# pe.exit()

print "---------------- JPython test script end -------------"
