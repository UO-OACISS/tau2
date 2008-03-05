from glue import Utilities
from edu.uoregon.tau.perfdmf import Trial
from glue import DifferenceOperation
from glue import DifferenceMetadataOperation
from rules import RuleHarness


True = 1

def glue():
	print "Comparing performance between two trials, including metadata"

	# load the trials
	Utilities.setSession("perfdmf_test")
	base = Utilities.getTrial("gtc_bench", "superscaling.jaguar", "64")
	comp = Utilities.getTrial("gtc_bench", "superscaling.jaguar", "128")

	# compare the performance
	diff = DifferenceOperation(base)
	diff.addInput(comp)
	diff.processData()

	# compare the metadata
	meta = DifferenceMetadataOperation(base, comp)

	# process the rules
	ruleHarness = RuleHarness("/rules/SampleRules.drl")
	ruleHarness.assertObject(meta)
	ruleHarness.assertObject(diff)
	ruleHarness.processRules()
	# print ruleHarness.getLog()

print "--------------- JPython test script start ------------"

glue()

# pe.exit()

print "---------------- JPython test script end -------------"
