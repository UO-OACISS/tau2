from client import ScriptFacade
from common import TransformationType
from common import AnalysisType
from common import EngineType

def GTC_s(pe):
	# simple database
	pe.setApplication("GTC_s")
	pe.setExperiment("tau_throttle")
	baseline = pe.setTrial("callpath")
	comparison = pe.setTrial("nocallpath")
	pe.runComparisonRules(baseline, comparison)

def mm2(pe):
	# test database
	pe.setApplication("mm2")
	pe.setExperiment("problem size")
	baseline = pe.setTrial("2000")
	comparison = pe.setTrial("1000")
	pe.runComparisonRules(baseline, comparison)

def gyro(pe):
	# perfdmf_uploaded database
	pe.setSession("perfdmf_uploaded")
	pe.setApplication("gyro.B1-std")
	pe.setExperiment("B1-std-nl2.cheetah.noaffnosng")
	baseline = pe.setTrial("B1-std-nl2.timing.cheetah.16.noaffnosng")
	comparison = pe.setTrial("B1-std-nl2.timing.cheetah.32.noaffnosng")
	pe.runComparisonRules(baseline, comparison)

	pe.setApplication("gyro.B1-std")
	pe.setExperiment("B1-std-nl2.phoenix.0x002scr")
	baseline = pe.setTrial("B1-std-nl2.timing.phoenix.16.0x002scr")
	comparison = pe.setTrial("B1-std-nl2.timing.phoenix.32.0x002scr")
	pe.runComparisonRules(baseline, comparison)

def GTC(pe):
	# perfdmf_uploaded database
	pe.setApplication("GTC compiler options loops")
	pe.setExperiment("ocracoke-440d")
	baseline = pe.setTrial("gtcmpi")
	comparison = pe.setTrial("gtcmpi-O2")
	pe.runComparisonRules(baseline, comparison)

	pe.setApplication("GTC compiler options loops")
	pe.setExperiment("ocracoke-440d")
	baseline = pe.setTrial("gtcmpi")
	comparison = pe.setTrial("gtcmpi-O3")
	pe.runComparisonRules(baseline, comparison)

	pe.setApplication("GTC compiler options loops")
	pe.setExperiment("ocracoke-440d")
	baseline = pe.setTrial("gtcmpi")
	comparison = pe.setTrial("gtcmpi-O4")
	pe.runComparisonRules(baseline, comparison)

	pe.setApplication("GTC compiler options loops")
	pe.setExperiment("ocracoke-440d")
	baseline = pe.setTrial("gtcmpi")
	comparison = pe.setTrial("gtcmpi-O5")
	pe.runComparisonRules(baseline, comparison)


print "--------------- JPython test script start ------------"

pe = ScriptFacade()
gyro(pe)

pe.exit()

print "---------------- JPython test script end -------------"
