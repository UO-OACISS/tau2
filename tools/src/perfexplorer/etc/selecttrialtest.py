from client import ScriptFacade
from common import TransformationType
from common import AnalysisType
from common import EngineType

print "--------------- JPython test script start ------------"

pe = ScriptFacade("/home/khuck/.ParaProf/perfdmf.cfg", EngineType.WEKA)
pe.doSomething()

# create a dictionary
criteria = "trial.node_count > 32 and experiment.id = 80"
trials = pe.getTrialList(criteria)
for t in trials:
	print t.getName()," ",t.getExperimentID()

pe.exit()

print "---------------- JPython test script end -------------"
