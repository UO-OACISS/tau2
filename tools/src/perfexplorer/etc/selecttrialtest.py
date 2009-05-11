from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.common import TransformationType
from edu.uoregon.tau.perfexplorer.common import AnalysisType

print "--------------- JPython test script start ------------"

pe = ScriptFacade("/home/khuck/.ParaProf/perfdmf.cfg")
pe.doSomething()

# create a dictionary
criteria = "trial.node_count > 32 and experiment.id = 80"
trials = pe.getTrialList(criteria)
for t in trials:
	print t.getName()," ",t.getExperimentID()

pe.exit()

print "---------------- JPython test script end -------------"
