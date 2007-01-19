from client import ScriptFacade
from common import TransformationType
from common import AnalysisType
from common import EngineType

print "--------------- JPython test script start ------------"

pe = ScriptFacade("/home/khuck/.ParaProf/perfdmf.cfg", EngineType.WEKA)
pe.doSomething()

# let's do something interesting...

pe.setApplication("gyro.B1-std")
pe.setExperiment("B1-std.seaborg")
pe.setTrial("B1-std.timing.seaborg.16")
pe.setMetric("WALL_CLOCK_TIME")
pe.setDimensionReduction(TransformationType.OVER_X_PERCENT, "2")
pe.setAnalysisType(AnalysisType.K_MEANS)
#pe.requestAnalysis()
pe.showDataSummary()

pe.exit()

print "---------------- JPython test script end -------------"
