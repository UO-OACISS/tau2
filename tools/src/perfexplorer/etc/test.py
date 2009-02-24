from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.common import TransformationType
from edu.uoregon.tau.perfexplorer.common import AnalysisType

print "--------------- JPython test script start ------------"

x = 2 + 5
print x

pe = ScriptFacade()
pe.doSomething()

# let's do something interesting...

#pe.setApplication("gyro.B1-std")
#pe.setExperiment("B1-std.seaborg")
#pe.setTrial("B1-std.timing.seaborg.16")
#pe.setMetric("WALL_CLOCK_TIME")
#pe.setDimensionReduction(TransformationType.OVER_X_PERCENT, "2")
#pe.setAnalysisType(AnalysisType.K_MEANS)
#pe.requestAnalysis()
#pe.showDataSummary()

pe.setApplication("flash")
pe.setExperiment("hydro-radiation-scaling")
pe.setTrial("64p")
pe.setMetric("Time")
pe.setDimensionReduction(TransformationType.OVER_X_PERCENT, "2")
# pe.requestAnalysis()
data = pe.getPerformanceData()
# data = pe.doDimensionReduction(data)
# pe.showDataSummary()

# pe.exit()

print "---------------- JPython test script end -------------"
