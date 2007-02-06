from client import ScriptFacade
from common import TransformationType
from common import AnalysisType
from common import EngineType

print "--------------- JPython test script start ------------"

pe = ScriptFacade()

pe.setApplication("gyro.B1-std")
pe.setExperiment("B1-std-nl2.cheetah.affnosng")
pe.addExperiment("B1-std-nl2.cheetah.affsng")
pe.addExperiment("B1-std-nl2.cheetah.noaffnosng")
# pe.setTrial("B1-std.timing.seaborg.16")
pe.setMetricName("WALL_CLOCK_TIME")
pe.doGeneralChart()

# pe.exit()

print "---------------- JPython test script end -------------"
