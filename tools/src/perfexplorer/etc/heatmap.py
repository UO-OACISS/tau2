from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0

print "--------------- JPython test script start ------------"
#Utilities.setSession("peris3d")
#trial = Utilities.getTrial("s3d", "intrepid-c2h4-misc", "512_com")
Utilities.setSession("test")
#trial = Utilities.getTrial("NPB_SP", "CONTEXT", "4p")
trial = Utilities.getTrial("NPB_SP", "CONTEXT", "64p")
#trial = Utilities.getTrial("NPB_SP", "TAU_EACH_SEND", "64p")
#trial = Utilities.getTrial("ring", "TAU_EACH_SEND", "64p")
input = TrialResult(trial)
messageHeatMap = BuildMessageHeatMap(input)
messageHeatMap.processData()
print "---------------- JPython test script end -------------"
