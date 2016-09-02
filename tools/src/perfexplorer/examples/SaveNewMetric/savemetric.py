from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0


def loadFromDB():
	Utilities.setSession("postgres-test")
	trial = Utilities.getTrialByName("threads").get(0)
	print trial.getName()
	return trial

def derive(trial):	
#	deriver = ScaleMetricOperation(trial, 2.0, "TIME", ScaleMetricOperation.MULTIPLY)
	firstMetric = "TIME"
	secondMetric = "PAPI_L1_DCM"
	deriver = DeriveMetricOperation(trial,secondMetric,firstMetric,DeriveMetricOperation.DIVIDE)
	tmp = deriver.processData()
	return tmp


def save(trial):
	saver = SaveResultOperation(trial)
	saver.setForceOverwrite(False)
#	saver.setDataSource(trial.getDataSource)
	saver.processData()

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	trial = loadFromDB()

	# create a derived metric
	derived = derive(TrialResult(trial))
	print derived.get(0).getMetrics()
#	Utilities.saveMetric(trial,derived.get(0).getMetrics())
	save(derived)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
