from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0

def deleteFromDB():
	Utilities.setSession("super_test")
	metadata = HashMap()
	conjoin = " and "
	# here are the conditions for the selection
	metadata.put("Application"," = 'MPAS-Ocean'")
	metadata.put("Experiment"," = 'ocean_QU_15km_eos_scaling'")
	trials = Utilities.getTrialsFromMetadata(metadata, conjoin)
	for trial in trials:
		try:
			Utilities.deleteTrial(trial)
		except:
			continue

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	inputs = deleteFromDB()
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()
