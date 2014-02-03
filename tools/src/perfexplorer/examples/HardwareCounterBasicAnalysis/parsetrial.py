from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfexplorer.glue import *
from java.util import *

def parsetrial(inputData):
	Utilities.setSession("local")
	files = []
	files.append(inputData)
	print "Parsing files:", files, "..."
	trial = DataSourceResult(DataSourceResult.PPK, files, False)
	trial.setIgnoreWarnings(True)
	print "Computing stats..."
	stats = BasicStatisticsOperation(trial)
	mean = stats.processData().get(BasicStatisticsOperation.MEAN)
	extractor = ExtractNonCallpathEventOperation(mean)
	extracted = extractor.processData().get(0)
	return extracted

