from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

True = 1
False = 0

def loadFile(fileName):
	# load the trial
	files = []
	files.append(fileName)
	input = DataSourceResult(DataSourceResult.PPK, files, False)
	return input

def loadFromFiles():
	inputs = ArrayList()
	inputs.add(loadFile("2.ppk"))
	inputs.add(loadFile("4.ppk"))
	inputs.add(loadFile("6.ppk"))
	inputs.add(loadFile("8.ppk"))
	return inputs

def loadDB(app, exp, trial):
	trial = Utilities.getTrial(app, exp, trial)
	input = TrialMeanResult(trial)
	return input

def loadFromDB():
	Utilities.setSession("your_database_configuration")
	inputs.add(loadDB("application", "experiment", "trial1"))
	inputs.add(loadDB("application", "experiment", "trial2"))
	inputs.add(loadDB("application", "experiment", "trial3"))
	inputs.add(loadDB("application", "experiment", "trial4"))
	return inputs

def extract(inputs):
	# extract the event of interest
	events = ArrayList()
	# change this to zoneLimitedGradient(PileOfScalars) as necessary
	events.add("MPI_Send()")
	events.add("MPI_Init()")
	extractor = ExtractEventOperation(inputs, events)
	extracted = extractor.processData()
	return extracted

def doStats(inputs):
	statMaker = BasicStatisticsOperation(inputs, False)
	stats = statMaker.processData()
	# stddevs = stats.get(BasicStatisticsOperation.STDDEV)
	means = stats.get(BasicStatisticsOperation.MEAN)
	# totals = stats.get(BasicStatisticsOperation.TOTAL)
	# maxs = stats.get(BasicStatisticsOperation.MAX)
	# mins = stats.get(BasicStatisticsOperation.MIN)
	return means

def export(results):
	# change this to P_WALL_CLOCK_TIME as necessary
	metric = "Time"
	f = open('export.csv', 'w')
	f.write('Trial,Metric,Event,Threads,Exclusive\n')
	for r in results:
		means = doStats(r)
		for event in r.getEvents():
			f.write('\"')
			f.write(r.getName())
			f.write('\",\"')
			f.write(metric)
			f.write('\",\"')
			f.write(event)
			f.write('\",\"')
			f.write(str(r.getOriginalThreads()))
			f.write('\",\"')
			f.write(str(means.getExclusive(0,event,metric)))
			f.write('\"\n')
	f.close()
	print "Data written to export.csv"

def main():
	print "--------------- JPython test script start ------------"
	# load the data
	inputs = loadFromFiles()

	# extract some events, if you like
	extracted = extract(inputs)

	export(extracted)
	print "---------------- JPython test script end -------------"

if __name__ == "__main__":
	main()
