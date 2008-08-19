from glue import PerformanceResult
from glue import PerformanceAnalysisOperation
from glue import ExtractEventOperation
from glue import Utilities
from glue import BasicStatisticsOperation
from glue import DeriveMetricOperation
from glue import MergeTrialsOperation
from glue import TrialMeanResult
from glue import AbstractResult
from glue import DrawGraph
from glue import TopXEvents
from client import PerfExplorerModel
from edu.uoregon.tau.perfdmf import Trial
from java.util import HashSet
from java.util import ArrayList

True = 1
False = 0
config = "proper"
inApp = "sweep3d_gnu"
inExp = "mpi-pdt_2008-08-15_15:34:07"
inTrial = ""

def load():
	print "loading data..."
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	for key in keys:
		print key, parameterMap.get(key)
	config = parameterMap.get("config")
	inApp = parameterMap.get("app")
	inExp = parameterMap.get("exp")
	Utilities.setSession(config)
	trials = Utilities.getTrialsForExperiment(inApp, inExp)
	print "...done."
	return trials

def extractMain(inputs):
	events = ArrayList()
	events.add(inputs.get(0).getMainEvent())

	print "extracting main event..."
	extractor = ExtractEventOperation(inputs, events)
	extracted = extractor.processData()
	print "...done."

	return extracted

def getTop5(inputs):
	print "extracting top 5 events..."
	reducer = TopXEvents(inputs, "Time", AbstractResult.EXCLUSIVE, 5)
	reduced = reducer.processData()
	return reduced

def drawGraph(results):
	print "drawing charts..."
	for metric in results.get(0).getMetrics():
		grapher = DrawGraph(results)
		metrics = HashSet()
		metrics.add(metric)
		grapher.set_metrics(metrics)
		grapher.setLogYAxis(False)
		grapher.setShowZero(False)
		grapher.setTitle(inApp + ": " + inExp + ": " + metric)
		grapher.setSeriesType(DrawGraph.EVENTNAME)
		grapher.setCategoryType(DrawGraph.METADATA)
		grapher.setMetadataField("BUILD:Optimization Level")
		#grapher.setValueType(AbstractResult.INCLUSIVE)
		grapher.setValueType(AbstractResult.EXCLUSIVE)
		grapher.setXAxisLabel("Optimization Level")
		grapher.setYAxisLabel("Inclusive " + metric)
		grapher.processData()
	print "...done."

	return

print "--------------- JPython test script start ------------"

trials = load()
results = ArrayList()
for trial in trials:
	loaded = TrialMeanResult(trial)
	results.add(loaded)

#extracted = extractMain(results)
extracted = getTop5(results)
#extracted = results
drawGraph(extracted)

print "---------------- JPython test script end -------------"
