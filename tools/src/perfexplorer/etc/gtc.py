from client import ScriptFacade
from common import TransformationType
from common import AnalysisType
from common import EngineType

def TotalExecutionTime(pe):
	pe.setApplication("GTC")
	pe.setMetricName("Time")
	pe.setExperiment("ocracoke-O2")
	pe.addExperiment("ocracoke-O3")
	pe.addExperiment("ocracoke-O4")
	pe.addExperiment("ocracoke-O5")
	pe.setChartTitle("GTC Total Execution Time")
	pe.setChartSeriesName("experiment.name")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Total Time (seconds)")
	pe.setChartMainEventOnly(1);
	pe.setChartLogYAxis(0);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.doGeneralChart()

def Speedup(pe):
	pe.setApplication("GTC")
	pe.setMetricName("Time")
	pe.setExperiment("ocracoke-O2")
	pe.addExperiment("ocracoke-O3")
	pe.addExperiment("ocracoke-O4")
	pe.addExperiment("ocracoke-O5")
	pe.setChartTitle("GTC Speedup, Weak Scaling")
	pe.setChartSeriesName("experiment.name")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Speedup")
	pe.setChartMainEventOnly(1);
	pe.setChartLogYAxis(0);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.setChartScalability(1)
	pe.setChartConstantProblem(1)
	pe.doGeneralChart()

def SpeedupByEvent(pe):
	pe.setApplication("GTC")
	pe.setMetricName("Time")
	pe.addExperiment("ocracoke-O5")
	pe.setChartTitle("GTC Total Execution Time")
	pe.setChartSeriesName("interval_event.name")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Total Time (seconds)")
	pe.setChartMainEventOnly(0);
	pe.setChartLogYAxis(0);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.setChartScalability(1)
	pe.setChartConstantProblem(1)
	pe.doGeneralChart()

def Simple(pe):
	pe.setApplication("GTC")
	pe.setMetricName("Time")
	pe.setExperiment("ocracoke-O2")
	pe.setChartTitle("GTC Total Execution Time")
	pe.setChartSeriesName("experiment.name")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Total Time (seconds)")
	pe.setChartMainEventOnly(1);
	pe.setChartLogYAxis(0);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.setChartConstantProblem(1)
	pe.doGeneralChart()

def XMLtest(pe):
	pe.setApplication("GTC")
	pe.setExperiment("ocracoke-O2")
	pe.addExperiment("Jacquard")
	pe.setMetricName("%TIME%")
	pe.setChartTitle("GTC Total Execution Time")
	pe.setChartSeriesName("temp_xml_metadata.metadata_value")
	pe.setChartMetadataFieldName("TAU Architecture")
	#pe.setChartMetadataFieldValue("bgl")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Total Time (seconds)")
	pe.setChartMainEventOnly(1);
	pe.setChartLogYAxis(0);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.setChartConstantProblem(1)
	pe.doGeneralChart()

def XMLharder(pe):
	pe.setApplication("GTC")
	pe.setExperiment("ocracoke-O2")
	pe.addExperiment("Jacquard")
	pe.setMetricName("%TIME%")
	pe.setChartTitle("GTC Total Execution Time")
	pe.setChartXAxisName("temp_xml_metadata.metadata_value", "TAU Architecture")
	pe.setChartMetadataFieldName("TAU Architecture")
	#pe.setChartMetadataFieldValue("bgl")
	pe.setChartSeriesName("trial.node_count * trial.contexts_per_node * trial.threads_per_context")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Total Time (seconds)")
	pe.setChartMainEventOnly(1);
	pe.setChartLogYAxis(0);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.setChartConstantProblem(1)
	pe.doGeneralChart()

def CompilerByEvent(pe):
	pe.setApplication("GTC compiler options")
	# pe.setExperiment("ocracoke-noinline")
	pe.setMetricName("Time")
	pe.setChartTitle("GTC Compiler options by event")
	pe.setDimensionReduction(TransformationType.OVER_X_PERCENT, "1")
	pe.setChartSeriesName("interval_event.name")
	pe.setChartXAxisName("temp_xml_metadata.metadata_value", "Compiler Options")
	pe.setChartMetadataFieldName("Compiler Options")
	pe.setChartYAxisName("avg(interval_mean_summary.exclusive)", "Total Time (seconds)")
	pe.setChartMainEventOnly(0);
	pe.setChartLogYAxis(0);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.setChartScalability(0)
	pe.setChartConstantProblem(0)
	pe.doGeneralChart()

print "--------------- JPython test script start ------------"

pe = ScriptFacade()
# XMLtest(pe)
# XMLharder(pe)
# TotalExecutionTime(pe)
# Speedup(pe)
# SpeedupByEvent(pe)
CompilerByEvent(pe)
# Simple(pe)

# pe.exit()

print "---------------- JPython test script end -------------"
