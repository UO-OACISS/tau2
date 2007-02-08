from client import ScriptFacade
from common import TransformationType
from common import AnalysisType
from common import EngineType

def Chart1(pe):
	pe.setApplication("MILC")
	pe.setExperiment("Jacquard scalability")
	pe.setMetricName("GET_TIME_OF_DAY")
	pe.setChartTitle("MILC Scalability by Event")
	pe.setDimensionReduction(TransformationType.OVER_X_PERCENT, "2")
	pe.setChartSeriesName("interval_event.name")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("avg(interval_mean_summary.exclusive)", "Exclusive Time (seconds)")
	pe.setChartMainEventOnly(0);
	pe.setChartLogYAxis(1);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.doGeneralChart()
	return

def Chart2(pe):
	pe.setApplication("MILC")
	pe.setExperiment("Jacquard scalability")
	pe.setChartTitle("MILC testing")
	pe.setChartSeriesName("metric.name")
	pe.setChartXAxisName("trial.name", "Trials")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
	pe.setChartMainEventOnly(1);
	pe.setChartLogYAxis(0);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.doGeneralChart()

def Chart3(pe):
	pe.setApplication("MILC")
	pe.setExperiment("Jacquard scalability")
	pe.setMetricName("GET_TIME_OF_DAY")
	pe.setChartTitle("MILC Scalability - HPMtoolkit data")
	pe.setChartSeriesName("SUBSTR(trial.name, 0, 4)")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
	pe.setChartMainEventOnly(1);
	pe.setChartLogYAxis(1);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.doGeneralChart()

def Chart4(pe):
	pe.setApplication("MILC")
	pe.setExperiment("Jacquard scalability")
	pe.setMetricName("GET_TIME_OF_DAY")
	pe.setChartTitle("MILC Scalability")
	pe.setChartSeriesName("experiment.name")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
	pe.setChartMainEventOnly(1);
	pe.setChartLogYAxis(1);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.doGeneralChart()

def TotalExecutionTime(pe):
	pe.setApplication("MILC")
	pe.setMetricName("GET_TIME_OF_DAY")
	pe.setChartTitle("MILC Scalability Total Execution Time")
	pe.setChartSeriesName("experiment.name")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Total Time (seconds)")
	pe.setChartMainEventOnly(1);
	pe.setChartLogYAxis(0);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.doGeneralChart()

def ComparingGroups(pe):
	pe.setApplication("MILC")
	pe.setExperiment("Jacquard scalability")
	pe.setMetricName("GET_TIME_OF_DAY")
	pe.setChartTitle("MILC Scalability by Event Group")
	pe.setChartSeriesName("interval_event.group_name")
	pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
	pe.setChartYAxisName("sum(interval_mean_summary.exclusive)", "Exclusive Time (seconds)")
	pe.setChartMainEventOnly(0);
	pe.setChartLogYAxis(1);
	pe.setChartEventNoCallPath(1)
	pe.setChartEventExclusive100(0)
	pe.doGeneralChart()

def Speedup(pe):
	pe.setApplication("MILC")
	pe.setMetricName("GET_TIME_OF_DAY")
	pe.setChartTitle("MILC Speedup, Weak Scaling")
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


print "--------------- JPython test script start ------------"

pe = ScriptFacade()
TotalExecutionTime(pe)
Speedup(pe)

# pe.exit()

print "---------------- JPython test script end -------------"
