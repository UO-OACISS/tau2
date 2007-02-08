from client import ScriptFacade
from common import TransformationType
from common import AnalysisType
from common import EngineType

print "--------------- JPython test script start ------------"

pe = ScriptFacade()

pe.setApplication("MILC")
pe.setExperiment("Jacquard scalability")
pe.setMetricName("GET_TIME_OF_DAY")
pe.setChartTitle("MILC Scalability by Event")
pe.setDimensionReduction(TransformationType.OVER_X_PERCENT, "2")
pe.setChartSeriesName("interval_event.name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.exclusive)", "Exclusive Time (seconds)")
pe.setChartMainEventOnly(0);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("MILC")
pe.setExperiment("Jacquard scalability")
pe.setChartTitle("MILC testing")
pe.setChartSeriesName("metric.name")
pe.setChartXAxisName("trial.name", "Trials")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(1);
pe.setChartLogAxis(0);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("MILC")
pe.setExperiment("Jacquard scalability")
pe.setMetricName("GET_TIME_OF_DAY")
pe.setChartTitle("MILC Scalability - HPMtoolkit data")
pe.setChartSeriesName("SUBSTR(trial.name, 0, 4)")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(1);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("MILC")
pe.setExperiment("Jacquard scalability")
pe.setMetricName("GET_TIME_OF_DAY")
pe.setChartTitle("MILC Scalability")
pe.setChartSeriesName("experiment.name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(1);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("MILC")
pe.setMetricName("GET_TIME_OF_DAY")
pe.setChartTitle("MILC Scalability")
pe.setChartSeriesName("experiment.name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(1);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("MILC")
pe.setExperiment("Jacquard scalability")
pe.setMetricName("GET_TIME_OF_DAY")
pe.setDimensionReduction(TransformationType.OVER_X_PERCENT, "2")
pe.setChartTitle("WRF Scalability")
pe.setChartSeriesName("interval_event.group_name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("sum(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(0);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("MILC")
pe.setExperiment("Jacquard scalability")
pe.setMetricName("GET_TIME_OF_DAY")
pe.setChartTitle("MILC Scalability")
pe.setChartSeriesName("interval_event.group_name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("sum(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(0);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

# pe.exit()

print "---------------- JPython test script end -------------"
