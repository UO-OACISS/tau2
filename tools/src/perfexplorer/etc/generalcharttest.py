from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.common import TransformationType
from edu.uoregon.tau.perfexplorer.common import AnalysisType

print "--------------- JPython test script start ------------"

pe = ScriptFacade()

pe.setApplication("gyro.B1-std")
pe.setExperiment("B1-std-nl2.cheetah.affnosng")
pe.setMetricName("WALL_CLOCK_TIME")
pe.setChartTitle("GYRO Scalability by Event")
pe.setChartSeriesName("interval_event.name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.exclusive)", "Exclusive Time (seconds)")
pe.setChartMainEventOnly(0);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("gyro.B1-std.HPM")
pe.setExperiment("HPM016")
pe.setChartTitle("GYRO testing")
pe.setChartSeriesName("metric.name")
pe.setChartXAxisName("trial.name", "Trials")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(1);
pe.setChartLogAxis(0);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("gyro.B1-std.HPM")
pe.setMetricName("Time")
pe.setChartTitle("GYRO Scalability - HPMtoolkit data")
pe.setChartSeriesName("SUBSTR(trial.name, 0, 4)")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(1);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("gyro.B1-std")
pe.setExperiment("B1-std-nl2.cheetah.affnosng")
pe.addExperiment("B1-std-nl2.cheetah.affsng")
pe.addExperiment("B1-std-nl2.cheetah.noaffnosng")
pe.setMetricName("WALL_CLOCK_TIME")
pe.setChartTitle("GYRO Scalability")
pe.setChartSeriesName("experiment.name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(1);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("gyro.B1-std")
pe.setMetricName("WALL_CLOCK_TIME")
pe.setChartTitle("GYRO Scalability")
pe.setChartSeriesName("experiment.name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(1);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("WRF")
pe.addExperiment("MCR scalability")
pe.setDimensionReduction(TransformationType.OVER_X_PERCENT, "2")
pe.setMetricName("Time")
pe.setChartTitle("WRF Scalability")
pe.setChartSeriesName("interval_event.group_name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("sum(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(0);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("gyro.B1-std")
pe.addExperiment("B1-std-nl2.cheetah.noaffnosng")
pe.setDimensionReduction(TransformationType.NONE, "2")
pe.setMetricName("WALL_CLOCK_TIME")
pe.setChartTitle("GYRO Scalability")
pe.setChartSeriesName("interval_event.group_name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("sum(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(0);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.doGeneralChart()

pe.setApplication("gyro.B1-std")
pe.addExperiment("B1-std-nl2.cheetah.noaffnosng")
pe.setMetricName("WALL_CLOCK_TIME")
pe.setChartTitle("GYRO Scalability by phase")
pe.setEventName("Iteration 0")
pe.addEventName("Iteration 1")
pe.addEventName("Iteration 2")
pe.addEventName("Iteration 3")
pe.addEventName("Iteration 4")
pe.addEventName("Iteration 5")
pe.addEventName("Iteration 6")
pe.addEventName("Iteration 7")
pe.addEventName("Iteration 8")
pe.addEventName("Iteration 9")
pe.setChartSeriesName("interval_event.name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(0);
pe.setChartLogAxis(1);
pe.setChartEventNoCallPath(0)
pe.setChartEventExclusive100(0)
pe.setChartScalability(0)
pe.setConstantProblem(1)
pe.doGeneralChart()

pe.setApplication("gyro.B1-std")
pe.addExperiment("B1-std-nl2.cheetah.noaffnosng")
pe.setMetricName("WALL_CLOCK_TIME")
pe.setChartTitle("GYRO Scalability")
pe.setChartSeriesName("experiment.name")
pe.setChartXAxisName("trial.node_count * trial.contexts_per_node * trial.threads_per_context", "Threads of Execution")
pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Inclusive Time (seconds)")
pe.setChartMainEventOnly(1);
pe.setChartLogAxis(0);
pe.setChartEventNoCallPath(1)
pe.setChartEventExclusive100(0)
pe.setChartScalability(1)
pe.setConstantProblem(1)
pe.doGeneralChart()

# pe.exit()

print "---------------- JPython test script end -------------"
