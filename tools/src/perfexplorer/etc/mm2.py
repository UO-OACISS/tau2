from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.common import TransformationType
from edu.uoregon.tau.perfexplorer.common import AnalysisType

def TotalExecutionTime(pe):
	pe.resetChartDefaults();
	pe.setApplication("mm2")
	pe.setMetricName("PAPI_L2_TCM")
	pe.setChartTitle("Matrix Multiply Total Cache Misses per problem size")
	pe.setChartSeriesName("experiment.name")
	pe.setChartXAxisName("trial.name", "Matrix dimension")
	pe.setChartYAxisName("avg(interval_mean_summary.inclusive)", "Total Cache misses (million)")
	pe.setChartMainEventOnly(1);
	pe.setChartEventNoCallPath(1)
	pe.doGeneralChart()

def TotalExecutionTimeByEvent(pe):
	pe.resetChartDefaults();
	pe.setApplication("mm2")
	pe.setMetricName("PAPI_L2_TCM")
	pe.setChartTitle("Matrix Multiply Total Cache Misses per problem size")
	#pe.setDimensionReduction(TransformationType.OVER_X_PERCENT, "1")
	pe.setChartSeriesName("interval_event.name")
	pe.setChartXAxisName("trial.name", "Matrix dimension")
	pe.setChartYAxisName("avg(interval_mean_summary.exclusive)", "Total Cache misses (million)")
	pe.setChartEventNoCallPath(1)
	pe.doGeneralChart()

print "--------------- JPython test script start ------------"

pe = ScriptFacade()
TotalExecutionTime(pe)
TotalExecutionTimeByEvent(pe)

# pe.exit()

print "---------------- JPython test script end -------------"
