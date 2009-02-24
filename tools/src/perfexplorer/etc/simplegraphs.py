from edu.uoregon.tau.perfexplorer.client import ScriptFacade
from edu.uoregon.tau.perfexplorer.common import TransformationType
from edu.uoregon.tau.perfexplorer.common import AnalysisType
from edu.uoregon.tau.perfexplorer.common import EngineType

def TotalExecutionTime1(pe, metric):
	pe.resetChartDefaults();
	pe.setApplication("simple_papi-DSTATIC_MATRIX")
	pe.setMetricName(metric)
	pe.setEventName("multiply void (void)")
	pe.setChartTitle("Static matrix methods, " + metric)
	pe.setChartSeriesName("trial.name")
	pe.setChartXAxisName("experiment.name", "Compiler optimization")
	pe.setChartYAxisName("avg(interval_mean_summary.exclusive)", metric + " (million)")
	pe.doGeneralChart()

def TotalExecutionTime2(pe, metric):
	pe.resetChartDefaults();
	pe.setApplication("simple_papi-DDYNAMIC_MATRIX")
	pe.setMetricName(metric)
	pe.setEventName("multiply void (void)")
	pe.setChartTitle("Dynamic matrix methods, " + metric)
	pe.setChartSeriesName("trial.name")
	pe.setChartXAxisName("experiment.name", "Compiler optimization")
	pe.setChartYAxisName("avg(interval_mean_summary.exclusive)", metric + " (million)")
	pe.doGeneralChart()

print "--------------- JPython test script start ------------"

pe = ScriptFacade()
TotalExecutionTime1(pe, "P_WALL_CLOCK_TIME")
TotalExecutionTime1(pe, "PAPI_FP_INS")
TotalExecutionTime1(pe, "PAPI_L1_TCM")
TotalExecutionTime2(pe, "P_WALL_CLOCK_TIME")
TotalExecutionTime2(pe, "PAPI_FP_INS")
TotalExecutionTime2(pe, "PAPI_L1_TCM")

print "---------------- JPython test script end -------------"


