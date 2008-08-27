package glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import glue.AbstractResult;
import glue.CorrelationOperation;
import glue.CorrelationResult;
import glue.ExtractEventOperation;
import glue.PerformanceAnalysisOperation;
import glue.PerformanceResult;
import glue.TrialResult;
import glue.Utilities;

import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;

public class CorrelationOperationTest extends TestCase {

	public final void testProcessData() {
		Utilities.setSession("perfdmf_uploaded");
		Trial trial = Utilities.getTrial("NPB", "LU.W.4-metadata", "p1");
		PerformanceResult result = new TrialResult(trial);
		List<String> events = new ArrayList<String>();
		events.add("MPI_Init()");
		events.add("rhs");
		PerformanceAnalysisOperation reducer = new ExtractEventOperation(result, events);
		List<PerformanceResult> reduced = reducer.processData(); 
		PerformanceAnalysisOperation correlation = new CorrelationOperation(reduced.get(0));
		List<PerformanceResult> outputs = correlation.processData();
		for (PerformanceResult output : outputs) {
			for (String event : output.getEvents()) {
				for (String metric : output.getMetrics()) {
//					for (Integer type : AbstractResult.getTypes()) {
					Integer type = AbstractResult.EXCLUSIVE;
						for (Integer thread : output.getThreads()) {
//					Integer thread = CorrelationResult.CORRELATION;
							if (!event.equals(metric + ":" + AbstractResult.typeToString(type))) {
								System.out.println(event + " " + CorrelationResult.typeToString(thread) + " " + metric + ":" + AbstractResult.typeToString(type) + " " + output.getDataPoint(thread, event, metric, type.intValue()));
							}
						}
//					}
				}
			}
		}
	}

}
