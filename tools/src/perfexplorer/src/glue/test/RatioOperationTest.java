package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.BasicStatisticsOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.RatioOperation;
import edu.uoregon.tau.perfexplorer.glue.TopXEvents;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

import java.util.List;

import junit.framework.TestCase;

public class RatioOperationTest extends TestCase {

	public final void testProcessData() {
		Utilities.setSession("perfdmf.demo");
		Trial trial = Utilities.getTrial("sweep3d", "neuronic", "150x150x150 with 16 procs");
		PerformanceResult result = new TrialResult(trial);
		System.out.println("getting top 3...");
		PerformanceAnalysisOperation operation = new TopXEvents(result, result.getTimeMetric(), AbstractResult.EXCLUSIVE, 3);
		List<PerformanceResult> outputs = operation.processData();
		System.out.println("stats...");
		BasicStatisticsOperation statter = new BasicStatisticsOperation(outputs.get(0));
		outputs = statter.processData();
		System.out.println("ratio operation...");
		RatioOperation ratio = new RatioOperation(outputs.get(3), outputs.get(1));
		List<PerformanceResult> ratios = ratio.processData();
		PerformanceResult output = ratios.get(0);
		assertNotNull(output);
		assertEquals(output.getThreads().size(), 1);
		assertEquals(output.getEvents().size(), 3);
		assertEquals(output.getMetrics().size(), 5);
		
		for (Integer thread : output.getThreads()) {
			for (String event : output.getEvents()) {
				for (String metric : output.getMetrics()) {
					assertEquals(output.getExclusive(thread, event, metric), 
							outputs.get(3).getExclusive(thread, event, metric) /
							outputs.get(1).getExclusive(thread, event, metric));
					assertEquals(output.getInclusive(thread, event, metric), 
							outputs.get(3).getInclusive(thread, event, metric) /
							outputs.get(1).getInclusive(thread, event, metric));
					System.out.println(event + ": " + metric + ": EXCLUSIVE: " + outputs.get(1).getExclusive(thread, event, metric) + ", " + output.getExclusive(thread, event, metric));
					System.out.println(event + ": " + metric + ": INCLUSIVE: " + outputs.get(1).getInclusive(thread, event, metric) + ", " + output.getInclusive(thread, event, metric));
				}
				assertEquals(output.getCalls(thread, event), 
						outputs.get(3).getCalls(thread, event) /
						outputs.get(1).getCalls(thread, event));
				assertEquals(output.getSubroutines(thread, event), 
						outputs.get(3).getSubroutines(thread, event) /
						outputs.get(1).getSubroutines(thread, event));
				System.out.println(event + ": CALLS: " + outputs.get(1).getCalls(thread, event) + ", " + output.getCalls(thread, event));
				System.out.println(event + ": SUBROUTINES: " + outputs.get(1).getSubroutines(thread, event) + ", " + output.getSubroutines(thread, event));
			}
		}

	}

}
