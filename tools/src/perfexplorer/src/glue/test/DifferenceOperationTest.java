package edu.uoregon.tau.perfexplorer.glue.test;

import java.util.List;

import junit.framework.TestCase;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.DifferenceOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

public class DifferenceOperationTest extends TestCase {

	public final void testProcessData() {
		Utilities.setSession("perigtc");
		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
		Trial trial2 = Utilities.getTrial("GTC", "ocracoke-O2", "128");
		PerformanceResult result = new TrialResult(trial);
		PerformanceResult result2 = new TrialResult(trial2);
		PerformanceAnalysisOperation operation = new DifferenceOperation(result);
		operation.addInput(result2);
		List<PerformanceResult> outputs = operation.processData();
		PerformanceResult output = outputs.get(0);
		assertNotNull(output);
		assertEquals(output.getThreads().size(), 128);
		assertEquals(output.getEvents().size(), 42);
		assertEquals(output.getMetrics().size(), 1);
		
		for (Integer thread : output.getThreads()) {
			for (String event : output.getEvents()) {
				for (String metric : output.getMetrics()) {
					assertEquals(output.getExclusive(thread, event, metric), 
							result.getExclusive(thread, event, metric) -
							result2.getExclusive(thread, event, metric));
					assertEquals(output.getInclusive(thread, event, metric), 
							result.getInclusive(thread, event, metric) -
							result2.getInclusive(thread, event, metric));
				}
				assertEquals(output.getCalls(thread, event), 
						result.getCalls(thread, event) -
						result2.getCalls(thread, event));
				assertEquals(output.getSubroutines(thread, event), 
						result.getSubroutines(thread, event) -
						result2.getSubroutines(thread, event));
			}
		}

	}

}
