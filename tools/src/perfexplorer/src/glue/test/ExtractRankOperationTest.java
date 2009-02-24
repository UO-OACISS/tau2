/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.ExtractRankOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

import java.util.List;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class ExtractRankOperationTest extends TestCase {

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.ExtractRankOperation#processData()}.
	 */
	public final void testProcessData() {
		Utilities.setSession("perigtc");
		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
		PerformanceResult result = new TrialResult(trial);
		int thread = 4;
		PerformanceAnalysisOperation operation = new ExtractRankOperation(result, thread);
		List<PerformanceResult> outputs = operation.processData();
		PerformanceResult output = outputs.get(0);
		assertNotNull(output);
		assertEquals(output.getThreads().size(), 1);
		assertEquals(output.getEvents().size(), 42);
		assertEquals(output.getMetrics().size(), 1);
		
		for (String event : output.getEvents()) {
			for (String metric : output.getMetrics()) {
				assertEquals(output.getExclusive(0, event, metric), 
						result.getExclusive(thread, event, metric));
				assertEquals(output.getInclusive(0, event, metric), 
						result.getInclusive(thread, event, metric));
			}
			assertEquals(output.getCalls(0, event), 
					result.getCalls(thread, event));
			assertEquals(output.getSubroutines(0, event), 
					result.getSubroutines(thread, event));
		}
	}
}
