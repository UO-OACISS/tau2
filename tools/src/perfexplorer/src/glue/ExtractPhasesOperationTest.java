/**
 * 
 */
package glue;

import edu.uoregon.tau.perfdmf.Trial;

import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class ExtractPhasesOperationTest extends TestCase {

	/**
	 * Test method for {@link glue.ExtractPhasesOperation#processData()}.
	 */
	public final void testProcessData() {
		Utilities.setSession("perfdmf_test");
		Trial trial = Utilities.getTrial("gtc_bench", "jaguar.longrun", "64.first");
		PerformanceResult result = new TrialMeanResult(trial);
		PerformanceAnalysisOperation operation = new ExtractPhasesOperation(result, "Iteration");
		List<PerformanceResult> outputs = operation.processData();
		PerformanceResult output = outputs.get(0);
		assertNotNull(output);
		assertEquals(output.getThreads().size(), 50);
		assertEquals(output.getEvents().size(), 2);
		assertEquals(output.getMetrics().size(), 2);
		
		for (String event : output.getEvents()) {
			for (String metric : output.getMetrics()) {
				for (Integer thread : output.getThreads()) {
					if (event.contains("measurement")) {
						System.out.println(metric + " " + thread + " " + output.getInclusive(thread, event, metric));
					}
				}
			}
		}
	}
}
