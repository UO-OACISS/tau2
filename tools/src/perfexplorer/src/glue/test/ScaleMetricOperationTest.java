/**
 * 
 */
package glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import glue.PerformanceResult;
import glue.ScaleMetricOperation;
import glue.TrialResult;
import glue.Utilities;
import junit.framework.TestCase;


/**
 * @author khuck
 *
 */
public class ScaleMetricOperationTest extends TestCase {

	/**
	 * Test method for {@link glue.ScaleMetricOperation#processData()}.
	 */
	public final void testProcessData() {
	    Utilities.setSession("perfdmf_test");
		Trial trial = Utilities.getTrial("gtc_bench", "superscaling.jaguar", "64");
		PerformanceResult result = new TrialResult(trial);
		Double value = 10.0;
	    ScaleMetricOperation derive = new ScaleMetricOperation(result, "P_WALL_CLOCK_TIME", 
				value, ScaleMetricOperation.DIVIDE);
	    PerformanceResult output = derive.processData().get(0);
		for (String metric : output.getMetrics()) {
			System.out.println(metric);
		}
		for (String event : result.getEvents()) {
			for (Integer thread : result.getThreads()) {
				double tmp = result.getInclusive(thread, event, "P_WALL_CLOCK_TIME");
				assertEquals((tmp == 0.0 ? 0.0 : tmp / value),
					output.getInclusive(thread, event, "(P_WALL_CLOCK_TIME/" + value.toString() + ")"));
				tmp = result.getExclusive(thread, event, "P_WALL_CLOCK_TIME");
				assertEquals((tmp == 0.0 ? 0.0 : tmp / value),
					output.getExclusive(thread, event, "(P_WALL_CLOCK_TIME/" + value.toString() + ")"));
			}
		}
	}

}
