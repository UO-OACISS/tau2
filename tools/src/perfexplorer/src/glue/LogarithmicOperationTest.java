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
public class LogarithmicOperationTest extends TestCase {

	/**
	 * Test method for {@link glue.LogarithmicOperation#processData()}.
	 */
	public void testProcessData() {
		Utilities.setSession("perfdmf_test");
		List<PerformanceResult> trials = new ArrayList<PerformanceResult>();
		Trial trial = Utilities.getTrial("Meng-Shiou", "luciferin", "luciferin-1.1");
		trials.add(new TrialMeanResult(trial));
		trial = Utilities.getTrial("Meng-Shiou", "luciferin", "luciferin-2.1");
		trials.add(new TrialMeanResult(trial));
		trial = Utilities.getTrial("Meng-Shiou", "luciferin", "luciferin-4.1");
		trials.add(new TrialMeanResult(trial));
		
		LogarithmicOperation loggy = new LogarithmicOperation(trials);
		double base = 2.0;
		loggy.setBase(base);
		List<PerformanceResult> outlogs = loggy.processData();
		
		for (int i = 0 ; i < trials.size() ; i++) {
			PerformanceResult output = outlogs.get(i);
			PerformanceResult input = trials.get(i);
			for (Integer thread : output.getThreads()) {
				for (String event : output.getEvents()) {
					for (String metric : output.getMetrics()) {
						assertEquals(
								Math.log(input.getExclusive(thread, event, metric))/Math.log(base), 
								output.getExclusive(thread, event, metric));
						assertEquals(
								Math.log(input.getInclusive(thread, event, metric))/Math.log(base), 
								output.getInclusive(thread, event, metric));
					}
					assertEquals(input.getCalls(thread, event), 
							output.getCalls(thread, event));
					assertEquals(input.getSubroutines(thread, event), 
							output.getSubroutines(thread, event));
				}
			}
		}
	}

}
