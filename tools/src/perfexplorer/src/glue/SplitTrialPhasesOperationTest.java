/**
 * 
 */
package glue;

import edu.uoregon.tau.perfdmf.Trial;

import java.util.Hashtable;
import java.util.List;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class SplitTrialPhasesOperationTest extends TestCase {

	/**
	 * Test method for {@link glue.SplitTrialPhasesOperation#processData()}.
	 */
	public final void testProcessData() {
		int sessionid = Utilities.setSession("cqos");
		Trial trial = Utilities.getTrial("simple", "test", "method 1");
		TrialMeanResult result = new TrialMeanResult(trial);
		PerformanceAnalysisOperation operator = new SplitTrialPhasesOperation(result, "Iteration");
		List<PerformanceResult> outputs = operator.processData();
		
		for (PerformanceResult output : outputs) {
			String metric = output.getTimeMetric();
			for (String event : output.getEvents()) {
				System.out.println(event + ": " + output.getInclusive(0, event, metric)/1000000 + " seconds");
			}
			System.out.println(output.getName() + ": Main: " + output.getMainEvent());
			System.out.println(output.getEventMap());
			System.out.println("");
		}
	}

}
