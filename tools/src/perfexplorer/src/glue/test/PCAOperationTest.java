/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import java.util.List;

import junit.framework.TestCase;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.DefaultResult;
import edu.uoregon.tau.perfexplorer.glue.PCAOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

/**
 * @author khuck
 *
 */
public class PCAOperationTest extends TestCase {

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.PCAOperation#processData()}.
	 */
	public final void testProcessData() {
		Utilities.setSession("perigtc");
		PerformanceResult result = new DefaultResult();
		result.putExclusive(0, "x", "time", 2.5);
		result.putExclusive(1, "x", "time", 0.5);
		result.putExclusive(2, "x", "time", 2.2);
		result.putExclusive(3, "x", "time", 1.9);
		result.putExclusive(4, "x", "time", 3.1);
		result.putExclusive(5, "x", "time", 2.3);
		result.putExclusive(6, "x", "time", 2.0);
		result.putExclusive(7, "x", "time", 1.0);
		result.putExclusive(8, "x", "time", 1.5);
		result.putExclusive(9, "x", "time", 1.1);
		result.putExclusive(0, "y", "time", 2.4);
		result.putExclusive(1, "y", "time", 0.7);
		result.putExclusive(2, "y", "time", 2.9);
		result.putExclusive(3, "y", "time", 2.2);
		result.putExclusive(4, "y", "time", 3.0);
		result.putExclusive(5, "y", "time", 2.7);
		result.putExclusive(6, "y", "time", 1.6);
		result.putExclusive(7, "y", "time", 1.1);
		result.putExclusive(8, "y", "time", 1.6);
		result.putExclusive(9, "y", "time", 0.9);
		int type = AbstractResult.EXCLUSIVE;
		for (String metric : result.getMetrics()) {
			System.out.println(metric);
			PCAOperation pca = new PCAOperation(result, metric, type);
			pca.setMaxComponents(2);
			List<PerformanceResult> outputs = pca.processData();
			for (PerformanceResult output : outputs) {
				for (Integer thread : output.getThreads()) {
					for (String event : output.getEvents()) {
						System.out.println("\t" + thread + " " + event + " " + output.getDataPoint(thread, event, metric, type));
					}
				}
			}
		}

		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
		result = new TrialResult(trial);
		type = AbstractResult.EXCLUSIVE;
		for (String metric : result.getMetrics()) {
			System.out.println(metric);
/*			PerformanceAnalysisOperation reducer = new TopXEvents(result, metric, type, 3);
			List<PerformanceResult> reduced = reducer.processData(); 
			PCAOperation pca = new PCAOperation(reduced.get(0), metric, type);
*/
			PCAOperation pca = new PCAOperation(result, metric, type);
			pca.setMaxComponents(2);
			List<PerformanceResult> outputs = pca.processData();
			for (PerformanceResult output : outputs) {
				for (Integer thread : output.getThreads()) {
					for (String event : output.getEvents()) {
						System.out.println("\t" + thread + " " + event + " " + output.getDataPoint(thread, event, metric, type));
					}
				}
			}
			
			// now, do k-means clustering on the result.  then, get the memberships,
			// and calcluate new cluster means based on the memberships.  got that?
		}
		
}

}
