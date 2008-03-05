/**
 * 
 */
package glue;

import edu.uoregon.tau.perfdmf.Trial;

import java.util.List;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class KMeansOperationTest extends TestCase {

	/**
	 * Test method for {@link glue.KMeansOperation#processData()}.
	 */
	public final void testProcessData() {
//		Utilities.setSession("peri_gtc");
//		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
		Utilities.setSession("local");
		Trial trial = Utilities.getTrial("sweep3d", "jaguar", "256");
		PerformanceResult result = new TrialResult(trial);
		int type = AbstractResult.EXCLUSIVE;
		boolean doingMean = false;
		for (String metric : result.getMetrics()) {
			for (int i = 2 ; i <= 10 ; i++) {
				PerformanceAnalysisOperation reducer = new TopXPercentEvents(result, metric, type, 2.0);
				List<PerformanceResult> reduced = reducer.processData(); 
				PerformanceAnalysisOperation kmeans = new KMeansOperation(reduced.get(0), metric, type, i);
				List<PerformanceResult> outputs = kmeans.processData();
	/*			for (PerformanceResult output : outputs) {
					for (Integer thread : output.getThreads()) {
						for (String event : output.getEvents()) {
							if (event.startsWith("CHARGEI") && thread == 1)
								System.out.println(thread + " " + event + " " + output.getDataPoint(thread, event, metric, type));
						}
					}
				}
*/			}
		}
	}

}
