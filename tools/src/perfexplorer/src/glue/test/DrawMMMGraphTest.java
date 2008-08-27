/**
 * 
 */
package glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import glue.AbstractResult;
import glue.BasicStatisticsOperation;
import glue.DrawGraph;
import glue.DrawMMMGraph;
import glue.ExtractEventOperation;
import glue.ExtractMetricOperation;
import glue.PerformanceAnalysisOperation;
import glue.PerformanceResult;
import glue.TrialResult;
import glue.Utilities;

import java.util.List;
import java.util.ArrayList;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class DrawMMMGraphTest extends TestCase {

	/**
	 * Test method for {@link glue.DrawMMMGraph#processData()}.
	 */
	public final void testProcessData() {
		Utilities.getClient();
		Utilities.setSession("perfdmf_test");
		// get the data
		Trial trial = Utilities.getTrial("gtc_bench", "jaguar.phases", "64");
		PerformanceResult result = new TrialResult(trial);
		
		// extract the phases
	    List<String>events = new ArrayList<String>();
	    for (String event : result.getEvents()) {
	        if (event.startsWith("Iteration") && event.contains("CHARGEI")) {
	            events.add(event);
	        }
	    }
	    PerformanceAnalysisOperation extractor = new ExtractEventOperation(result, events);
		PerformanceResult extracted = extractor.processData().get(0);
		
		// extract the metric
	    List<String>metrics = new ArrayList<String>();
        metrics.add("P_WALL_CLOCK_TIME");
	    PerformanceAnalysisOperation extractor2 = new ExtractMetricOperation(extracted, metrics);
		PerformanceResult extracted2 = extractor2.processData().get(0);

		// get the statistics
		PerformanceAnalysisOperation doStats = new BasicStatisticsOperation(extracted2, false);
		List<PerformanceResult> stats = doStats.processData();
/*		for (String event : stats.get(0).getEvents()) {
			for (String metric : stats.get(0).getMetrics()) {
				for (Integer thread : stats.get(0).getThreads()) {
					System.out.println(thread + " " + event + " " + stats.get(3).getInclusive(thread, event, metric));
				}
			}
		}
*/
		DrawGraph grapher = new DrawMMMGraph(stats);
        grapher.setTitle("CHARGEI");
        grapher.setSeriesType(DrawGraph.TRIALNAME);
        grapher.setCategoryType(DrawGraph.EVENTNAME);
        grapher.setValueType(AbstractResult.INCLUSIVE);
        grapher.processData();
		try {
			java.lang.Thread.sleep(600000);
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
	}

}
