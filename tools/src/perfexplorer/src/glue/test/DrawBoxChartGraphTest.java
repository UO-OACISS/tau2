/**
 * 
 */
package glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import glue.AbstractResult;
import glue.DrawBoxChartGraph;
import glue.DrawGraph;
import glue.ExtractEventOperation;
import glue.ExtractMetricOperation;
import glue.PerformanceAnalysisOperation;
import glue.PerformanceResult;
import glue.TrialResult;
import glue.Utilities;

import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class DrawBoxChartGraphTest extends TestCase {

	/**
	 * Test method for {@link glue.DrawGraph#processData()}.
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
	        if (event.startsWith("Iteration   1") && event.contains("CHARGEI")) {
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

		DrawGraph grapher = new DrawBoxChartGraph(extracted2);
        grapher.setTitle("CHARGEI");
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
