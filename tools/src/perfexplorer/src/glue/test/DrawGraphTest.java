/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.DrawGraph;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

import java.util.HashSet;
import java.util.Set;
import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class DrawGraphTest extends TestCase {

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.DrawGraph#processData()}.
	 */
	public final void testProcessData() {
		Utilities.getClient();
		Utilities.setSession("perigtc");
		Trial trial = Utilities.getTrial("GTC", "jacquard", "64");
		PerformanceResult result = new TrialResult(trial);
		String event = result.getMainEvent();
		Set<String> events = new HashSet<String>();
		events.add(event);
		DrawGraph operation = new DrawGraph(result);
		operation.set_events(events);
		operation.setTitle(event);
		operation.setYAxisLabel("process ID");
		operation.setXAxisLabel("value");
		operation.processData();
	}

}
