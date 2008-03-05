/**
 * 
 */
package glue;

import edu.uoregon.tau.perfdmf.Trial;

import java.util.HashSet;
import java.util.Set;

import javax.swing.JFrame;

import client.PerfExplorerClient;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class DrawGraphTest extends TestCase {

	/**
	 * Test method for {@link glue.DrawGraph#processData()}.
	 */
	public final void testProcessData() {
		Utilities.getClient();
		Utilities.setSession("peri_gtc");
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
