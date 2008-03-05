package glue;

import edu.uoregon.tau.perfdmf.Trial;

import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;

public class ExtractEventOperationTest extends TestCase {

	public final void testProcessData() {
		Utilities.setSession("peri_gtc");
		Trial trial = Utilities.getTrial("GTC", "jacquard", "64");
		PerformanceResult result = new TrialMeanResult(trial);
		String event = result.getMainEvent();
		List<String> events = new ArrayList<String>();
		events.add(event);
		PerformanceAnalysisOperation operation = new ExtractEventOperation(result, events);
		List<PerformanceResult> outputs = operation.processData();
		PerformanceResult output = outputs.get(0);
		assertNotNull(output);
		assertEquals(output.getThreads().size(), 1);
		assertEquals(output.getMetrics().size(), 5);
		assertEquals(output.getEvents().size(), 1);
		
		for (String metric : output.getMetrics()) {
			for (Integer thread : output.getThreads()) {
				assertEquals(output.getExclusive(thread, event, metric), 
						result.getExclusive(thread, event, metric));
				assertEquals(output.getInclusive(thread, event, metric), 
						result.getInclusive(thread, event, metric));
				assertEquals(output.getCalls(thread, event), 
						result.getCalls(thread, event));
				assertEquals(output.getSubroutines(thread, event), 
						result.getSubroutines(thread, event));
			}
		}
	}

}
