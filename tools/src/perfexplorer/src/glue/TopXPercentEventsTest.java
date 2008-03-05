package glue;

import edu.uoregon.tau.perfdmf.Trial;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import junit.framework.TestCase;

public class TopXPercentEventsTest extends TestCase {

	public TopXPercentEventsTest(String arg0) {
		super(arg0);
	}

	public final void testProcessData() {
		Utilities.setSession("peri_gtc");
		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
//		PerformanceResult result = new TrialResult(trial);
		boolean doingMean = true;
		PerformanceResult result = new TrialMeanResult(trial);
//		PerformanceResult result = new TrialTotalResult(trial);
		for (String metric : result.getMetrics()) {
			System.out.println("\t--- INCLUSIVE --");
			PerformanceAnalysisOperation top2percent = new TopXPercentEvents(result, metric, AbstractResult.INCLUSIVE, 2.0);
			List<PerformanceResult> outputs = top2percent.processData();
			for (PerformanceResult output : outputs) {
				if (doingMean)
					assertEquals(3,output.getEvents().size());
				Map<String, Double> sorted = new HashMap<String, Double>();
				for (String event : output.getEvents()) {
					sorted.put(event, output.getInclusive(0, event, metric));
				}
				sorted = Utilities.sortHashMapByValues(sorted, false);
				for (String event : sorted.keySet()) {
					System.out.println(event + " " + sorted.get(event));
				}
			}
			System.out.println("\t--- EXCLUSIVE --");
			top2percent = new TopXPercentEvents(result, metric, AbstractResult.EXCLUSIVE, 2.0);
			outputs = top2percent.processData();
			for (PerformanceResult output : outputs) {
				if (doingMean) 
					assertEquals(3,output.getEvents().size());
				Map<String, Double> sorted = new HashMap<String, Double>();
				for (String event : output.getEvents()) {
					sorted.put(event, output.getExclusive(0, event, metric));
				}
				sorted = Utilities.sortHashMapByValues(sorted, false);
				for (String event : sorted.keySet()) {
					System.out.println(event + " " + sorted.get(event));
				}
			}
			System.out.println("\t--- CALLS --");
			top2percent = new TopXPercentEvents(result, null, AbstractResult.CALLS, 2.0);
			outputs = top2percent.processData();
			for (PerformanceResult output : outputs) {
				if (doingMean) 
					assertEquals(11,output.getEvents().size());
				Map<String, Double> sorted = new HashMap<String, Double>();
				for (String event : output.getEvents()) {
					sorted.put(event, output.getCalls(0, event));
				}
				sorted = Utilities.sortHashMapByValues(sorted, false);
				for (String event : sorted.keySet()) {
					System.out.println(event + " " + sorted.get(event));
				}
			}
			System.out.println("\t--- SUBROUTINES ---");
			top2percent = new TopXPercentEvents(result, null, AbstractResult.SUBROUTINES, 2.0);
			outputs = top2percent.processData();
			for (PerformanceResult output : outputs) {
				if (doingMean) 
					assertEquals(11,output.getEvents().size());
				Map<String, Double> sorted = new HashMap<String, Double>();
				for (String event : output.getEvents()) {
					sorted.put(event, output.getSubroutines(0, event));
				}
				sorted = Utilities.sortHashMapByValues(sorted, false);
				for (String event : sorted.keySet()) {
					System.out.println(event + " " + sorted.get(event));
				}
			}
		}
	}

}
