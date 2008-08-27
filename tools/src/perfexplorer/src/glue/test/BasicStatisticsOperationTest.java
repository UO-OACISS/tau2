package glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import glue.BasicStatisticsOperation;
import glue.ExtractEventOperation;
import glue.PerformanceAnalysisOperation;
import glue.PerformanceResult;
import glue.TrialResult;
import glue.Utilities;

import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;

public class BasicStatisticsOperationTest extends TestCase {

	public final void testProcessData() {
		return;
/*		Utilities.setSession("peri_gtc");
		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
		Trial trial2 = Utilities.getTrial("GTC", "ocracoke-O2", "128");
		Trial trial3 = Utilities.getTrial("GTC", "ocracoke-O2", "256");
		PerformanceResult result = new TrialResult(trial);
		PerformanceResult result2 = new TrialResult(trial2);
		PerformanceResult result3 = new TrialResult(trial3);
		PerformanceAnalysisOperation operation = new BasicStatisticsOperation(result, true);
		operation.addInput(result2);
		operation.addInput(result3);
		List<PerformanceResult> outputs = operation.processData();
		PerformanceResult total = outputs.get(0);
		PerformanceResult mean = outputs.get(1);
		PerformanceResult variance = outputs.get(2);
		PerformanceResult stdev = outputs.get(3);
		assertNotNull(total);
		assertNotNull(mean);
		assertNotNull(variance);
		assertNotNull(stdev);
		assertEquals(total.getThreads().size(), 256);
		assertEquals(total.getEvents().size(), 42);
		assertEquals(total.getMetrics().size(), 1);
		assertEquals(mean.getThreads().size(), 256);
		assertEquals(mean.getEvents().size(), 42);
		assertEquals(mean.getMetrics().size(), 1);
		assertEquals(variance.getThreads().size(), 256);
		assertEquals(variance.getEvents().size(), 42);
		assertEquals(variance.getMetrics().size(), 1);
		assertEquals(stdev.getThreads().size(), 256);
		assertEquals(stdev.getEvents().size(), 42);
		assertEquals(stdev.getMetrics().size(), 1);
		for (Integer thread : total.getThreads()) {
			for (String event : total.getEvents()) {
				for (String metric : total.getMetrics()) {
					assertEquals(total.getExclusive(thread, event, metric)/3.0, 
							mean.getExclusive(thread, event, metric));
					assertEquals(total.getInclusive(thread, event, metric)/3.0, 
							mean.getInclusive(thread, event, metric));
					System.out.println(thread + " " + event + " " + metric + " " + stdev.getExclusive(thread, event, metric));
				}
				assertEquals(total.getCalls(thread, event)/3.0, 
						mean.getCalls(thread, event));
				assertEquals(total.getSubroutines(thread, event)/3.0, 
						mean.getSubroutines(thread, event));
			}
		}
*/	}
	
	public final void testProcessData2() {
		Utilities.setSession("perfdmf_uploaded");
		Trial trial = Utilities.getTrial("NPB", "LU.W.4-metadata", "p1");
		PerformanceResult result = new TrialResult(trial);
		List<String> events = new ArrayList<String>();
		events.add("MPI_Init()");
//		events.add("rhs");
		PerformanceAnalysisOperation reducer = new ExtractEventOperation(result, events);
		List<PerformanceResult> reduced = reducer.processData(); 
		PerformanceAnalysisOperation stats = new BasicStatisticsOperation(reduced.get(0));
		List<PerformanceResult> outputs = stats.processData();
		PerformanceResult total = outputs.get(0);
		PerformanceResult mean = outputs.get(1);
		PerformanceResult variance = outputs.get(2);
		PerformanceResult stdev = outputs.get(3);
		PerformanceResult min = outputs.get(4);
		PerformanceResult max = outputs.get(5);
		assertNotNull(total);
		assertNotNull(mean);
		assertNotNull(variance);
		assertNotNull(stdev);
		assertNotNull(min);
		assertNotNull(max);
		assertEquals(total.getThreads().size(), 1);
		assertEquals(total.getEvents().size(), events.size());
		assertEquals(total.getMetrics().size(), 1);
		assertEquals(mean.getThreads().size(), 1);
		assertEquals(mean.getEvents().size(), events.size());
		assertEquals(mean.getMetrics().size(), 1);
		assertEquals(variance.getThreads().size(), 1);
		assertEquals(variance.getEvents().size(), events.size());
		assertEquals(variance.getMetrics().size(), 1);
		assertEquals(stdev.getThreads().size(), 1);
		assertEquals(stdev.getEvents().size(), events.size());
		assertEquals(stdev.getMetrics().size(), 1);
		assertEquals(min.getThreads().size(), 1);
		assertEquals(min.getEvents().size(), events.size());
		assertEquals(min.getMetrics().size(), 1);
		assertEquals(max.getThreads().size(), 1);
		assertEquals(max.getEvents().size(), events.size());
		assertEquals(max.getMetrics().size(), 1);
		for (Integer thread : total.getThreads()) {
			for (String event : total.getEvents()) {
				for (String metric : total.getMetrics()) {
					assertEquals(total.getExclusive(thread, event, metric)/4.0, 
							mean.getExclusive(thread, event, metric));
					assertEquals(total.getInclusive(thread, event, metric)/4.0, 
							mean.getInclusive(thread, event, metric));
					System.out.print(thread + " " + event + " " + metric + " " + total.getExclusive(thread, event, metric) + ", ");
					System.out.print(mean.getExclusive(thread, event, metric) + ", ");
					System.out.print(variance.getExclusive(thread, event, metric) + ", ");
					System.out.print(stdev.getExclusive(thread, event, metric) + ", ");
					System.out.print(min.getExclusive(thread, event, metric) + ", ");
					System.out.println(max.getExclusive(thread, event, metric));
				}
				assertEquals(total.getCalls(thread, event)/4.0, 
						mean.getCalls(thread, event));
				assertEquals(total.getSubroutines(thread, event)/4.0, 
						mean.getSubroutines(thread, event));
			}
		}
	}
}
