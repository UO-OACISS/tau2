package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.CopyOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialMeanResult;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.TrialTotalResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

import java.util.List;

import junit.framework.TestCase;

public class CopyOperationTest extends TestCase {

	public final void testProcessData() {
		Utilities.setSession("perigtc");
		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
		PerformanceResult input = new TrialResult(trial);
		PerformanceAnalysisOperation operation = new CopyOperation(input);
		List<PerformanceResult> outputs = operation.processData();
		PerformanceResult output = outputs.get(0);
		assertNotNull(output);
		assertEquals(input.getThreads().size(), 64);
		assertEquals(input.getThreads().size(), output.getThreads().size());
		for (Integer thread : input.getThreads()) {
			assertEquals(input.getEvents().size(), 42);
			assertEquals(input.getEvents().size(), output.getEvents().size());
			for (String event : input.getEvents()) {
				assertEquals(input.getMetrics().size(), 1);
				assertEquals(input.getMetrics().size(), output.getMetrics().size());
				for (String metric : input.getMetrics()) {
					assertEquals(output.getExclusive(thread, event, metric), 
							input.getExclusive(thread, event, metric));
					assertEquals(output.getInclusive(thread, event, metric), 
							input.getInclusive(thread, event, metric));
				}
				assertEquals(output.getCalls(thread, event), 
						input.getCalls(thread, event));
				assertEquals(output.getSubroutines(thread, event), 
						input.getSubroutines(thread, event));
			}
		}
		input = new TrialMeanResult(trial);
		operation = new CopyOperation(input);
		outputs = operation.processData();
		output = outputs.get(0);
		assertNotNull(output);
		assertEquals(input.getThreads().size(), 1);
		assertEquals(input.getThreads().size(), output.getThreads().size());
		for (Integer thread : input.getThreads()) {
			assertEquals(input.getEvents().size(), 42);
			assertEquals(input.getEvents().size(), output.getEvents().size());
			for (String event : input.getEvents()) {
				assertEquals(input.getMetrics().size(), 1);
				for (String metric : input.getMetrics()) {
					assertEquals(output.getExclusive(thread, event, metric), 
							input.getExclusive(thread, event, metric));
					assertEquals(output.getInclusive(thread, event, metric), 
							input.getInclusive(thread, event, metric));
				}
				assertEquals(output.getCalls(thread, event), 
						input.getCalls(thread, event));
				assertEquals(output.getSubroutines(thread, event), 
						input.getSubroutines(thread, event));
			}
		}
		input = new TrialTotalResult(trial);
		operation = new CopyOperation(input);
		outputs = operation.processData();
		output = outputs.get(0);
		assertNotNull(output);
		assertEquals(input.getThreads().size(), 1);
		assertEquals(input.getThreads().size(), output.getThreads().size());
		for (Integer thread : input.getThreads()) {
			assertEquals(input.getEvents().size(), 42);
			assertEquals(input.getEvents().size(), output.getEvents().size());
			for (String event : input.getEvents()) {
				assertEquals(input.getMetrics().size(), 1);
				assertEquals(input.getMetrics().size(), output.getMetrics().size());
				for (String metric : input.getMetrics()) {
					assertEquals(output.getExclusive(thread, event, metric), 
							input.getExclusive(thread, event, metric));
					assertEquals(output.getInclusive(thread, event, metric), 
							input.getInclusive(thread, event, metric));
				}
				assertEquals(output.getCalls(thread, event), 
						input.getCalls(thread, event));
				assertEquals(output.getSubroutines(thread, event), 
						input.getSubroutines(thread, event));
			}
		}
	}

}
