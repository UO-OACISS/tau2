/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.MergeTrialsOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

import java.util.List;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class MergeTrialsOperationTest extends TestCase {

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.MergeTrialsOperation#processData()}.
	 */
	public final void testProcessData() {
		Utilities.setSession("perfdmf_test");
		Trial trial = Utilities.getTrial("gtc_bench", "jaguar.longrun", "64.first");
		Trial trial2 = Utilities.getTrial("gtc_bench", "jaguar.longrun", "64.second");
		Trial trial3 = Utilities.getTrial("gtc_bench", "jaguar.longrun", "64.third");
		PerformanceResult result = new TrialResult(trial);
		PerformanceResult result2 = new TrialResult(trial2);
		PerformanceResult result3 = new TrialResult(trial3);
		PerformanceAnalysisOperation operation = new MergeTrialsOperation(result);
		operation.addInput(result2);
		operation.addInput(result3);
		List<PerformanceResult> outputs = operation.processData();
		PerformanceResult output = outputs.get(0);
		assertNotNull(output);
		assertEquals(output.getThreads().size(), 64);
		assertEquals(output.getMetrics().size(), 6);
		assertEquals(output.getEvents().size(), 91);
		
		for (String event : result.getEvents()) {
			for (String metric : result.getMetrics()) {
				for (Integer thread : result.getThreads()) {
					assertEquals(result.getDataPoint(thread, event, metric, AbstractResult.EXCLUSIVE), output.getDataPoint(thread, event, metric, AbstractResult.EXCLUSIVE));
					assertEquals(result.getDataPoint(thread, event, metric, AbstractResult.INCLUSIVE), output.getDataPoint(thread, event, metric, AbstractResult.INCLUSIVE));
				}
			}
		}
		for (String event : result2.getEvents()) {
			for (String metric : result2.getMetrics()) {
				for (Integer thread : result2.getThreads()) {
					assertEquals(result2.getDataPoint(thread, event, metric, AbstractResult.EXCLUSIVE), output.getDataPoint(thread, event, metric, AbstractResult.EXCLUSIVE));
					assertEquals(result2.getDataPoint(thread, event, metric, AbstractResult.INCLUSIVE), output.getDataPoint(thread, event, metric, AbstractResult.INCLUSIVE));
				}
			}
		}
		for (String event : result3.getEvents()) {
			for (String metric : result3.getMetrics()) {
				for (Integer thread : result3.getThreads()) {
					assertEquals(result3.getDataPoint(thread, event, metric, AbstractResult.EXCLUSIVE), output.getDataPoint(thread, event, metric, AbstractResult.EXCLUSIVE));
					assertEquals(result3.getDataPoint(thread, event, metric, AbstractResult.INCLUSIVE), output.getDataPoint(thread, event, metric, AbstractResult.INCLUSIVE));
				}
			}
		}
	}

}
