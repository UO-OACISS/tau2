/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.CorrelateEventsWithMetadata;
import edu.uoregon.tau.perfexplorer.glue.CorrelationResult;
import edu.uoregon.tau.perfexplorer.glue.ExtractNonCallpathEventOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TopXEvents;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.TrialThreadMetadata;
import edu.uoregon.tau.perfexplorer.glue.Utilities;
import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class CorrelateEventsWithMetadataTest extends TestCase {

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.CorrelateEventsWithMetadata#processData()}.
	 */
	public void testProcessData() {
//		int sessionid = Utilities.setSession("localhost:5432/perfdmf");
		Utilities.setSession("apart");
		Trial trial = Utilities.getTrial("sweep3d", "jaguar", "256");
//		Trial trial = Utilities.getTrial("gtc", "jaguar", "512");
		
		PerformanceResult trialData = new TrialResult(trial);
		TrialThreadMetadata trialMetadata = new TrialThreadMetadata(trial);
		
		// get the callpath events
		PerformanceAnalysisOperation callpath = new ExtractNonCallpathEventOperation(trialData);
		PerformanceResult callpathOnly = callpath.processData().get(0);
		// get only the top 5 events
		PerformanceAnalysisOperation extractor = new TopXEvents(callpathOnly, trialData.getTimeMetric(), AbstractResult.EXCLUSIVE, 5);
		PerformanceResult extracted = extractor.processData().get(0);
		
		PerformanceAnalysisOperation correlator = new CorrelateEventsWithMetadata(extracted, trialMetadata);
		List<PerformanceResult> outputs = correlator.processData();

		for (PerformanceResult output : outputs) {
			for (String event : output.getEvents()) {
				for (String metric : output.getMetrics()) {
					Integer type = AbstractResult.EXCLUSIVE;
//					for (Integer thread : output.getThreads()) {
					Integer thread = CorrelationResult.CORRELATION;
					double value = output.getDataPoint(thread, event, metric, type.intValue());
					if ((value > 0.85 || value < -0.85)/* && event.contains("P_WALL_CLOCK_TIME:EXCLUSIVE")*/)
//					if (metric.startsWith("local grid size:isize") || metric.startsWith("local grid size:jsize")/* || metric.startsWith("total Neighbors")*/)
						System.out.println(event + " " + CorrelationResult.typeToString(thread) + " " + metric + ":" + AbstractResult.typeToString(type) + " " + value);
				}
			}
		}

/*		List<PerformanceResult> inputs = new ArrayList<PerformanceResult>();
		List<TrialMetadata> metas = new ArrayList<TrialMetadata>();
		trial = Utilities.getTrial("sweep3d", "jaguar", "64");
		inputs.add(new TrialMeanResult(trial));
		metas.add(new TrialMetadata(trial));
//		trial = Utilities.getTrial("sweep3d", "ocracoke", "64");
		trial = Utilities.getTrial("sweep3d", "jaguar", "128");
		inputs.add(new TrialMeanResult(trial));
		metas.add(new TrialMetadata(trial));

		// get only the top 5 events
		extractor = new TopXEvents(inputs, trialData.getTimeMetric(), AbstractResult.EXCLUSIVE, 5);
		inputs = extractor.processData();
		
		// correlate away!
		correlator = new CorrelateEventsWithMetadata(inputs, metas);
		outputs = correlator.processData();


		System.out.println("*******************************");
		for (PerformanceResult output : outputs) {
			for (String event : output.getEvents()) {
				for (String metric : output.getMetrics()) {
					Integer type = AbstractResult.EXCLUSIVE;
					Integer thread = CorrelationResult.CORRELATION;
					double value = output.getDataPoint(thread, event, metric, type.intValue());
					if ((value > 0.75 || value < -0.75) && event.contains("P_WALL_CLOCK_TIME:EXCLUSIVE"))
//					if (metric.startsWith("local grid size:isize") || metric.startsWith("local grid size:jsize") || metric.startsWith("total Neighbors"))
						System.out.println(event + " " + CorrelationResult.typeToString(thread) + " " + metric + ":" + AbstractResult.typeToString(type) + " " + value);
				}
			}
		}
*/		
	}

}
