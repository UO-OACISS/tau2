/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.CorrelateEventsWithMetadata;
import edu.uoregon.tau.perfexplorer.glue.CorrelationResult;
import edu.uoregon.tau.perfexplorer.glue.LinearRegressionOperation;
import edu.uoregon.tau.perfexplorer.glue.LogarithmicOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialMeanResult;
import edu.uoregon.tau.perfexplorer.glue.TrialMetadata;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

/**
 * @author khuck
 *
 */
public class LinearRegressionOperationTest extends TestCase {

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.LinearRegressionOperation#processData()}.
	 */
	public final void testProcessData() {
		// create a list to store the trials in
		List<PerformanceResult> trials = new ArrayList<PerformanceResult>();

		Utilities.setSession("test");		
		// load each trial, and add them to the list
		Trial trial = Utilities.getTrial("Meng-Shiou", "luciferin", "luciferin-1.1");
		trials.add(new TrialMeanResult(trial));
		trial = Utilities.getTrial("Meng-Shiou", "luciferin", "luciferin-2.1");
		trials.add(new TrialMeanResult(trial));
		trial = Utilities.getTrial("Meng-Shiou", "luciferin", "luciferin-4.1");
		trials.add(new TrialMeanResult(trial));

/*		// load each trial, and add them to the list
		Trial trial = Utilities.getTrial("test", "test", "1");
		trials.add(new TrialMeanResult(trial));
		trial = Utilities.getTrial("test", "test", "2");
		trials.add(new TrialMeanResult(trial));
*/
		// get metadata for each trial, and get the differences
		List<TrialMetadata> metadata = new ArrayList<TrialMetadata>();
		for (PerformanceResult tmp : trials) {
			Trial tmpTrial = tmp.getTrial();
			metadata.add(new TrialMetadata(tmpTrial));
		}

		// because we are predicting based on processor counts, get the log of 
		// each inclusive and exclusive value.
		LogarithmicOperation loggy = new LogarithmicOperation(trials);
		double base = 2.0;
		loggy.setBase(base);
		List<PerformanceResult> outlogs = loggy.processData();
		
		PerformanceAnalysisOperation correlator = new CorrelateEventsWithMetadata(outlogs, metadata);
		List<PerformanceResult> correlations = correlator.processData();

		for (PerformanceResult output : correlations) {
			for (String event : output.getEvents()) {
				for (String metric : output.getMetrics()) {
					for (Integer thread : output.getThreads()) {
						Integer type = AbstractResult.EXCLUSIVE;
						double value = output.getDataPoint(thread, event, metric, type.intValue());
//						if (event.contains("Time:EXCLUSIVE"))
							if (metric.startsWith("node_count"))
								System.out.println(event + " " + CorrelationResult.typeToString(thread) + " " + metric + ":" + AbstractResult.typeToString(type) + " " + value);
/*						type = AbstractResult.INCLUSIVE;
						value = output.getDataPoint(thread, event, metric, type.intValue());
//							if (event.contains("Time:EXCLUSIVE"))
							if (metric.startsWith("node_count"))
								System.out.println(event + " " + CorrelationResult.typeToString(thread) + " " + metric + ":" + AbstractResult.typeToString(type) + " " + value);
								*/
					}
				}
			}
		} 
		
		// create a LinearRegressionOperation to predict a value for us
		double prediction = 6.0;
		double predictedValue = 0.0;
		LinearRegressionOperation regressor = new LinearRegressionOperation(correlations, "Time", "node_count", new Double(prediction));
		List<PerformanceResult> regressions = regressor.processData();
		
		for (PerformanceResult tmp : regressions) {
			for (Integer thread : tmp.getThreads()) {
				for (String event : tmp.getEvents()) {
					for (String metric: tmp.getMetrics()) {
						System.out.println(event + " " + tmp.getExclusive(thread, event, metric));
						predictedValue += Math.pow(2.0, tmp.getExclusive(thread, event, metric));
//						System.out.println("Inclusive" + event + " " + tmp.getInclusive(thread, event, metric));
					}
				}
			}
		}
		// output the prediction
		for (PerformanceResult tmp : trials) {
			System.out.println("Total: " + tmp.getInclusive(0, tmp.getMainEvent(), tmp.getTimeMetric()));
		}
		System.out.println("Prediction for " + prediction + ": " + predictedValue);
	}

}
