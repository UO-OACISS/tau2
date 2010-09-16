/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.clustering.LinearRegressionInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.clustering.weka.AnalysisFactory;

/**
 * @author khuck
 *
 */
public class CorrelateEventsWithMetadata extends AbstractPerformanceOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7728642909863765823L;
	private PerformanceResult trialData = null;
	private TrialThreadMetadata trialMetadata = null;
	private List<TrialMetadata> metadatas = null;
	
	/**
	 * @param input
	 */
	protected CorrelateEventsWithMetadata(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	protected CorrelateEventsWithMetadata(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	protected CorrelateEventsWithMetadata(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}
	
	public CorrelateEventsWithMetadata(PerformanceResult trialData, TrialThreadMetadata trialMetadata) {
		super(trialData);
		this.inputs.add(trialMetadata);
		this.trialData = trialData;
		this.trialMetadata = trialMetadata;
	}

	public CorrelateEventsWithMetadata(List<PerformanceResult> inputs, List<TrialMetadata> trialMetadata) {
		super(inputs);
		this.trialData = null;
		this.metadatas = trialMetadata;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		if (trialData != null) {
			processOneTrial();
		} else {
			processNTrials();
		}
		return outputs;
	}

	private void processOneTrial() {

        
		CorrelationResult correlation = new CorrelationResult(trialData, false);
		outputs.add(correlation);
		// now, loop over all event / metric / type
		
		for (String event : trialData.getEvents()) {
			for (String metric : trialData.getMetrics()) {
				for (Integer type : AbstractResult.getTypes()) {
//					Integer type = AbstractResult.EXCLUSIVE;
					// now, loop over all metadata fields
					for (String event2 : trialMetadata.getEvents()) {
						String metric2 = "METADATA";
						// "Exclusive" is the only type of data in metadata
						Integer type2 = AbstractResult.EXCLUSIVE;
						// solve for r
						double r = 0.0;
						double[] y1 = new double[trialData.getThreads().size()];
						double[] y2 = new double[trialData.getThreads().size()];
						List<String> eventList = new ArrayList<String>();
						eventList.add(event);
						eventList.add(event2);
						RawDataInterface data = AnalysisFactory.createRawData("Correlation Test", eventList, trialData.getThreads().size(), eventList.size(), null);
						for (Integer thread : trialData.getThreads()) {
							// BE CAREFUL!  The first value is the predictor, and the second is the response.
							// When working with metadata, be sure to correlate the PERFORMANCE with the METADATA!
							y1[thread.intValue()] = trialMetadata.getDataPoint(thread, event2, metric2, type2);
							y2[thread.intValue()] = trialData.getDataPoint(thread, event, metric, type);
	        				data.addValue(thread, 0, trialMetadata.getDataPoint(thread, event2, metric2, type2));
	        				data.addValue(thread, 1, trialData.getDataPoint(thread, event, metric, type));
						}
						// if all the measurement or metadata values are the same, ignore it.
						boolean same = true;
						for (int i = 1 ; i < trialData.getThreads().size(); i++) {
							if (y2[i] != y2[i-1] && y1[i] != y1[i-1]) {
								same = false;
								break;
							}
						}
						if (!same) {
							// solve with Clustering Utilities
							r = AnalysisFactory.getUtilities().doCorrelation(y1, y2, trialData.getThreads().size());
							if (Double.isNaN(r) || Double.isInfinite(r)) {
								r = 0.0;
							}
							correlation.putDataPoint(CorrelationResult.CORRELATION, event + ":" + metric + ":" + AbstractResult.typeToString(type), event2 + ":" + metric2, type2, r);
//							correlation.putDataPoint(CorrelationResult.CORRELATION, event + ":" + metric, event2 + ":" + metric2, type, r);
							
							LinearRegressionInterface regression = AnalysisFactory.createLinearRegressionEngine();
							regression.setInputData(data);
							try {
								regression.findCoefficients();
							} catch (Exception e) {
								System.err.println("failure to perform linear regression.");
								System.exit(0);
							}
						
							List<Double> coefficients = regression.getCoefficients();
			
							double slope = coefficients.get(0);
							double intercept = coefficients.get(2);
							if (Double.isNaN(slope) || Double.isInfinite(slope)) {
								slope = 0.0;
							}
							if (Double.isNaN(intercept) || Double.isInfinite(intercept)) {
								intercept = 0.0;
							}
							correlation.putDataPoint(CorrelationResult.SLOPE, event + ":" + metric + ":" + AbstractResult.typeToString(type), event2 + ":" + metric2, type2, slope);
							correlation.putDataPoint(CorrelationResult.INTERCEPT, event + ":" + metric + ":" + AbstractResult.typeToString(type), event2 + ":" + metric2, type2, intercept);
							correlation.assertFact(event, metric, type, event2, metric2, CorrelationResult.METADATA, r, slope, intercept);
						}
					} // for events2
				} // for types
			} // for metrics
		} // for events
	}

	private void processNTrials() {

		// find the intersection of trial events
		// merge the trials
		trialData = new TrialResult();
		trialMetadata = new TrialThreadMetadata();
		for (int i = 0; i < inputs.size(); i++) {
			PerformanceResult tmp = inputs.get(i);
//			if (tmp instanceof TrialMeanResult) {
//				TrialMeanResult input = (TrialMeanResult)tmp;
				for (String event : tmp.getEvents()) {
					for (String metric : tmp.getMetrics()) {
						trialData.putInclusive(i, event, metric, 
							tmp.getInclusive(0, event, metric));
						trialData.putExclusive(i, event, metric, 
							tmp.getExclusive(0, event, metric));
						trialData.putCalls(i, event,  
							tmp.getCalls(0, event));
						trialData.putSubroutines(i, event,  
							tmp.getSubroutines(0, event));
					}
//				}
			}
		}

			// find the intersection of metadata fields
			// merge the metadata
		int i = 0;
		for (TrialMetadata metadata : metadatas) {
			Hashtable<String,String> commonAttributes = metadata.getCommonAttributes();
			for (String key : commonAttributes.keySet()) {
				if (!key.equals("pid") && !key.toLowerCase().contains("time")) {
					try {
						Double tmpDouble = Double.parseDouble(commonAttributes.get(key));
						// The metric name is "metadata"
						trialMetadata.putExclusive(i, key, "METADATA", tmpDouble.doubleValue());
					} catch (NumberFormatException e) { /* do nothing for now */ }
				}
			}
			i++;
		}
		processOneTrial();
	}

}
