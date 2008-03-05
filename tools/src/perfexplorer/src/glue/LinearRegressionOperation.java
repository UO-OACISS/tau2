/**
 * 
 */
package glue;

import java.util.HashSet;
import java.util.List;
import java.util.ArrayList;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class LinearRegressionOperation extends AbstractPerformanceOperation {

	String metric = null;
	String metadata = null;
	Double value = null;
	
	/**
	 * @param input
	 */
	public LinearRegressionOperation(PerformanceResult input, String metric, String metadata, Double value) {
		super(input);
		this.metric = metric;
		this.metadata = metadata;
		this.value = value;
	}

	/**
	 * @param trial
	 */
	public LinearRegressionOperation(Trial trial, String metric, String metadata, Double value) {
		super(trial);
		this.metric = metric;
		this.metadata = metadata;
		this.value = value;
	}

	/**
	 * @param inputs
	 */
	public LinearRegressionOperation(List<PerformanceResult> inputs, String metric, String metadata, Double value) {
		super(inputs);
		this.metric = metric;
		this.metadata = metadata;
		this.value = value;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 * 
	 * iterate through the events, and for each trial, perform a linear regression on the data.
	 * The output will be one trial, with two metrics - slope and intercept - for each event.
	 */
	public List<PerformanceResult> processData() {
		List<PerformanceResult> outputs = new ArrayList<PerformanceResult>();

		for (PerformanceResult input : inputs) {
			PerformanceResult output = new DefaultResult();
			outputs.add(output);
			for (String event : input.getEvents()) {
				for (String metric : input.getMetrics()) {
					if (event.endsWith(this.metric + ":EXCLUSIVE") && metric.startsWith(this.metadata)) {
						// HANDLE THE EXCLUSIVE VALUES
						Integer type = AbstractResult.EXCLUSIVE;
						Integer thread = CorrelationResult.SLOPE;
						double slope = input.getDataPoint(thread, event, metric, type.intValue());
						thread = CorrelationResult.INTERCEPT;
						double intercept = input.getDataPoint(thread, event, metric, type.intValue());
						double response = intercept + (slope * this.value);
						output.putExclusive(0, event, metric, response);
					}
/*					if (event.endsWith(this.metric + ":INCLUSIVE") && metric.startsWith(this.metadata)) {
						// HANDLE THE INCLUSIVE VALUES
						Integer type = AbstractResult.INCLUSIVE;
						Integer thread = CorrelationResult.SLOPE;
						double slope = input.getDataPoint(thread, event, metric, type.intValue());
						thread = CorrelationResult.INTERCEPT;
						double intercept = input.getDataPoint(thread, event, metric, type.intValue());
						double response = intercept + (slope * this.value);
						output.putInclusive(0, event, metric, response);
					}*/
				}
			}
		}
		return outputs;
	}

}

