/**
 * 
 */
package glue;

import java.util.ArrayList;
import java.util.List;

import server.PerfExplorerServer;
import clustering.AnalysisFactory;
import clustering.LinearRegressionInterface;
import clustering.RawDataInterface;

/**
 * @author khuck
 *
 */
public class CorrelationOperation extends DefaultOperation {

	/**
	 * @param inputs
	 */
	public CorrelationOperation(List<PerformanceResult> inputs) {
		super(inputs);
	}

	/**
	 * @param input
	 */
	public CorrelationOperation(PerformanceResult input) {
		super(input);
	}

	/**
	 * Dummy implementation which is a no-op on the input data
	 */
	public List<PerformanceResult> processData() {
	    AnalysisFactory factory = null;
	    PerfExplorerServer server = null;
        server = PerfExplorerServer.getServer();
        factory = server.getAnalysisFactory();

        for (PerformanceResult input : inputs) {
			// first, since we need the average and stddev foreach event/metric,
			// get the basic stats for this input
			PerformanceResult correlation = new CorrelationResult();
			// now, loop over all event / metric / type
			for (String event : input.getEvents()) {
				for (String metric : input.getMetrics()) {
					for (Integer type : AbstractResult.getTypes()) {
//					Integer type = AbstractResult.EXCLUSIVE;
						// now, loop over all event / metric / type AGAIN
						for (String event2 : input.getEvents()) {
							for (String metric2 : input.getMetrics()) {
								for (Integer type2 : AbstractResult.getTypes()) {
//								Integer type2 = AbstractResult.EXCLUSIVE;
									// solve for r
									double r = 0.0;
									double[] y1 = new double[input.getThreads().size()];
									double[] y2 = new double[input.getThreads().size()];
									List<String> eventList = new ArrayList<String>();
									eventList.add(event);
									eventList.add(event2);
									RawDataInterface data = factory.createRawData("Correlation Test", eventList, input.getThreads().size(), eventList.size());
									for (Integer thread : input.getThreads()) {
										y1[thread.intValue()] = input.getDataPoint(thread, event, metric, type);
										y2[thread.intValue()] = input.getDataPoint(thread, event2, metric2, type2);
					        			data.addValue(thread, 0, input.getDataPoint(thread, event, metric, type));
					        			data.addValue(thread, 1, input.getDataPoint(thread, event2, metric2, type2));
									}
									// solve with Clustering Utilities
									r = PerfExplorerServer.getServer().getAnalysisFactory().getUtilities().doCorrelation(y1, y2, input.getThreads().size());
									correlation.putDataPoint(CorrelationResult.CORRELATION, event + ":" + metric + ":" + AbstractResult.typeToString(type), event2 + ":" + metric2, type2, r);

									LinearRegressionInterface regression = factory.createLinearRegressionEngine();
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
									correlation.putDataPoint(CorrelationResult.SLOPE, event + ":" + metric + ":" + AbstractResult.typeToString(type), event2 + ":" + metric2, type2, slope);
									correlation.putDataPoint(CorrelationResult.INTERCEPT, event + ":" + metric + ":" + AbstractResult.typeToString(type), event2 + ":" + metric2, type2, intercept);
								}
							}
						}
					}
				}
			}
			outputs.add(correlation);
		}
		return outputs;
	}
}

