/**
 * 
 */
package glue;

import java.util.ArrayList;
import java.util.List;

import server.PerfExplorerServer;
import clustering.AnalysisFactory;
import clustering.ClassifierInterface;
import clustering.KMeansClusterInterface;
import clustering.RawDataInterface;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class NaiveBayesOperation extends AbstractPerformanceOperation {

	private String metric = null;
	private int type = AbstractResult.EXCLUSIVE;
	
	/**
	 * @param input
	 */
	public NaiveBayesOperation(PerformanceResult input, String metric, int type) {
		super(input);
		this.metric = metric;
		this.type = type;
	}

	/**
	 * @param trial
	 */
	public NaiveBayesOperation(Trial trial) {
		super(trial);
	}

	/**
	 * @param inputs
	 */
	public NaiveBayesOperation(List<PerformanceResult> inputs) {
		super(inputs);
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
	    AnalysisFactory factory = null;
	    PerfExplorerServer server = null;
        server = PerfExplorerServer.getServer();
        factory = server.getAnalysisFactory();

        for (PerformanceResult input : inputs) {
        	List<String> eventList = new ArrayList<String>(input.getEvents());
        	System.out.println("instances: " + input.getThreads().size());
        	System.out.println("dimensions: " + input.getEvents().size());
        	RawDataInterface data = factory.createRawData("Naive Bayes Test", eventList, input.getThreads().size(), eventList.size());
    		for(Integer thread : input.getThreads()) {
            	int eventIndex = 0;
            	for (String event : eventList) {
        			data.addValue(thread, eventIndex++, input.getDataPoint(thread, event, metric, type));
        			if (event.equals(input.getMainEvent())) {
        				data.addMainValue(thread, eventIndex-1, input.getDataPoint(thread, event, metric, type));
        			}
        		}
        	}
			ClassifierInterface classifier = factory.createNaiveBayesClassifier(data);
			try {
				classifier.buildClassifier();
			} catch (Exception e) {
				System.err.println("failure to build classifier.");
				System.exit(0);
			}
        }

		return outputs;
	}

	public String getMetric() {
		return metric;
	}

	public void setMetric(String metric) {
		this.metric = metric;
	}

}
