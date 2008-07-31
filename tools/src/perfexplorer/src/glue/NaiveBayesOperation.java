/**
 * 
 */
package glue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
    private AnalysisFactory factory = null;
    private PerfExplorerServer server = null;
    private ClassifierInterface classifier = null;
	
	/**
	 * @param input
	 */
	public NaiveBayesOperation(PerformanceResult input, String metric, int type) {
		super(input);
		this.metric = metric;
		this.type = type;
		getFactory();
	}

	/**
	 * @param trial
	 */
	public NaiveBayesOperation(Trial trial) {
		super(trial);
		getFactory();
	}

	/**
	 * @param inputs
	 */
	public NaiveBayesOperation(List<PerformanceResult> inputs) {
		super(inputs);
		getFactory();
	}

	/**
	 * Builds the analysis factory so we can get a classifier.
	 */
	private void getFactory() {
		if (this.factory == null) {
			if (this.server == null) {
				this.server = PerfExplorerServer.getServer();
			}
			this.factory = server.getAnalysisFactory();
		}
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// For the classifier, each input is one catetgory/class.  We need to merge the inputs
		// into one set, so we have to iterate over them.
		
		RawDataInterface data = null;
    	Set<String> eventSet = new HashSet<String>();
    	int instances = 0;
    	List<String> classNames = new ArrayList<String>();
		
		// ok, first loop through the inputs, and get the superset of events, and the total
    	// number of threads.
        for (PerformanceResult input : inputs) {
        	for (String event : input.getEvents()) {
        		eventSet.add(event);
        	}
        	instances += input.getThreads().size();
        	classNames.add(input.getName());
        }
        // create a list from the set, and add the category name attribute.
        
        List<String> eventList = new ArrayList<String>(eventSet);

        // create our data storage object
    	System.out.println("instances: " + instances);
    	System.out.println("dimensions: " + eventList.size());
    	data = factory.createRawData("Naive Bayes Training", eventList, instances, eventList.size(), classNames);

        // now, iterate over the inputs, and add them to the raw data interface object.
        for (PerformanceResult input : inputs) {
    		for(Integer thread : input.getThreads()) {
            	for (int eventIndex = 0 ; eventIndex < eventList.size(); eventIndex++) {
            		String event = eventList.get(eventIndex);
           			data.addValue(thread, eventIndex, input.getDataPoint(thread, event, metric, type));
        		}
    			data.addValue(thread, eventList.size(), input.getName());
        	}
        }
		this.classifier = factory.createNaiveBayesClassifier(data);
		try {
			classifier.buildClassifier();
		} catch (Exception e) {
			System.err.println("failure to build classifier.");
			System.exit(0);
		}

		return outputs;
	}

	public String getMetric() {
		return metric;
	}

	public void setMetric(String metric) {
		this.metric = metric;
	}
	
	public List<String> classifyInstances(PerformanceResult input) {
		List<String> categories = null;
		
    	List<String> eventList = new ArrayList<String>(input.getEvents());
    	System.out.println("instances: " + input.getThreads().size());
    	System.out.println("dimensions: " + input.getEvents().size());
    	RawDataInterface data = factory.createRawData("Naive Bayes Test", eventList, input.getThreads().size(), eventList.size(), null);
		for(Integer thread : input.getThreads()) {
        	int eventIndex = 0;
        	for (String event : eventList) {
    			data.addValue(thread, eventIndex++, input.getDataPoint(thread, event, metric, type));
    			if (event.equals(input.getMainEvent())) {
    				data.addMainValue(thread, eventIndex-1, input.getDataPoint(thread, event, metric, type));
    			}
    		}
    	}
		
		categories = classifier.classifyInstances(data);
		
		return categories;
	}

}
