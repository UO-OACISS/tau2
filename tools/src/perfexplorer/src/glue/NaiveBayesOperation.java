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

	protected String metric = null;
	protected int type = AbstractResult.EXCLUSIVE;
    protected AnalysisFactory factory = null;
    protected PerfExplorerServer server = null;
    protected ClassifierInterface classifier = null;
    protected List<String> classNames = null;
    protected final String trainString = "Naive Bayes Training";
    protected final String testString = "Naive Bayes Test";

	
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
    	this.classNames = new ArrayList<String>();
		
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
    	data = factory.createRawData(this.trainString, eventList, instances, eventList.size(), this.classNames);

    	int masterIndex = 0;
        // now, iterate over the inputs, and add them to the raw data interface object.
        for (PerformanceResult input : inputs) {
    		for(Integer thread : input.getThreads()) {
            	for (int eventIndex = 0 ; eventIndex < eventList.size(); eventIndex++) {
            		String event = eventList.get(eventIndex);
           			data.addValue(masterIndex, eventIndex, input.getDataPoint(thread, event, metric, type));
        		}
    			data.addValue(masterIndex, eventList.size(), input.getName());
       			masterIndex++;
        	}
        }
		getClassifierFromFactory(data);
		try {
			classifier.buildClassifier();
		} catch (Exception e) {
			System.err.println("failure to build classifier.");
			System.exit(0);
		}

		return outputs;
	}

	/**
	 * @param data
	 */
	protected void getClassifierFromFactory(RawDataInterface data) {
		this.classifier = factory.createNaiveBayesClassifier(data);
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
    	RawDataInterface data = factory.createRawData(this.testString, eventList, input.getThreads().size(), eventList.size(), this.classNames);
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
