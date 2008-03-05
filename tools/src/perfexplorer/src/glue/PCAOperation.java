package glue;

import java.util.ArrayList;
import java.util.List;

import server.PerfExplorerServer;
import clustering.AnalysisFactory;
import clustering.KMeansClusterInterface;
import clustering.PrincipalComponentsAnalysisInterface;
import clustering.RawDataInterface;
import edu.uoregon.tau.perfdmf.Trial;

/**
 * 
 */

/**
 * @author khuck
 *
 */
public class PCAOperation extends AbstractPerformanceOperation {
	
	private String metric = null;
	private int type = AbstractResult.EXCLUSIVE;
	private int maxComponents = -1;  // include all
	
	/**
	 * @param input
	 */
	public PCAOperation(PerformanceResult input, String metric, int type) {
		super(input);
		this.metric = metric;
		this.type = type;
	}

	/**
	 * @param trial
	 */
	public PCAOperation(Trial trial) {
		super(trial);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public PCAOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
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
        	RawDataInterface data = factory.createRawData("Cluster Test", eventList, input.getThreads().size(), eventList.size());
    		for(Integer thread : input.getThreads()) {
            	int eventIndex = 0;
            	for (String event : eventList) {
        			data.addValue(thread, eventIndex++, input.getDataPoint(thread, event, metric, type));
        			if (event.equals(input.getMainEvent())) {
        				data.addMainValue(thread, eventIndex-1, input.getDataPoint(thread, event, metric, type));
        			}
        		}
        	}
    		PrincipalComponentsAnalysisInterface clusterer = factory.createPCAEngine(data);
			clusterer.setMaxComponents(this.maxComponents);
			try {
				clusterer.doPCA();
			} catch (Exception e) {
				System.err.println("failure to perform PCA.");
				System.exit(0);
			}
			PerformanceResult components = new ClusterOutputResult(clusterer.getResults(), metric, type);
			outputs.add(components);
        }

		return outputs;
	}

	/**
	 * @return the maxComponents
	 */
	public int getMaxComponents() {
		return maxComponents;
	}

	/**
	 * @param maxComponents the maxComponents to set
	 */
	public void setMaxComponents(int maxComponents) {
		this.maxComponents = maxComponents;
	}

	/**
	 * @return the metric
	 */
	public String getMetric() {
		return metric;
	}

	/**
	 * @param metric the metric to set
	 */
	public void setMetric(String metric) {
		this.metric = metric;
	}

	/**
	 * @return the type
	 */
	public int getType() {
		return type;
	}

	/**
	 * @param type the type to set
	 */
	public void setType(int type) {
		this.type = type;
	}

}
