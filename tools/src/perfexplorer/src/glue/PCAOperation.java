package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.clustering.PrincipalComponentsAnalysisInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.clustering.weka.AnalysisFactory;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;

/**
 * 
 */

/**
 * @author khuck
 *
 */
public class PCAOperation extends AbstractPerformanceOperation {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -7122252507773375487L;
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
//	    PerfExplorerServer server = null;
//        server = 
        	PerfExplorerServer.getServer();

        for (PerformanceResult input : inputs) {
        	List<String> eventList = new ArrayList<String>(input.getEvents());
        	RawDataInterface data = AnalysisFactory.createRawData("Cluster Test", eventList, input.getThreads().size(), eventList.size(), null);
    		for(Integer thread : input.getThreads()) {
            	int eventIndex = 0;
            	for (String event : eventList) {
        			data.addValue(thread, eventIndex++, input.getDataPoint(thread, event, metric, type));
        			if (event.equals(input.getMainEvent())) {
        				data.addMainValue(thread, eventIndex-1, input.getDataPoint(thread, event, metric, type));
        			}
        		}
        	}
    		PrincipalComponentsAnalysisInterface clusterer = AnalysisFactory.createPCAEngine(data);
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
