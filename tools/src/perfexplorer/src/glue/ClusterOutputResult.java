/**
 * 
 */
package glue;

import java.util.List;

import clustering.RawDataInterface;

/**
 * @author khuck
 *
 */
public class ClusterOutputResult extends DefaultResult {

	private String metric;
	private int type;
	
	/**
	 * 
	 */
	public ClusterOutputResult() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public ClusterOutputResult(PerformanceResult input) {
		super(input);
	}

	public ClusterOutputResult(RawDataInterface rawData, String metric, int type) {
		super();
		this.metric = metric;
		this.type = type;
		for (int threadIndex = 0 ; threadIndex < rawData.numVectors(); threadIndex++) {
			List<String> events = rawData.getEventNames();
			int eventIndex = 0;
			for (String event : events) {
				switch (type) {
				case INCLUSIVE:
					this.putInclusive(threadIndex, event, metric, rawData.getValue(threadIndex, eventIndex));
				case EXCLUSIVE:
					this.putExclusive(threadIndex, event, metric, rawData.getValue(threadIndex, eventIndex));
				case CALLS:
					this.putCalls(threadIndex, event, rawData.getValue(threadIndex, eventIndex));
				case SUBROUTINES:
					this.putSubroutines(threadIndex, event, rawData.getValue(threadIndex, eventIndex));
				}
				eventIndex++;
			}
			
		}
		
		
	}
}
