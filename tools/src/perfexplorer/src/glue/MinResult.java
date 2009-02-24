/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.HashMap;
import java.util.Map;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class MinResult extends DefaultResult {

	/**
	 * 
	 */
	public MinResult() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public MinResult(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	public MinResult(PerformanceResult input, boolean doFullCopy) {
		super(input, doFullCopy);
	}

	/* (non-Javadoc)
	 * @see glue.AbstractResult#addDataPoint(java.lang.Integer, java.lang.String, java.lang.String, int, double)
	 * 
	 * Because the default value is 0.0 when there isn't a value,  we need special processing
	 * to handle the case where the value hasn't been stored yet, and is greater than 0.0.
	 * 
	 */
	@Override
	public void putDataPoint(Integer thread, String event, String metric, int type, double value) {
		boolean newEntry = false;
		if (type == INCLUSIVE) {
			if (!inclusiveData.containsKey(thread) || 
				!inclusiveData.get(thread).containsKey(event) ||
				!inclusiveData.get(thread).get(event).containsKey(metric)) {
				super.putInclusive(thread, event, metric, value);
				newEntry = true;
			}
		} else if (type == EXCLUSIVE) {
			if (!exclusiveData.containsKey(thread) || 
				!exclusiveData.get(thread).containsKey(event) ||
				!exclusiveData.get(thread).get(event).containsKey(metric)) {
				super.putExclusive(thread, event, metric, value);
				newEntry = true;
			}
		} else if (type == CALLS) {
			if (!callData.containsKey(thread) || 
				!callData.get(thread).containsKey(event)) {
				super.putCalls(thread, event, value);
				newEntry = true;
			}
		} else if (type == SUBROUTINES) {
			if (!callData.containsKey(thread) || 
				!callData.get(thread).containsKey(event)) {
				super.putSubroutines(thread, event, value);
				newEntry = true;
			}
		}
		
		if (!newEntry) {
			double oldValue = super.getDataPoint(thread, event, metric, type);
			if (value < oldValue) {
				super.putDataPoint(thread, event, metric, type, value);
			}
		}
	}

	public String toString() {
		return "MIN";
	}

}
