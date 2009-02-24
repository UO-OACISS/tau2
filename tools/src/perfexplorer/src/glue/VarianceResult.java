/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class VarianceResult extends DefaultResult {

	/**
	 * 
	 */
	public VarianceResult() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public VarianceResult(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	public VarianceResult(PerformanceResult input, boolean doFullCopy) {
		super(input, doFullCopy);
	}

	public String toString() {
		return "VARIANCE";
	}

}
