/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class TotalResult extends DefaultResult {

	/**
	 * 
	 */
	public TotalResult() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public TotalResult(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	public TotalResult(PerformanceResult input, boolean doFullCopy) {
		super(input, doFullCopy);
	}

	public String toString() {
		return "TOTAL";
	}
}
