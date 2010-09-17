/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;


/**
 * @author khuck
 *
 */
public class TotalResult extends DefaultResult {

	/**
	 * 
	 */
	private static final long serialVersionUID = -503934195802079286L;

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
