/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;


/**
 * @author khuck
 *
 */
public class StDevResult extends DefaultResult {

	/**
	 * 
	 */
	private static final long serialVersionUID = 9104561440541540304L;

	/**
	 * 
	 */
	public StDevResult() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public StDevResult(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	public StDevResult(PerformanceResult input, boolean doFullCopy) {
		super(input, doFullCopy);
	}

	public String toString() {
		return "STDDEV";
	}

}
