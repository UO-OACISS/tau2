/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;


/**
 * @author khuck
 *
 */
public class VarianceResult extends DefaultResult {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3180885146491602231L;

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
