/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;


/**
 * @author khuck
 *
 */
public class MeanResult extends DefaultResult {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8264599953003122282L;

	/**
	 * 
	 */
	public MeanResult() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public MeanResult(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	public MeanResult(PerformanceResult input, boolean doFullCopy) {
		super(input, doFullCopy);
	}

	public String toString() {
		return "MEAN";
	}

}
