/**
 * 
 */
package glue;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class StDevResult extends DefaultResult {

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
