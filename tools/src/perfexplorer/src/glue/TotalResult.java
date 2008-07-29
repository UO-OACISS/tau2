/**
 * 
 */
package glue;

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

	public TotalResult(Trial trial) {
		super(trial);
	}

	public String toString() {
		return "TOTAL";
	}
}
