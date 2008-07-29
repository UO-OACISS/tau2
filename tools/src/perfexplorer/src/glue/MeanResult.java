/**
 * 
 */
package glue;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class MeanResult extends DefaultResult {

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

	public MeanResult(Trial trial) {
		super(trial);
	}

	public String toString() {
		return "MEAN";
	}

}
