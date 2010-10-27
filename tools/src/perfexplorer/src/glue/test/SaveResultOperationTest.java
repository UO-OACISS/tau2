/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import junit.framework.TestCase;
import edu.uoregon.tau.perfexplorer.glue.DeriveMetricOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.SaveResultOperation;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

/**
 * @author khuck
 *
 */
public class SaveResultOperationTest extends TestCase {

	/**
	 * @param arg0
	 */
	public SaveResultOperationTest(String arg0) {
		super(arg0);
	}

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.SaveResultOperation#processData()}.
	 */
	public final void testProcessData() {
		Utilities.setSession("test");
		PerformanceResult input = new TrialResult(Utilities.getTrial("msap_parametric.optix.dynamic.2", "size.100", "4.threads"));
		
		String firstMetric = "BACK_END_BUBBLE_ALL";
		String secondMetric = "CPU_CYCLES";
		PerformanceAnalysisOperation derivor = new DeriveMetricOperation(input, firstMetric, secondMetric, DeriveMetricOperation.DIVIDE);
		PerformanceResult derived = derivor.processData().get(0);
		
		SaveResultOperation saver = new SaveResultOperation(derived);
		saver.setForceOverwrite(false);
		saver.processData();
	}

}
