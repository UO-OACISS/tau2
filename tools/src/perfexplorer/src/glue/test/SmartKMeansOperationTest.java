/**
 * 
 */
package glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import glue.AbstractResult;
import glue.SmartKMeansOperation;
import glue.PerformanceAnalysisOperation;
import glue.PerformanceResult;
import glue.TopXEvents;
import glue.TrialResult;
import glue.Utilities;

import java.util.List;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class SmartKMeansOperationTest extends TestCase {

	/**
	 * @param arg0
	 */
	public SmartKMeansOperationTest(String arg0) {
		super(arg0);
	}

	/**
	 * Test method for {@link glue.SmartKMeansOperation#processData()}.
	 */
	public final void testProcessData() {
/*		Utilities.setSession("apart");
		Trial trial = Utilities.getTrial("gtc", "jaguar", "256");
		int type = AbstractResult.EXCLUSIVE;
		String metric = "P_WALL_CLOCK_TIME";
		System.out.println("Loading data...");
		PerformanceResult result = new TrialResult(trial, null, null, null, false);*/

/*		Utilities.setSession("spaceghost");
		Trial trial = Utilities.getTrial("sPPM", "Frost", "16.16");
		int type = AbstractResult.EXCLUSIVE;
		String metric = "PAPI_FP_INS";
		System.out.println("Loading data...");
		PerformanceResult result = new TrialResult(trial);*/
		
		Utilities.setSession("peris3d");
		Trial trial = Utilities.getTrial("S3D", "hybrid-study", "XT3/XT4");
		int type = AbstractResult.EXCLUSIVE;
		String metric = "P_WALL_CLOCK_TIME";
		System.out.println("Loading data...");
		PerformanceResult result = new TrialResult(trial, null, null, null, false);

		System.out.println("Reducing data...");
		PerformanceAnalysisOperation reducer = new TopXEvents(result, metric, type, 11);
		List<PerformanceResult> reduced = reducer.processData(); 
		for (String event : reduced.get(0).getEvents())
			System.out.println(event + " ");
		System.out.println("\nClustering data...");
		PerformanceAnalysisOperation kmeans = new SmartKMeansOperation(reduced.get(0), metric, type, 10);
		kmeans.processData();
	}

}
