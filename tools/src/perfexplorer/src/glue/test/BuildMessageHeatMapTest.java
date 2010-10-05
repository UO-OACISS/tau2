package edu.uoregon.tau.perfexplorer.glue.test;

import junit.framework.TestCase;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.BuildMessageHeatMap;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

public class BuildMessageHeatMapTest extends TestCase {

	public final void testProcessData() {
		Thread thread = new Thread() { 
			public void run() {
/*				Utilities.setSession("test");
				Trial trial = Utilities.getTrial("NPB_SP", "CONTEXT", "64p");
*/				Utilities.setSession("peris3d");
				Trial trial = Utilities.getTrial("s3d", "intrepid-c2h4-misc", "512_com");
				PerformanceResult input = new TrialResult(trial);
//				String[] files = {"/home/khuck/data/Heatmap/matsc72"};
//				PerformanceResult input = new DataSourceResult(DataSourceResult.TAUPROFILE, files, false);
				PerformanceAnalysisOperation messageHeatMap = new BuildMessageHeatMap(input);
				messageHeatMap.processData();
			}
		};
		thread.run();
		try {
			Thread.sleep(30000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
