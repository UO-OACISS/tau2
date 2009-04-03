package edu.uoregon.tau.perfexplorer.glue.test;

import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.*;
import junit.framework.TestCase;

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
