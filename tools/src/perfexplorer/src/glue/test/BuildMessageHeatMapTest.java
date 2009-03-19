package edu.uoregon.tau.perfexplorer.glue.test;

import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.BuildMessageHeatMap;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;
import junit.framework.TestCase;

public class BuildMessageHeatMapTest extends TestCase {

	public final void testProcessData() {
		Thread thread = new Thread() { 
			public void run() {
				Utilities.setSession("test");
				Trial trial = Utilities.getTrial("ring", "TAU_EACH_SEND", "4p");
				PerformanceResult input = new TrialResult(trial);
				PerformanceAnalysisOperation messageHeatMap = new BuildMessageHeatMap(input);
				messageHeatMap.processData();
			}
		};
		thread.run();
	}

}
