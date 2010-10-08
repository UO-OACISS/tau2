package edu.uoregon.tau.perfexplorer.glue.test;

import junit.framework.TestCase;
import edu.uoregon.tau.perfexplorer.glue.DataSourceResult;
import edu.uoregon.tau.perfexplorer.glue.ExtractChildrenOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;

public class ExtractChildrenOperationTest extends TestCase {

	public void testProcessData() {
		String[] files = new String[1];
		files[0] = new String("/home/khuck/data/mn/wrf/gprof.out");
		PerformanceResult trial = new DataSourceResult(DataSourceResult.GPROF, files, true);
		
		PerformanceAnalysisOperation extractor = new ExtractChildrenOperation(trial, "SOLVE_EM");
		PerformanceResult output = extractor.processData().get(0);
		
		for (String event : output.getEvents()) {
			System.out.println(event);
		}
		assertEquals(76, output.getEvents().size());
	}

}
