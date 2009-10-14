package edu.uoregon.tau.perfexplorer.glue.test;

import java.io.File;
import junit.framework.TestCase;
import edu.uoregon.tau.perfexplorer.glue.*;

public class ExtractChildrenOperationTest extends TestCase {

	public void testProcessData() {
		String[] files = new String[1];
		files[0] = new String("/home/khuck/data/mn/wrf/wrf-fullcallpath.ppk");
		PerformanceResult trial = new DataSourceResult(DataSourceResult.PPK, files, false);
		
		PerformanceAnalysisOperation extractor = new ExtractChildrenOperation(trial, "SOLVE_EM");
		PerformanceResult output = extractor.processData().get(0);
		
		for (String event : output.getEvents()) {
			System.out.println(event);
		}
		assertEquals(76, output.getEvents().size());
	}

}
