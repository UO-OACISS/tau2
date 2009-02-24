package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.CorrelationOperation;
import edu.uoregon.tau.perfexplorer.glue.CorrelationResult;
import edu.uoregon.tau.perfexplorer.glue.ExtractEventOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;

public class CorrelationOperationTest extends TestCase {

	public final void testProcessData() {
		//Utilities.setSession("spaceghost");
		//Trial trial = Utilities.getTrial("NPB", "LU.W.4-metadata", "p1");
		Utilities.setSession("openuh");
		Trial trial = Utilities.getTrial("static", "size.400", "16.threads");
		PerformanceResult result = new TrialResult(trial);
		List<String> events = new ArrayList<String>();
		//events.add("MPI_Init()");
		//events.add("rhs");
		events.add("LOOP #3 [file:/mnt/netapp/home1/khuck/openuh/src/fpga/msap.c <63, 163>]");
		//events.add("LOOP #2 [file:/mnt/netapp/home1/khuck/openuh/src/fpga/msap.c <65, 158>]");
		PerformanceAnalysisOperation reducer = new ExtractEventOperation(result, events);
		List<PerformanceResult> reduced = reducer.processData(); 
		PerformanceAnalysisOperation correlation = new CorrelationOperation(reduced.get(0));
		List<PerformanceResult> outputs = correlation.processData();
		for (PerformanceResult output : outputs) {
			for (String event : output.getEvents()) {
				//for (String metric : output.getMetrics()) {
				String metric = "P_WALL_CLOCK_TIME";
//					for (Integer type : AbstractResult.getTypes()) {
					Integer type = AbstractResult.EXCLUSIVE;
						for (Integer thread : output.getThreads()) {
//					Integer thread = CorrelationResult.CORRELATION;
							if (!event.equals(metric + ":" + AbstractResult.typeToString(type))) {
								System.out.println(event + " " + CorrelationResult.typeToString(thread) + " " + metric + ":" + AbstractResult.typeToString(type) + " " + output.getDataPoint(thread, event, metric, type.intValue()));
							}
						}
//					}
				//}
			}
		}
	}

}
