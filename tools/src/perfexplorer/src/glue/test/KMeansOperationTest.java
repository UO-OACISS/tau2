/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;
import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.DataSourceResult;
import edu.uoregon.tau.perfexplorer.glue.ExtractEventOperation;
import edu.uoregon.tau.perfexplorer.glue.KMeansOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;

/**
 * @author khuck
 *
 */
public class KMeansOperationTest extends TestCase {

	/**
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.KMeansOperation#processData()}.
	 */
	public final void testProcessData() {
//		Utilities.setSession("perigtc");
//		Trial trial = Utilities.getTrial("GTC", "ocracoke-O2", "64");
//		Utilities.setSession("local");
//		Trial trial = Utilities.getTrial("sweep3d", "jaguar", "256");
//		PerformanceResult result = new TrialResult(trial);
	    String[] files = new String[1];
        files[0] = "/home/khuck/data/mn/gromacs/Nucleosome.MareNostrum.Scaling/16";
        PerformanceResult result = new DataSourceResult(DataSourceResult.TAUPROFILE, files, false);
        result.setIgnoreWarnings(true);
		int type = AbstractResult.EXCLUSIVE;
		for (String metric : result.getMetrics()) {
			for (int i = 2 ; i <= 2 ; i++) {
//				PerformanceAnalysisOperation reducer = new TopXEvents(result, metric, type, 1.0);
//				PerformanceResult reduced = reducer.processData().get(0); 
				List<String> tmpEvents = new ArrayList<String>();
				tmpEvents.add("void do_nonbonded(t_commrec *, t_forcerec *, rvec *, rvec *, t_mdatoms *, real *, real *, real *, t_nrnb *, real, real *, int, int, int, int) C [{nonbonded.c} {271,1}-{517,1}]");
				tmpEvents.add("MPI_Sendrecv()");
				tmpEvents.add("MPI_Waitall()");
				PerformanceAnalysisOperation reducer = new ExtractEventOperation(result, tmpEvents);
				PerformanceResult reduced = reducer.processData().get(0);
				
//				for (Integer th : reduced.getThreads()) {
//					for (String ev : reduced.getEvents()) {
//						if (reduced.getDataPoint(th, ev, metric, type) == 0.0) {
//							reduced.putDataPoint(th, ev, metric, type, 0.0);
//						} else {
//							reduced.putDataPoint(th, ev, metric, type, reduced.getDataPoint(th, ev, metric, type)/1.0);
//						}
//						System.out.println(reduced.getDataPoint(th, ev, metric, type));
//					}
//				}
				PerformanceAnalysisOperation kmeans = new KMeansOperation(reduced, metric, type, i);
				List<PerformanceResult> outputs = kmeans.processData();
				//ClusterOutputResult first = (ClusterOutputResult)outputs.get(0); 
				//ClusterOutputResult second = (ClusterOutputResult)outputs.get(1);
				//Integer thread = 0;
//				String event = "void do_nonbonded(t_commrec *, t_forcerec *, rvec *, rvec *, t_mdatoms *, real *, real *, real *, t_nrnb *, real, real *, int, int, int, int) C [{nonbonded.c} {271,1}-{517,1}]";
				for (String event : tmpEvents) {
					System.out.println(event);
					System.out.println("Centroid: " + outputs.get(0).getDataPoint(0, event, metric, type) + " " + outputs.get(0).getDataPoint(1, event, metric, type));
					System.out.println("StdDev: " + outputs.get(1).getDataPoint(0, event, metric, type) + " " + outputs.get(1).getDataPoint(1, event, metric, type));
					System.out.println("Min: " + outputs.get(2).getDataPoint(0, event, metric, type) + " " + outputs.get(2).getDataPoint(1, event, metric, type));
					System.out.println("Max: " + outputs.get(3).getDataPoint(0, event, metric, type) + " " + outputs.get(3).getDataPoint(1, event, metric, type));
					System.out.println("Counts[0]: " + outputs.get(4).getDataPoint(0, "count", metric, type) + " " + outputs.get(4).getDataPoint(1, "count", metric, type));
				}
			}
		}
	}

}
