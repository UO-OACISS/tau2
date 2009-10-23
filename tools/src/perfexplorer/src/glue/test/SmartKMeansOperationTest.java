/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.AbstractResult;
import edu.uoregon.tau.perfexplorer.glue.BasicStatisticsOperation;
import edu.uoregon.tau.perfexplorer.glue.DataSourceResult;
import edu.uoregon.tau.perfexplorer.glue.DefaultResult;
import edu.uoregon.tau.perfexplorer.glue.ExtractEventOperation;
import edu.uoregon.tau.perfexplorer.glue.NormalizeOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.SmartKMeansOperation;
import edu.uoregon.tau.perfexplorer.glue.TopXEvents;
import edu.uoregon.tau.perfexplorer.glue.TrialResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

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
	 * Test method for {@link edu.uoregon.tau.perfexplorer.glue.SmartKMeansOperation#processData()}.
	 */
	public final void testProcessData() {
		
		Trial trial = null;
		int type = AbstractResult.EXCLUSIVE;
		String metric = "P_WALL_CLOCK_TIME";
		PerformanceResult result = null;
		PerformanceAnalysisOperation kmeans = null;
		List<PerformanceResult> clusterResult = null;
		
/*		Utilities.setSession("peris3d");
		type = AbstractResult.EXCLUSIVE;
		metric = "P_WALL_CLOCK_TIME";
		System.out.println("Generating data...");
		result = new DefaultResult();
		Random generator = new Random();
		
		result.putDataPoint(0, "ENSDART", metric, type, 1352);
		result.putDataPoint(1, "ENSDART", metric, type, 1141);
		result.putDataPoint(2, "ENSDART", metric, type, 705);
		result.putDataPoint(3, "ENSDART", metric, type, 694);
		result.putDataPoint(4, "ENSDART", metric, type, 327);
		result.putDataPoint(5, "ENSDART", metric, type, 324);
		result.putDataPoint(6, "ENSDART", metric, type, 321);
		result.putDataPoint(7, "ENSDART", metric, type, 316);
		result.putDataPoint(8, "ENSDART", metric, type, 308);
		result.putDataPoint(9, "ENSDART", metric, type, 306);
		result.putDataPoint(10, "ENSDART", metric, type, 305);
		result.putDataPoint(11, "ENSDART", metric, type, 300);
		result.putDataPoint(12, "ENSDART", metric, type, 295);
		result.putDataPoint(13, "ENSDART", metric, type, 294);
		result.putDataPoint(14, "ENSDART", metric, type, 294);

		System.out.println("\nClustering data...");
		kmeans = new SmartKMeansOperation(result, metric, type, 10);
		clusterResult = kmeans.processData();
		System.out.println("Estimated value for k: " + clusterResult.get(0).getThreads().size());*/

		// synthesize some data with 2 natural clusters
/*		for (int outer = 1 ; outer < 11 ; outer++) {
			for (int i = 0 ; i < 100 ; i+=outer) {
				for (int inner = 0 ; inner < outer ; inner++) {
					result.putDataPoint(i+inner, "w", metric, type, Math.random()+(5*inner));
					result.putDataPoint(i+inner, "x", metric, type, Math.random()+(5*inner));
					//result.putDataPoint(i+inner, "y", metric, type, Math.random()+(5*inner));
					//result.putDataPoint(i+inner, "z", metric, type, Math.random()+(5*inner));
				}
			}
	
			System.out.println("\nClustering data...");
			kmeans = new SmartKMeansOperation(result, metric, type, 10);
			clusterResult = kmeans.processData();
			System.out.println("Estimated value for k: " + clusterResult.get(0).getThreads().size());
			assertEquals(outer, clusterResult.get(0).getThreads().size());
		}*/

/*		Utilities.setSession("apart");
		trial = Utilities.getTrial("gtc", "jaguar", "256");
		type = AbstractResult.EXCLUSIVE;
		metric = "P_WALL_CLOCK_TIME";
		System.out.println("Loading data...");
		result = new TrialResult(trial, null, null, null, false);

		System.out.println("\nReducinging data...");
		PerformanceAnalysisOperation reducer = new TopXEvents(result, metric, type, 5);
		PerformanceResult reduced = reducer.processData().get(0);
		System.out.println("\nClustering data...");
		kmeans = new SmartKMeansOperation(reduced, metric, type, 10);
		clusterResult = kmeans.processData();
		System.out.println("Estimated value for k: " + clusterResult.get(0).getThreads().size());
		assertEquals(2, clusterResult.get(0).getThreads().size());*/
		
//		Utilities.setSession("spaceghost");
//		trial = Utilities.getTrial("sPPM", "Frost", "16.16");
//		System.out.println("Loading data...");
//		result = new TrialResult(trial);
		
	    String[] files = new String[1];
        files[0] = "/home/khuck/data/mn/gromacs/Nucleosome.MareNostrum.Scaling/32";
        result = new DataSourceResult(DataSourceResult.TAUPROFILE, files, false);
        result.setIgnoreWarnings(true);

		type = AbstractResult.EXCLUSIVE;
		metric = result.getTimeMetric();

		System.out.println("\nReducinging data...");
		// first, get the stats
		BasicStatisticsOperation stats = new BasicStatisticsOperation(result);
		PerformanceResult means = stats.processData().get(BasicStatisticsOperation.MEAN);
		// then, using the stats, find the top X event names
		PerformanceAnalysisOperation reducer = new TopXEvents(means, metric, type, 3);
		PerformanceResult reduced = reducer.processData().get(0);
		// then, extract those events from the actual data
		List<String> tmpEvents = new ArrayList<String>(reduced.getEvents());
		reducer = new ExtractEventOperation(result, tmpEvents);
		reduced = reducer.processData().get(0);
/*		for (String e : reduced.getEvents()) {
			for (Integer t : reduced.getThreads()) {
				System.out.println(e + " " + reduced.getExclusive(t, e, metric));
			}
		}*/
		NormalizeOperation normalizer = new NormalizeOperation(reduced);
	    PerformanceResult normalized = normalizer.processData().get(0);
	    normalized.setIgnoreWarnings(true);

		System.out.println("\nClustering data...");
		kmeans = new SmartKMeansOperation(normalized, metric, type, 5);
		clusterResult = kmeans.processData();
		System.out.println("Estimated value for k: " + clusterResult.get(0).getThreads().size());
		assertEquals(3, clusterResult.get(0).getThreads().size());

/*		Utilities.setSession("peris3d");
		trial = Utilities.getTrial("S3D", "hybrid-study", "XT3/XT4");
		type = AbstractResult.EXCLUSIVE;
		metric = "GET_TIME_OF_DAY";
		System.out.println("Loading data...");
		result = new TrialResult(trial, null, null, null, false);

		System.out.println("\nClustering data...");
		kmeans = new SmartKMeansOperation(result, metric, type, 10);
		clusterResult = kmeans.processData();
		System.out.println("Estimated value for k: " + clusterResult.get(0).getThreads().size());
		assertEquals(2, clusterResult.get(0).getThreads().size());*/
		
	}
	
	

}
