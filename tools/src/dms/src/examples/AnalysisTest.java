package examples;

import java.util.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.analysis.*;

public class AnalysisTest {

    public AnalysisTest() {
		super();
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
		try {

		// Create a PerfDMFSession object
		DataSession session = new PerfDMFSession();
		session.initialize(args[0]);
		System.out.println ("API loaded...");

		// test out the analysis!
		// Trial trial = session.setTrial(50);
		Trial trial = session.setTrial(55); // sppm, 256 threads on blue
		// Trial trial = session.setTrial(61); // sppm, 256 threads on frost
		// Trial trial = session.setTrial(67); // sphot, 32 nodes, 2 threads
		// Trial trial = session.setTrial(68); // pprof.dat example
		// Trial trial = session.setTrial(69); // sphot, 4 nodes 2 threads
		// Trial trial = session.setTrial(72); // sppm, openmp, 8 threads
		Vector metrics = session.getMetrics();
		Metric metric = (Metric)(metrics.elementAt(0));
		// Metric metric = (Metric)(metrics.elementAt(3));
		// DistanceAnalysis distance = new ThreadDistance((PerfDMFSession)session, trial, metric);
		DistanceAnalysis distance = new EventDistance((PerfDMFSession)session, trial, metric);
		/*
		double[][] matrix = distance.getEuclideanDistance();
		System.out.print(distance.toString());
		System.out.println("Euclidian distance:");
		for (int i = 0 ; i < distance.getThreadCount(); i++ ) {
			System.out.print("thread " + i + ": ");
			for (int j = 0 ; j < distance.getEventCount(); j++ ) {
				if (j > 0) System.out.print(", ");
				System.out.print(matrix[i][j]);
			}
			System.out.println("");
		}
		*/
		double[][] matrix = distance.getManhattanDistance();
		System.out.print(distance.toString());
		/*
		System.out.println("Manhattan distance:");
		for (int i = 0 ; i < distance.getThreadCount(); i++ ) {
			System.out.print("thread " + i + ": ");
			for (int j = 0 ; j < distance.getEventCount(); j++ ) {
				if (j > 0) System.out.print(", ");
				System.out.print(matrix[i][j]);
			}
			System.out.println("");
		}
		*/
		// disconnect and exit.
		session.terminate();
		System.out.println ("Exiting.");

		} catch (Exception e) {
			e.printStackTrace();
		}
		return;
    }
}

