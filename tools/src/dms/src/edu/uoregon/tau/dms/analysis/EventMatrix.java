package edu.uoregon.tau.dms.analysis;

import java.lang.StrictMath;

public class EventMatrix {

	public double[][] threadMatrix = null;
	public double[][] distanceMatrix = null;
	public String[] eventNames = null;
	public double[] threadTotal = null;
	public int threadCount = 0;
	public int eventCount = 0;

	public EventMatrix (int threadCount, int eventCount) {
		this.threadMatrix = new double[threadCount][eventCount];
		this.distanceMatrix = new double[threadCount][eventCount];
		this.eventNames = new String[eventCount];
		this.eventNames = new String[threadCount];
	}

	public void getRawData() {
	}

	public void getEuclidianDistance() {
		double tmpVar = 0;
		double runningTotal = 0;
		for (int i = 0 ; i < threadCount ; i++ ) {
			for (int j = i ; j < threadCount ; j++ ) {
				// what's the distance between these two nodes?
				runningTotal = 0;
				for (int k = 0 ; k < eventCount ; k++ ) {
					tmpVar = threadMatrix[i][k] - threadMatrix[j][k];
					runningTotal += tmpVar * tmpVar;
				}
				distanceMatrix[i][j] = java.lang.StrictMath.sqrt(runningTotal);
			}
		}
	}

	public void getManhattanDistance() {
		double tmpVar = 0;
		double runningTotal = 0;
		for (int i = 0 ; i < threadCount ; i++ ) {
			for (int j = i ; j < threadCount ; j++ ) {
				// what's the distance between these two nodes?
				runningTotal = 0;
				for (int k = 0 ; k < eventCount ; k++ ) {
					tmpVar = threadMatrix[i][k] - threadMatrix[j][k];
					runningTotal += java.lang.StrictMath.abs(tmpVar);
				}
				distanceMatrix[i][j] = runningTotal;
			}
		}
	}

}
