package edu.uoregon.tau.dms.analysis;

import java.lang.StrictMath;

public class EventMatrix {

	public double[][] threadMatrix = null;
	public double[][] distanceMatrix = null;
	public String[] eventName = null;
	public double[] threadTotal = null;
	public int threadCount = 0;
	public int eventCount = 0;

	public EventMatrix (int threadCount, int eventCount) {
		this.threadMatrix = new double[threadCount][eventCount];
		this.distanceMatrix = new double[threadCount][threadCount];
		this.eventName = new String[eventCount];
		this.threadTotal = new double[threadCount];
		this.threadCount = threadCount;
		this.eventCount = eventCount;
	}

	public void getEuclidianDistance() {
		double tmpVar = 0;
		double runningTotal = 0;
		for (int i = 0 ; i < threadCount ; i++ ) {
			for (int j = 0 ; j < threadCount ; j++ ) {
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

	public String toString() {
		StringBuffer buf = new StringBuffer();
		buf.append("Thread Count: " + threadCount + "\n");
		buf.append("Event Count: " + eventCount + "\n");
        buf.append("Normalized Values:\n");
        for (int i = 0 ; i < threadCount; i++ ) {
            buf.append("thread " + i + ":\n");
            for (int j = 0 ; j < eventCount; j++ ) {
				buf.append("    ");
                buf.append(eventName[j]);
				buf.append(": ");
                buf.append(threadMatrix[i][j]);
            	buf.append("\n");
            }
        }
        buf.append("Distance Matrix:\n");
        for (int i = 0 ; i < threadCount; i++ ) {
            buf.append("thread " + i + ":\n");
            for (int j = 0 ; j < threadCount; j++ ) {
				buf.append("  ");
                buf.append(distanceMatrix[i][j]);
            }
            buf.append("\n");
        }
		return buf.toString();
	}

}
