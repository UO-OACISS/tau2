package edu.uoregon.tau.dms.analysis;

import java.lang.StrictMath;

public class DistanceMatrix {

	public double[][] dataMatrix = null;
	public double[][] distanceMatrix = null;
	public String[] eventName = null;
	public double[] total = null;
	public int matrixSize = 0;
	public int dimensionCount = 0;

	public DistanceMatrix (int matrixSize, int dimensionCount) {
		this.dataMatrix = new double[matrixSize][dimensionCount];
		this.total = new double[matrixSize];
		this.matrixSize = matrixSize;
		this.dimensionCount = dimensionCount;
		// we need to initialize this matrix, because there is no
		// guarantee that every event will happen on every thread
		for (int i = 0 ; i < matrixSize ; i++) {
			for (int j = 0 ; j < dimensionCount ; j++) {
				dataMatrix[i][j] = 0.0;
			}
		}
	}

	public void getEuclideanDistance() {
		this.distanceMatrix = new double[matrixSize][matrixSize];
		double tmpVar = 0;
		double runningTotal = 0;
		for (int i = 0 ; i < matrixSize ; i++ ) {
			for (int j = i ; j < matrixSize ; j++ ) {
				// what's the distance between these two items?
				runningTotal = 0;
				for (int k = 0 ; k < dimensionCount ; k++ ) {
					tmpVar = dataMatrix[i][k] - dataMatrix[j][k];
					runningTotal += tmpVar * tmpVar;
				}
				distanceMatrix[i][j] = java.lang.StrictMath.sqrt(runningTotal);
				distanceMatrix[j][i] = java.lang.StrictMath.sqrt(runningTotal);
			}
		}
	}

	public void getManhattanDistance() {
		this.distanceMatrix = new double[matrixSize][matrixSize];
		double tmpVar = 0;
		double runningTotal = 0;
		for (int i = 0 ; i < matrixSize ; i++ ) {
			for (int j = i ; j < matrixSize ; j++ ) {
				// what's the distance between these two items?
				runningTotal = 0;
				for (int k = 0 ; k < dimensionCount ; k++ ) {
					tmpVar = dataMatrix[i][k] - dataMatrix[j][k];
					runningTotal += java.lang.StrictMath.abs(tmpVar);
				}
				distanceMatrix[i][j] = runningTotal;
				distanceMatrix[j][i] = runningTotal;
			}
		}
	}

	public String toString() {
		StringBuffer buf = new StringBuffer();
		buf.append("# Matrix Size: " + matrixSize + "\n");
		buf.append("# Dimension Count: " + dimensionCount + "\n");
        buf.append("# Normalized Values:\n");
		double total = 0.0;
        for (int i = 0 ; i < matrixSize; i++ ) {
			total = 0.0;
            buf.append("# row " + i + ":\n");
            for (int j = 0 ; j < dimensionCount; j++ ) {
				buf.append("#    col ");
                buf.append(j);
				buf.append(": ");
                buf.append(dataMatrix[i][j]);
            	buf.append("\n");
				total += dataMatrix[i][j];
            }
			buf.append("# Total: " + total + "\n");
        }
        buf.append("# Distance Matrix:\n");
        for (int i = 0 ; i < matrixSize; i++ ) {
            buf.append("# row " + i + ":\n");
            for (int j = 0 ; j < matrixSize; j++ ) {
                buf.append(distanceMatrix[i][j]);
            	buf.append("\n");
            }
            buf.append("\n");
        }
		return buf.toString();
	}

}
