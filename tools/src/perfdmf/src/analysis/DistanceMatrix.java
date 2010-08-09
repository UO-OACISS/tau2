package edu.uoregon.tau.perfdmf.analysis;


public class DistanceMatrix {

	public double[][] dataMatrix = null;
	public double[][] distanceMatrix = null;
	public String[] eventName = null;
	public double[] total = null;
	public int matrixSize = 0;
	public int dimensionCount = 0;
	private double maxDistance = 0.0;
	private final int[] colorMap = { 0xFF000000,
                               0xFF000011,
                               0xFF000022,
                               0xFF000033,
                               0xFF000044,
                               0xFF000055,
                               0xFF000066,
                               0xFF000077,
                               0xFF000088,
                               0xFF000099,
                               0xFF0000AA,
                               0xFF0000BB,
                               0xFF0000CC,
                               0xFF0000DD,
                               0xFF0000EE,
                               0xFF0000FF,
                               0xFF001100,
                               0xFF002200,
                               0xFF003300,
                               0xFF004400,
                               0xFF005500,
                               0xFF006600,
                               0xFF007700,
                               0xFF008800,
                               0xFF009900,
                               0xFF00AA00,
                               0xFF00BB00,
                               0xFF00CC00,
                               0xFF00DD00,
                               0xFF00EE00,
                               0xFF00FF00,
                               0xFF11EE00,
                               0xFF22DD00,
                               0xFF33CC00,
                               0xFF44BB00,
                               0xFF55AA00,
                               0xFF669900,
                               0xFF778800,
                               0xFF887700,
                               0xFF996600,
                               0xFFAA5500,
                               0xFFBB4400,
                               0xFFCC3300,
                               0xFFDD2200,
                               0xFFEE1100,
                               0xFFFF0000 };

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
				if (maxDistance < runningTotal) 
					maxDistance = runningTotal;
			}
		}
	}

	public String toString() {
		StringBuffer buf = new StringBuffer();
		buf.append("# Matrix Size: " + matrixSize + "\n");
		buf.append("# Dimension Count: " + dimensionCount + "\n");
        /*
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
		*/
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

	/* We need to convert the NxN distance matrix to a black and
	 white NxN image.  Each element in the matrix will become a pixel.
	 The image data has to be in a single dimension array, so output
	 it that way.  Need to do a color implementation, too.
	*/

	public int[] toImage(boolean scaledRange, boolean triangle) {
		int[] data = new int[matrixSize*matrixSize];
		// these variables are for the color code
        int red, green, blue;
        int opaque = 255;
		int value;
		// the range of the matrix values are from 0.0 to 2.0.
		// that needs to be converted to 0 to 256.  Therefore,
		// multiply the matrix value by 128.0 and convert to 
		// an integer.
		double factor = scaledRange ? 255.0 / maxDistance : 127.5;
		int i, j;//, k, l;
		int idx = 0;

		if (triangle) {
        	for (i = 0 ; i < matrixSize; i++ ) {
            	for (j = i ; j < matrixSize; j++ ) {
					idx = i * matrixSize + j;
					red = green = blue = (int)(distanceMatrix[i][j] * factor);
        			value = (opaque << 24 ) | (red << 16 ) | (green << 8 ) | blue;
					data[idx] = value;
            	}
        	}
        } else {
        	for (i = 0 ; i < matrixSize; i++ ) {
            	for (j = 0 ; j < matrixSize; j++ ) {
					red = green = blue = (int)(distanceMatrix[i][j] * factor);
        			value = (opaque << 24 ) | (red << 16 ) | (green << 8 ) | blue;
					data[idx++] = value;
            	}
        	}
        }

		System.out.println("Image range: 0.0 to " + maxDistance);
		return data;
	}

	public int[] toColorImage(boolean scaledRange, boolean triangle) {
		int[] data = new int[matrixSize*matrixSize];
		// the range of the matrix values are from 0.0 to 2.0.
		// that needs to be converted to 0 to 45.  Therefore,
		// multiply the matrix value by 22.5 and convert to 
		// an integer.
		double factor = scaledRange ? 45.0 / maxDistance : 22.5;
		int i, j;//, k, l;
		int idx = 0;

		if (triangle) {
        	for (i = 0 ; i < matrixSize; i++ ) {
            	for (j = i ; j < matrixSize; j++ ) {
					idx = i * matrixSize + j;
					data[idx] = colorMap[(int)(distanceMatrix[i][j] * factor)];
            	}
        	}
        } else {
        	for (i = 0 ; i < matrixSize; i++ ) {
            	for (j = 0 ; j < matrixSize; j++ ) {
					data[idx++] = colorMap[(int)(distanceMatrix[i][j] * factor)];
            	}
        	}
        }

		System.out.println("Image range: 0.0 to " + maxDistance);
		return data;
	}
}
