/*
 * Created on Mar 16, 2005
 *
 */
package clustering.r;

import clustering.*;
import java.util.List;
import java.util.ArrayList;

/**
 * The RawData class.
 * This class is simply a container object which holds a matrix of performance data.
 * The row and column headers are stored in this class, as well.
 * 
 * @author khuck
 *
 */
public class RRawData implements RawDataInterface {

	private int vectors = 0;
	private int dimensions = 0;
	private double[][] data = null;
	private String[] eventNames = null;
	private double maximum = 0.0;
	private double[] mainData = null;
	private int mainEvent = 0;
	private boolean normalize = false;
	private double ranges[][] = null;
	
	/**
	 * Default constructor.
	 */
	public RRawData(int vectors, int dimensions) {
		super();
		this.vectors = vectors;
		this.dimensions = dimensions;
		this.data = new double[vectors][dimensions];
		this.eventNames = new String[dimensions];
		this.mainData = new double[vectors];
		//initialize();
	}

	public RRawData(int vectors, int dimensions, String[] eventNames) {
		super();
		this.vectors = vectors;
		this.dimensions = dimensions;
		this.data = new double[vectors][dimensions];
		this.eventNames = eventNames;
		this.mainData = new double[vectors];
		//initialize();
	}

	public RRawData(int vectors, int dimensions, double[] inData) {
		super();
		this.vectors = vectors;
		this.dimensions = dimensions;
		this.eventNames = new String[dimensions];
		this.data = new double[vectors][dimensions];
		this.mainData = new double[vectors];
		int i = 0;
		for (int x = 0 ; x < vectors ; x++) {
			for (int y = 0 ; y < dimensions ; y++) {
				data[x][y] = inData[i++];
			}
		}
	}

	public void initialize() {
		for (int x = 0 ; x < vectors ; x++) {
			for (int y = 0 ; y < dimensions ; y++) {
				data[x][y] = 0.0;
			}
		}
	}
	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#addValue(int, int, double)
	 */
	public void addValue(int vectorIndex, int dimensionIndex, double value) {
		//assert vectorIndex >= 0 && vectorIndex < vectors : vectorIndex;
		//assert dimensionIndex >= 0 && dimensionIndex < dimensions : dimensionIndex;
		data[vectorIndex][dimensionIndex] = value;
		if (maximum < value)
			maximum = value;
	}

	public void addEventNames(String[] eventNames) {
		this.eventNames = eventNames;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getValue(int, int)
	 */
	public double getValue(int vectorIndex, int dimensionIndex) {
		// assert vectorIndex > 0 && vectorIndex < vectors : vectorIndex;
		// assert dimensionIndex > 0 && dimensionIndex < dimensions : dimensionIndex;
		if (normalize) {
			double tmp = data[vectorIndex][dimensionIndex];
			// subtract the min
			tmp = tmp - ranges[dimensionIndex][0];
			// divide by the range
			tmp = tmp / ranges[dimensionIndex][1];
			return tmp;
		} else {
			return data[vectorIndex][dimensionIndex];
		}
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getCartesianDistance(int, int)
	 */
	public double getCartesianDistance(int firstVector, int secondVector)  {
		// assert firstVector > 0 && firstVector < vectors : firstVector;
		// assert secondVector > 0 && secondVector < vectors : secondVector;
		double distance = 0.0;
		for (int i = 0 ; i < dimensions ; i++ ) {
			double tmp = Math.abs(data[firstVector][i] - data[secondVector][i]);
			distance += tmp * tmp;
		}
		return Math.sqrt(distance);
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getManhattanDistance(int, int)
	 */
	public double getManhattanDistance(int firstVector, int secondVector) {
		// assert firstVector > 0 && firstVector < vectors : firstVector;
		// assert secondVector > 0 && secondVector < vectors : secondVector;

		double distance = 0.0;
		for (int i = 0 ; i < dimensions ; i++ ) {
			distance += Math.abs(data[firstVector][i] - data[secondVector][i]);
		}
		return distance;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getData()
	 */
	public Object getData() {
		return data;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getEventNames()
	 */
	public List getEventNames() {
		List names = new ArrayList (eventNames.length);
		for (int i = 0 ; i < eventNames.length ; i++) {
			names.add(eventNames[i]);
		}
		return names;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#numVectors()
	 */
	public int numVectors() {
		return this.vectors;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#numDimensions()
	 */
	public int numDimensions() {
		return this.dimensions;
	}

	public double getMaximum() {
		return maximum;
	}

	public double[] getVector(int i) {
		return data[i];
	}

	public double getCorrelation (int x, int y) {
		double r = 0.0;
		double xAvg = 0.0;
		double yAvg = 0.0;
		double xStDev = 0.0;
		double yStDev = 0.0;
		double sum = 0.0;

		for (int i = 0 ; i < vectors ; i++ ) {
			xAvg += data[i][x];
			yAvg += data[i][y];
		}

		// find the average for the first vector
		xAvg = xAvg / vectors;
		// find the average for the second vector
		yAvg = yAvg / vectors;


		for (int i = 0 ; i < vectors ; i++ ) {
			xStDev += (data[i][x] - xAvg) * (data[i][x] - xAvg);
			yStDev += (data[i][y] - yAvg) * (data[i][y] - yAvg);
		}

		// find the standard deviation for the first vector
		xStDev = xStDev / (vectors - 1);
		xStDev = Math.sqrt(xStDev);
		// find the standard deviation for the second vector
		yStDev = yStDev / (vectors - 1);
		yStDev = Math.sqrt(yStDev);


		// solve for r
		double tmp1 = 0.0;
		double tmp2 = 0.0;
		for (int i = 0 ; i < vectors ; i++ ) {
			tmp1 = (data[i][x] - xAvg) / xStDev;
			tmp2 = (data[i][y] - yAvg) / yStDev;
			r += tmp1 * tmp2;
		}
		r = r / (vectors - 1);

		//System.out.println("Avg(x) = " + xAvg + ", Avg(y) = " + yAvg);
		//System.out.println("Stddev(x) = " + xStDev + ", Stddev(y) = " + yStDev);
		//System.out.println("r = " + r);

		return r;
	}

    public void addMainValue(int vectorIndex, int eventIndex, double value) {
		mainData[vectorIndex] = value;
		this.mainEvent = eventIndex;
	}

	public double getMainValue(int vectorIndex) {
		return mainData[vectorIndex];
	}
		
	public String getMainEventName() {
		return new String(eventNames[this.mainEvent] + "(inclusive)");
	}

	public void normalizeData(boolean normalize) {
		this.normalize = normalize;
		if (normalize) {
			// calcuate the ranges
			ranges = new double[dimensions][2];		

			for (int i = 0 ; i < dimensions ; i++ ) {
				ranges[i][0] = data[0][i];
				ranges[i][1] = data[0][i];
				for (int j = 0 ; j < vectors ; j++ ) {
					// check against the min
					if (ranges[i][0] > data[j][i])
						ranges[i][0] = data[j][i];
					// check against the max
					if (ranges[i][1] < data[j][i])
						ranges[i][1] = data[j][i];
				}
				// subtract the min from the max
				ranges[i][1] = ranges[i][1] - ranges[i][0];
			}
		}
	}
}
