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
	
	/**
	 * Default constructor.
	 */
	public RRawData(int vectors, int dimensions) {
		super();
		this.vectors = vectors;
		this.dimensions = dimensions;
		this.data = new double[vectors][dimensions];
		this.eventNames = new String[dimensions];
		//initialize();
	}

	public RRawData(int vectors, int dimensions, String[] eventNames) {
		super();
		this.vectors = vectors;
		this.dimensions = dimensions;
		this.data = new double[vectors][dimensions];
		this.eventNames = eventNames;
		//initialize();
	}

	public RRawData(int vectors, int dimensions, double[] inData) {
		super();
		this.vectors = vectors;
		this.dimensions = dimensions;
		this.eventNames = new String[dimensions];
		this.data = new double[vectors][dimensions];
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
		return data[vectorIndex][dimensionIndex];
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
}
