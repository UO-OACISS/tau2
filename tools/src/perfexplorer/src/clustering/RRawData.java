/*
 * Created on Mar 16, 2005
 *
 */
package clustering;

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
	private String[] threadNames = null;
	
	/**
	 * Default constructor.
	 */
	public RRawData(int vectors, int dimensions) {
		super();
		this.vectors = vectors;
		this.dimensions = dimensions;
		data = new double[vectors][dimensions];
		eventNames = new String[dimensions];
		threadNames = new String[dimensions];
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#addValue(int, int, double)
	 */
	public void addValue(int vectorIndex, int dimensionIndex, double value) {
		assert vectorIndex > 0 && vectorIndex < vectors : vectorIndex;
		assert dimensionIndex > 0 && dimensionIndex < dimensions : dimensionIndex;
		data[vectorIndex][dimensionIndex] = value;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getValue(int, int)
	 */
	public double getValue(int vectorIndex, int dimensionIndex) {
		assert vectorIndex > 0 && vectorIndex < vectors : vectorIndex;
		assert dimensionIndex > 0 && dimensionIndex < dimensions : dimensionIndex;
		return data[vectorIndex][dimensionIndex];
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getCartesianDistance(int, int)
	 */
	public double getCartesianDistance(int firstVector, int secondVector)  {
		assert firstVector > 0 && firstVector < vectors : firstVector;
		assert secondVector > 0 && secondVector < vectors : secondVector;
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
		assert firstVector > 0 && firstVector < vectors : firstVector;
		assert secondVector > 0 && secondVector < vectors : secondVector;

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

}
