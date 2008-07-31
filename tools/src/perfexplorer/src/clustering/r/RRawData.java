/*
 * Created on Mar 16, 2005
 *
 */
package clustering.r;


import java.util.List;
import java.util.ArrayList;

import clustering.RawDataInterface;

import java.io.Serializable;

/**
 * This class is the R implementation of the analysis data class.
 * This class is package private - it should only be accessed from the
 * interface methods.  To access these methods, create an AnalysisFactory,
 * and the factory will be able to create a analysis data object.
 * This class is simply a container object which holds a matrix of
 * performance data.
 * The row and column headers are stored in this class, as well.
 *
 * <P>CVS $Id: RRawData.java,v 1.8 2008/07/31 18:43:48 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 */
public class RRawData implements RawDataInterface, Serializable {

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
	private RRawData() {}
	
    /**
     * Constructor restricted to package private.
     *
     * @param vectors
     * @param dimensions
     */
	RRawData(int vectors, int dimensions) {
		super();
		this.vectors = vectors;
		this.dimensions = dimensions;
		this.data = new double[vectors][dimensions];
		this.eventNames = new String[dimensions];
		this.mainData = new double[vectors];
		//initialize();
	}

    /**
     * constructor restricted to package private.
     *
     * @param vectors
     * @param dimensions
     * @param eventNames
     */
	public RRawData(int vectors, int dimensions, String[] eventNames) {
		super();
		this.vectors = vectors;
		this.dimensions = dimensions;
		this.data = new double[vectors][dimensions];
		this.eventNames = eventNames;
		this.mainData = new double[vectors];
		//initialize();
	}

    /**
     * Constructor restricted to package private.
     *
     * @param vectors
     * @param dimensions
     * @param inData
     */
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

    /**
     * initialization method.
     */
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

    /* (non-Javadoc)
     * @see clustering.RawDataInterface#getName()
     */
    public String getName() {
        return null;
    }

    /* (non-Javadoc)
     * @see clustering.RawDataInterface#getMaximum()
     */
	public double getMaximum() {
		return maximum;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getVector(int)
	 */
	public double[] getVector(int i) {
		return data[i];
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getCorrelation(int, int)
	 */
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

	/*
	 * (non-Javadoc)
	 * @see clustering.RawDataInterface#addMainValue(int, int, double)
	 */
    public void addMainValue(int vectorIndex, int eventIndex, double value) {
		mainData[vectorIndex] = value;
		this.mainEvent = eventIndex;
	}

    /*
     * (non-Javadoc)
     * @see clustering.RawDataInterface#getMainValue(int)
     */
	public double getMainValue(int vectorIndex) {
		return mainData[vectorIndex];
	}
		
	/*
	 * (non-Javadoc)
	 * @see clustering.RawDataInterface#getMainEventName()
	 */
	public String getMainEventName() {
		return new String(eventNames[this.mainEvent] + "(inclusive)");
	}

	public void addValue(int vectorIndex, int dimensionIndex, String value) {
		// TODO Auto-generated method stub
		System.err.println("addValue() not implemented for string attributes.");
	}
}
