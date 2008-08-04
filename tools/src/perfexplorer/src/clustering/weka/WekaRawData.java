/*
 * Created on Mar 17, 2005
 *
 */
package clustering.weka;


import weka.core.Instances;
import weka.core.Instance;
import weka.core.FastVector;
import weka.core.Attribute;

import java.util.List;
import java.util.ArrayList;
import java.util.Enumeration;

import clustering.RawDataInterface;

import java.io.Serializable;

/**
 * Implementation of the RawData interface for Weka data.
 *
 * <P>CVS $Id: WekaRawData.java,v 1.11 2008/08/04 22:46:28 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 */
public class WekaRawData implements RawDataInterface, Serializable {

	private Instances instances = null;
	private int vectors = 0;
	private int dimensions = 0;
	private double maximum = 0.0;
	private boolean normalize = false;
	private double ranges[][] = null;
	
    /**
     * Package private constructor.
     * @param name
     * @param attributes
     * @param vectors
     * @param dimensions
     */
	WekaRawData (String name, List attributes, int vectors, int dimensions, List<String> classAttributes) {
		this.vectors = vectors;
		this.dimensions = dimensions;
		FastVector fastAttributes = new FastVector(attributes.size());
		for (int i = 0 ; i < attributes.size() ; i++) {
			String attr = (String) attributes.get(i);
			fastAttributes.addElement(new Attribute(attr));
		}
		Attribute tmp = null;
		if (classAttributes != null) {
			String attr = "class";
			FastVector vect = new FastVector(classAttributes.size());
			for (String tmpClass : classAttributes) {
				vect.addElement(tmpClass);
			}
			tmp = new Attribute(attr, vect);
			fastAttributes.addElement(tmp);
		}
		
		instances = new Instances(name, fastAttributes, vectors);
		
		if (classAttributes != null) {
			instances.setClass(tmp);
			for (int i = 0 ; i < vectors ; i++) {
				Instance tmpInst = new Instance(fastAttributes.size());
				tmpInst.setDataset(instances);
				for (int j = 0 ; j < dimensions ; j++) {
					tmpInst.setValue(j, 0.0);
				}
				instances.add(tmpInst);
			}
		} else {
			for (int i = 0 ; i < vectors ; i++) {
				double[] values = new double[dimensions];
				for (int j = 0 ; j < dimensions ; j++) {
					values[j] = 0.0;
				}
				instances.add(new Instance(1.0, values));
			}
		}
	}
	
    /**
     * Package private constructor - more or less a copy constructor.
     *
     * @param instances
     */
    WekaRawData (Instances instances) {
		this.instances = instances;
		this.vectors = instances.numInstances();
		if (this.vectors > 0)
			this.dimensions = instances.instance(0).numAttributes();
	}
	
	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#addValue(int, int, double)
	 */
	public void addValue(int vectorIndex, int dimensionIndex, double value) {
		//System.out.println("RawData: " + vectorIndex + ", " + dimensionIndex + ", " + value);
		Instance i = instances.instance(vectorIndex);
		i.setValue(dimensionIndex, value);
		if (maximum < value)
			maximum = value;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getValue(int, int)
	 */
	public double getValue(int vectorIndex, int dimensionIndex) {
		if (normalize) {
			double tmp = instances.instance(vectorIndex).value(dimensionIndex);
			// subtract the min
			tmp = tmp - ranges[dimensionIndex][0];
			// divide by the range
			tmp = tmp / ranges[dimensionIndex][1];
			return tmp;
		} else {
			return instances.instance(vectorIndex).value(dimensionIndex);
		}
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getManhattanDistance(int, int)
	 */
	public double getManhattanDistance(int firstVector, int secondVector) {
		//assert firstVector >= 0 && firstVector < vectors : firstVector;
		//assert secondVector >= 0 && secondVector < vectors : secondVector;

		double distance = 0.0;
		for (int i = 0 ; i < dimensions ; i++ ) {
			distance += Math.abs(instances.instance(firstVector).value(i) - 
					instances.instance(secondVector).value(i));
		}
		return distance;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getCartesianDistance(int, int)
	 */
	public double getCartesianDistance(int firstVector, int secondVector) {
		//assert firstVector > 0 && firstVector < vectors : firstVector;
		//assert secondVector > 0 && secondVector < vectors : secondVector;
		double distance = 0.0;
		for (int i = 0 ; i < dimensions ; i++ ) {
			double tmp = Math.abs(instances.instance(firstVector).value(i) - 
					instances.instance(secondVector).value(i));
			distance += tmp * tmp;
		}
		return Math.sqrt(distance);
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getData()
	 */
	public Object getData() {
		return instances;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getEventNames()
	 */
	public List getEventNames() {
		Enumeration e = instances.enumerateAttributes();
		List names = new ArrayList (instances.numDistinctValues(0));
		while (e.hasMoreElements()) {
			Attribute tmp = (Attribute) e.nextElement();
			names.add(tmp.name());
		}
		return names;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#numVectors()
	 */
	public int numVectors() {
		return vectors;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#numDimensions()
	 */
	public int numDimensions() {
		return dimensions;
	}

    /* (non-Javadoc)
     * @see clustering.RawDataInterface#getName()
     */
    public String getName() {
        return this.instances.relationName();
    }

    /* (non-Javadoc)
     * @see clustering.RawDataInterface#getMaximum()
     */
	public double getMaximum() {
		return maximum;
	}

    /* (non-Javadoc)
     * @see clustering.RawDataInterface#getVector()
     */
	public double[] getVector(int i) {
		double[] data = new double[dimensions];
		for (int j = 0 ; j < dimensions ; j++) {
			data[j] = instances.instance(i).value(j);
		}
		return data;
	}

    /* (non-Javadoc)
     * @see clustering.RawDataInterface#getCorrelation()
     */
	public double getCorrelation (int x, int y) {
		double r = 0.0;
		double xAvg = 0.0;
		double yAvg = 0.0;
		double xStDev = 0.0;
		double yStDev = 0.0;
		double sum = 0.0;

		for (int i = 0 ; i < vectors ; i++ ) {
			xAvg += instances.instance(i).value(x);
			yAvg += instances.instance(i).value(y);
		}

		// find the average for the first vector
		xAvg = xAvg / vectors;
		// find the average for the second vector
		yAvg = yAvg / vectors;


		for (int i = 0 ; i < vectors ; i++ ) {
			xStDev += (instances.instance(i).value(x) - xAvg) * (instances.instance(i).value(x) - xAvg);
			yStDev += (instances.instance(i).value(y) - yAvg) * (instances.instance(i).value(y) - yAvg);
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
			tmp1 = (instances.instance(i).value(x) - xAvg) / xStDev;
			tmp2 = (instances.instance(i).value(y) - yAvg) / yStDev;
			r += tmp1 * tmp2;
		}
		r = r / (vectors - 1);

		//System.out.println("Avg(x) = " + xAvg + ", Avg(y) = " + yAvg);
		//System.out.println("Stddev(x) = " + xStDev + ", Stddev(y) = " + yStDev);
		//System.out.println("r = " + r);

		return r;
	}

    /* (non-Javadoc)
     * @see clustering.RawDataInterface#addMainValue()
     */
	public void addMainValue(int threadIndex, int eventIndex, double value) {
	}

    /* (non-Javadoc)
     * @see clustering.RawDataInterface#getMainValue()
     */
	public double getMainValue(int threadIndex) {
		return 0.0;
	}

    /* (non-Javadoc)
     * @see clustering.RawDataInterface#getMainEventName()
     */
	public String getMainEventName() {
		String name = new String("");
		return name;
	}

	public void addValue(int vectorIndex, int dimensionIndex, String value) {
		Instance i = instances.instance(vectorIndex);
		i.setValue(dimensionIndex, value);
	}
}
