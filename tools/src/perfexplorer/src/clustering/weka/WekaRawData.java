/*
 * Created on Mar 17, 2005
 *
 */
package clustering.weka;

import clustering.*;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.FastVector;
import java.util.List;
import java.util.ArrayList;
import weka.core.Attribute;
import java.util.Enumeration;

/**
 * @author khuck
 *
 */
public class WekaRawData implements RawDataInterface {

	private Instances instances = null;
	private int vectors = 0;
	private int dimensions = 0;
	private double maximum = 0.0;
	
	public WekaRawData (String name, List attributes, int vectors, int dimensions) {
		this.vectors = vectors;
		this.dimensions = dimensions;
		FastVector fastAttributes = new FastVector(attributes.size());
		for (int i = 0 ; i < attributes.size() ; i++) {
			String attr = (String) attributes.get(i);
			fastAttributes.addElement(new Attribute(attr));
		}
		instances = new Instances(name, fastAttributes, vectors);
		
		for (int i = 0 ; i < vectors ; i++) {
			instances.add(new Instance(dimensions));
		}
		
	}
	
	public WekaRawData (Instances instances) {
		this.instances = instances;
		this.vectors = instances.numInstances();
		if (this.vectors > 0)
			this.dimensions = instances.instance(0).numAttributes();
	}
	
	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#addValue(int, int, double)
	 */
	public void addValue(int vectorIndex, int dimensionIndex, double value) {
		Instance i = instances.instance(vectorIndex);
		i.setValue(dimensionIndex, value);
		if (maximum < value)
			maximum = value;
	}

	/* (non-Javadoc)
	 * @see clustering.RawDataInterface#getValue(int, int)
	 */
	public double getValue(int vectorIndex, int dimensionIndex) {
		return instances.instance(vectorIndex).value(dimensionIndex);
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

	public double getMaximum() {
		return maximum;
	}

	public double[] getVector(int i) {
		double[] data = new double[dimensions];
		for (int j = 0 ; j < dimensions ; j++) {
			data[j] = instances.instance(i).value(j);
		}
		return data;
	}

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

	public void addMainValue(int threadIndex, int eventIndex, double value) {
	}

	public double getMainValue(int threadIndex) {
		return 0.0;
	}

	public String getMainEventName() {
		String name = new String("");
		return name;
	}
}
