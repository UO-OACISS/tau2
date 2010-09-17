/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;

import java.util.Iterator;
import java.util.LinkedHashMap;

/**
 * This class is used as a list of names and values to describe 
 * a cluster created during some type of clustering operation.
 * 
 * <P>CVS $Id: ClusterDescription.java,v 1.3 2009/02/24 00:53:34 khuck Exp $</P>
 * @author khuck
 *
 */
public class ClusterDescription {

	private LinkedHashMap<String,Double> attributes = null;
	// set some defaults for the linked hash map - see java doc for details
	private final int defaultSize = 16;
	private final float floatFactor = 0.75F;
	private final boolean accessOrder = false; // insertion order!
	
	/**
	 * Default constructor
	 */
	public ClusterDescription() {
		attributes = new LinkedHashMap<String, Double>(defaultSize, floatFactor, accessOrder);
	}

	/**
	 * Constructor which specifies the number of attributes in this cluster
	 */
	public ClusterDescription(int numAttributes) {
		attributes = new LinkedHashMap<String, Double>(numAttributes, floatFactor, accessOrder);
	}
	
	/**
	 * Returns an Iterator of the attribute names.
	 * 
	 * @return
	 */
	public Iterator<String> getAttributeNames() {
		return attributes.keySet().iterator();
	}
	
	/**
	 * Returns the value of the object identified by the key.
	 * 
	 * @param key
	 * @return
	 */
	public double getValue(String key) {
		Double temp = attributes.get(key);
		return temp.doubleValue();
	}
}
