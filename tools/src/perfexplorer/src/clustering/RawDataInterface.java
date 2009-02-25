/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;

import java.util.List;

/**
 * This interface is used to define the methods to implement a class
 * which holds clustering data, whether it be input data or
 * cluster output data, or PCA output data.
 * 
 * @author khuck
 * <P>CVS $Id: RawDataInterface.java,v 1.9 2009/02/25 19:51:46 wspear Exp $</P>
 * @version 0.1
 * @since 0.1
 */
public interface RawDataInterface {
	/**
	 * Add the value to the object, at the specified indices.
	 * 
	 * @param vectorIndex
	 * @param dimensionIndex
	 * @param value
	 * @
	 */
	public void addValue(int vectorIndex, int dimensionIndex, double value) ;

	/**
	 * Get the value from the object at the specified indices.
	 * 
	 * @param vectorIndex
	 * @param dimensionIndex
	 * @return
	 * @
	 */
	public double getValue(int vectorIndex, int dimensionIndex) ;

	/**
	 * Get the distance between the two vectors.
	 * The distance calculated should be a simple Manhattan distance calculation.
	 * 
	 * @param firstVector
	 * @param secondVector
	 * @return
	 * @
	 */
	public double getManhattanDistance(int firstVector, int secondVector) ;

	/**
	 * Get the distance between the two vectors.
	 * The distance calculated should be a simple Cartesian distance calculation.
	 * 
	 * @param firstVector
	 * @param secondVector
	 * @return
	 * @
	 */
	public double getCartesianDistance(int firstVector, int secondVector) ;
	
	/**
	 * Get the data structure which stores the data.
	 * This makes doing the clustering easier for the respective engines.
	 * 
	 * @return
	 */
	public Object getData();

	/**
	 * Returns the dimension names in the data.
	 *
     * @return
     */
	public List<String> getEventNames();
	
	/**
     * Returns the number of vectors in this data set.
     *
     * @return
     */
	public int numVectors();
	
	/**
     * Returns the number of dimensions in this data set.
     *
     * @return
     */
	public int numDimensions();

	/**
     * Returns the relation name from the data set.
     *
     * @return
     */
     public String getName();

    /**
     * Gets the maximum value for the entire data set.
     *
     * @return
     */
	public double getMaximum();
	
	/**
     * Returns the vector of data at index "i".
     *
     * @param i
     * @return
     */
	public double[] getVector(int i);
		
	/**
     * Returns the correlation coefficient between vectors "x" and "y".
     *
     * @param x
     * @param y
     * @return
     */
	public double getCorrelation(int x, int y);
	
	/**
     * Adds the value to the dimension "eventIndex" on vector "threadIndex".
     *
     * @param threadIndex
     * @param eventIndex
     * @param value
     */
	public void addMainValue(int threadIndex, int eventIndex, double value);
	
	/**
     * Gets the inclusive value of the main function at vector "threadIndex".
     *
     * @param threadIndex
     * @return
     */
	public double getMainValue(int threadIndex);
	
	/**
     * Gets the name of the main function.
     *
     * @return
     */
	public String getMainEventName();

	/**
	 * Sets an attribute in the instance, where the attribute is a string.
	 * @param thread
	 * @param i
	 * @param name
	 */
	public void addValue(int vectorIndex, int dimensionIndex, String value);
}
