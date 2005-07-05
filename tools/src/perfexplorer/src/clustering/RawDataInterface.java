/*
 * Created on Mar 16, 2005
 *
 */
package clustering;

import java.util.List;

/**
 * @author khuck
 *
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

	public List getEventNames();
	
	public int numVectors();
	
	public int numDimensions();
}
