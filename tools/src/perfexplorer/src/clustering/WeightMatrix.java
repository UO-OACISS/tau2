/*
 * Created on Mar 26, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;

/**
 * @author khuck
 *
 * This class defines a square diagonal matrix of weights.
 * The values are stored in a single-dimension array, indexed
 * as such:
 * 0
 * 1 2
 * 3 4 5
 * 6 7 8 9
 * etc...
 *  
 * <P>CVS $Id: WeightMatrix.java,v 1.3 2009/02/24 00:53:35 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class WeightMatrix {

    int dimension = 0;
    int arraySize = 0;
    double[] weights = null;
    
    /**
     * Default constructor for the matrix, shouldn't be called, 
     * so make it private.
     */
    @SuppressWarnings("unused")
	private WeightMatrix() {};
    
    /**
     * Constructor for the matrix.
     * @param dimension
     */
    public WeightMatrix(int dimension) {
        this.dimension = dimension;
        this.arraySize = (dimension * (dimension-1)) / 2;
        this.weights = new double[arraySize];
        for (int i = 0 ; i < arraySize ; i++) {
            this.weights[i] = 1.0;
        }
    }
    
    /**
     * Returns the weight value at location "x","y".
     * @param x
     * @param y
     * @return
     */
    public double getWeight(int x, int y) {
        // if x == 0, then y == 0.
        // if x == 1, then y == 0 or 1.
        // if x == 2, then y == 0 or 1 or 2.  And so on...
        if (x == 0) return weights[0];
        int location = ((x * (x-1)) / 2) + y + 1;
        return weights[location];
    }
    
    /**
     * Sets the weight value at location "x","y".
     * @param x
     * @param y
     * @param value
     */
    public void setWeight(int x, int y, double value) {
        if (x == 0) weights[0] = value;
        int location = ((x * (x-1)) / 2) + y + 1;
        weights[location] = value;		
    }
	
}
