/*
 * Created on Mar 16, 2005
 *
 */
package edu.uoregon.tau.perfexplorer.clustering;


 /**
  * This class is used to build a square distance matrix.  The distance matrix
  * is used to hold the distances between elements in a vector.  The matrix
  * is a diagonal matrix, and only the lower-left quadrants below the
  * diagonal are used.
  *
  * <P>CVS $Id: DistanceMatrix.java,v 1.7 2009/11/18 10:17:32 khuck Exp $</P>
  * @author khuck
  * @version 0.1
  * @since   0.1
  *
  */
public class DistanceMatrix {

    protected int dimension = 0;
    protected double[][] distances = null;
    protected double[] weights = null;
    
    /**
    * Default constructor - requires a dimension parameter.
    * 
    * @param dimension The height/width of the matrix.
    */
    public DistanceMatrix(int dimension) {
        this.dimension = dimension;
        this.distances = new double[dimension][dimension];
        this.weights = new double[dimension];
        for (int i = 0; i < dimension ; i++) {
            for (int j = i+1; j < dimension ; j++) {
                this.distances[i][j] = 0.0;
            }
            weights[i] = 1.0;
        }
    }
    
    /**
    * Copy constructor - requires a DistanceMatrix parameter.
    * 
    * @param distances Another distance matrix to copy.
    */
    public DistanceMatrix(DistanceMatrix distances) {
        this.dimension = distances.dimension;
        this.distances = new double[dimension][dimension];
        this.weights = new double[dimension];
        for (int i = 0; i < dimension ; i++) {
            for (int j = 0; j < dimension ; j++) {
                this.distances[i][j] = distances.distances[i][j];
            }
            weights[i] = distances.weights[i];
        }
    }
    
    /**
    * Accessor method to get the dimension of the distance matrix.
    * 
    * @return
    */
    public int getDimension() {
        return this.dimension;
    }
    
    /**
    * Accessor method to get the value at a specified location.
    * 
    * @param x
    * @param y
    * @return
    */
    public double elementAt(int x, int y) {
        return distances[x][y];
    }
    
    /**
    * Create a sparse matrix of Manhattan distances.  This method populates
    * the matrix with the cartesian distances between the items.
    * 
    * @param data
    */
    public void solveManhattanDistances(RawDataInterface data) {
		System.out.println("Using Manhattan distances...");
        for (int i = 0; i < dimension ; i++) {
            for (int j = 0; j < i ; j++) {
                distances[i][j] = distances[j][i] = data.getManhattanDistance(i,j);
            }
        }
    }
    
    /**
    * Create a sparse matrix of Cartesian distances.  This method populates 
    * the matrix with the cartesian distances between the items.
    * 
    * @param data
    */
    public void solveCartesianDistances(RawDataInterface data) {
		System.out.println("Using Cartesian distances...");
        for (int i = 0; i < dimension ; i++) {
            for (int j = 0; j < i ; j++) {
                distances[i][j] = distances[j][i] = data.getCartesianDistance(i,j);
            }
        }
    }
    
    /**
     * Useful method for debugging.
     * 
     * @return
     */
    public String toString() {
        StringBuilder buf = new StringBuilder();
        for (int i = 0; i < dimension ; i++) {
            for (int j = 0; j < dimension ; j++) {
                buf.append(distances[i][j] + " ");
            }
            buf.append("\n");
        }		
        return buf.toString();
    }
    
    /**
    * Merge the distances between two vectors
    * 
    * I would prefer that this method use the Lance Williams dissimilarity
    * update formula algorithm, if I could find it...
    * 
    * @param x
    * @param y
    */
    public void mergeDistances (int x, int y) {
    	// merge the horizontal distances
        for (int i = 0 ; i < dimension ; i++) {
            double firstWeight = weights[x];
            double secondWeight = weights[y];
            double totalWeight = firstWeight + secondWeight;
            distances[x][i] = ((distances[x][i] * firstWeight) + 
                    (distances[y][i] * firstWeight)) / totalWeight;
            weights[x] = totalWeight;
        }
    }
}
