/*
 * Created on Mar 26, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package clustering;

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
 */
public class WeightMatrix {

	int dimension = 0;
	int arraySize = 0;
	double[] weights = null;
	
	/**
	 * 
	 */
	public WeightMatrix(int dimension) {
		this.dimension = dimension;
		this.arraySize = (dimension * (dimension-1)) / 2;
		this.weights = new double[arraySize];
		for (int i = 0 ; i < arraySize ; i++) {
			this.weights[i] = 1.0;
		}
	}
	
	public double getWeight(int x, int y) {
		// if x == 0, then y == 0.
		// if x == 1, then y == 0 or 1.
		// if x == 2, then y == 0 or 1 or 2.  And so on...
		if (x == 0) return weights[0];
		int location = ((x * (x-1)) / 2) + y + 1;
		return weights[location];
	}
	
	public void setWeight(int x, int y, double value) {
		if (x == 0) weights[0] = value;
		int location = ((x * (x-1)) / 2) + y + 1;
		weights[location] = value;		
	}
	
}
