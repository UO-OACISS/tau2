/**
 * 
 */
package edu.uoregon.tau.perfexplorer.clustering;

import java.util.List;

/**
 * @author khuck
 *
 */
public interface LinearRegressionInterface {

	public void findCoefficients() ;
	
	public List<Double> getCoefficients();

    /**
     * Sets the input data for the clustering operation.
     * 
     * @param inputData
     */
    public void setInputData(RawDataInterface inputData);


}
