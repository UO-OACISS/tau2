/*
 * Created on Mar 16, 2005
 *
 */
package clustering;

/**
 * @author khuck
 *
 */
public interface DimensionReductionInterface {
	/**
	 * This method performs the dimension reduction
	 * @throws ClusterException
	 */
	public void reduce() throws ClusterException;
	
	public void setInputData(RawDataInterface data);
	
	public RawDataInterface getOutputData();
}
