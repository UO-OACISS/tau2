/**
 * Created on Mar 16, 2005
 *
 */
package clustering;

 /**
  * This interface defines the methods to be used for implementing dimension
  * reduction in the analysis methods.
  *
  * <P>CVS $Id: DimensionReductionInterface.java,v 1.2 2007/01/04 21:20:01 khuck Exp $</P>
  * @author khuck
  * @version 0.1
  * @since   0.1
  *
  */
public interface DimensionReductionInterface {
    /**
     * This method performs the dimension reduction
     * 
     * @throws ClusterException
     */
    public void reduce() throws ClusterException;
	
    /**
     * This method sets the input data to be reduced.
     * 
     * @param data
     */
    public void setInputData(RawDataInterface data);

    /**
     * This method returns the reduced data.
     * 
     * @return
     */
    public RawDataInterface getOutputData();
}
