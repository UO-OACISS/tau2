/**
 * Created Feb. 13, 2006
 */
package edu.uoregon.tau.perfexplorer.clustering;


/**
 * Interface for normalizing data.
 * This interface defines the methods for normalizing analysis data.
 * 
 * <P>CVS $Id: DataNormalizer.java,v 1.3 2009/02/24 00:53:35 khuck Exp $</P>
 * @author khuck
 * @version 0.2
 * @since   0.2
 */
public interface DataNormalizer {
	/**
	 * This method normalizes the data and returns the results.
	 * 
	 * @return
	 */
    public RawDataInterface getNormalizedData(); 
}
