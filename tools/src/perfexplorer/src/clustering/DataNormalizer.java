/**
 * Created Feb. 13, 2006
 */
package clustering;

/**
 * Interface for normalizing data.
 * This interface defines the methods for normalizing analysis data.
 * 
 * <P>CVS $Id: DataNormalizer.java,v 1.1 2007/01/04 21:20:01 khuck Exp $</P>
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
