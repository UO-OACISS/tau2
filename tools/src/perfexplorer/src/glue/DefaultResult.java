/**
 * 
 */
package glue;

import java.io.Serializable;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * This is a default implementation of the AbstractResult class.
 * 
 * <P>CVS $Id: DefaultResult.java,v 1.6 2008/08/26 23:21:52 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0 
 */
public class DefaultResult extends AbstractResult implements Serializable {
	
	/**
	 * 
	 */
	public DefaultResult() {
		super();
	}
	
	/**
	 * @param input
	 */
	public DefaultResult(PerformanceResult input) {
		super(input);
	}

	public DefaultResult(PerformanceResult input, boolean notFullCopy) {
		super(input, notFullCopy);
	}

}
