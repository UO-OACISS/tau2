/**
 * 
 */
package glue;

import java.io.Serializable;

/**
 * This is a default implementation of the AbstractResult class.
 * 
 * <P>CVS $Id: DefaultResult.java,v 1.2 2008/03/05 00:25:54 khuck Exp $</P>
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

}
