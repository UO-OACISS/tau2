/**
 * 
 */
package glue;

import java.io.Serializable;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * This is a default implementation of the AbstractResult class.
 * 
 * <P>CVS $Id: DefaultResult.java,v 1.3 2008/07/29 23:40:18 khuck Exp $</P>
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
		System.out.println("HERE!");
		Exception e = new Exception("WTF");
		e.printStackTrace();
	}
	
	/**
	 * @param input
	 */
	public DefaultResult(PerformanceResult input) {
		super(input);
	}

	public DefaultResult(Trial trial) {
		super(trial);
	}
}
