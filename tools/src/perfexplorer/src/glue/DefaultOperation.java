/**
 * 
 */
package glue;

import java.io.Serializable;
import java.util.List;

/**
 * This class represents a default operation example.
 * 
 * <P>CVS $Id: DefaultOperation.java,v 1.2 2008/03/05 00:25:54 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0
 */
public class DefaultOperation extends AbstractPerformanceOperation implements Serializable {

	public DefaultOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/**
	 * Constructor which includes the inputData object
	 * @param input
	 */
	public DefaultOperation(PerformanceResult input) {
		super(input);
	}


	/**
	 * Dummy implementation which is a no-op on the input data
	 */
	public List<PerformanceResult> processData() {
		outputs.add(inputs.get(0));
		return outputs;
	}

}
