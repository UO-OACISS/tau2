/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.List;

/**
 * This interface is defined as the methods all analysis operations
 * should support.  All operations should be refered to through
 * this interface.
 * 
 * <P>CVS $Id: PerformanceAnalysisOperation.java,v 1.3 2009/02/24 00:53:39 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0
 */
public interface PerformanceAnalysisOperation {
	/**
	 * Adds an input PerformanceResult for the analysis operation.
	 * @param input
	 */
	public void addInput(PerformanceResult input) ;
	
	/**
	 * Sets the input PerformanceResult for the analysis operation. 
	 * @param input
	 */
	public void setInput(PerformanceResult input) ;
	
	/**
	 * Sets a List of input PerformanceResults for the analysis operation. 
	 * @param inputs
	 */
	public void setInputs(List<PerformanceResult> inputs) ;
	
	/**
	 * Returns the List of PerformanceResult inputs for the analysis operation
	 * @return the list of inputs
	 */
	public List<PerformanceResult> getInputs() ;

	/**
	 * Performs the analysis operation.
	 * @return a List of PerformanceResult output
	 */
	public List<PerformanceResult> processData() ;

	/**
	 * Returns a List of PerformanceResult objects, the output from the analysis operation.
	 * @return a List of PerformanceResult output
	 */
	public List<PerformanceResult> getOutputs() ;

	/**
	 * Sets the List of output PerformanceResult objects.
	 */
	public void setOutputs(List<PerformanceResult> outputs) ;

	/**
	 * Returns a particular PerformanceResult object from the list of PerfomanceResult output.
	 * @param index
	 * @return the PerformanceResult output
	 */
	public PerformanceResult getOutputAtIndex(int index) ;
	
	/**
	 * Returns a printable string of this analysis operation.
	 * @return the string
	 */
	public String toString();
	
	/**
	 * Resets the analysis operation to it's initial state.
	 *
	 */
	public void reset();

}
