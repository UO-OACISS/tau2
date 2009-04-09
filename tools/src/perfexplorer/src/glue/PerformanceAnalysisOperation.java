/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.List;

/**
 * <p>
 * The PerformanceAnalysisOperation interface is defined as the methods all
 * analysis operations should support.  All operations should be refered to
 * through this interface.
 * </p>
 * 
 * <P>CVS $Id: PerformanceAnalysisOperation.java,v 1.4 2009/04/09 00:23:51 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public interface PerformanceAnalysisOperation {
	/**
	 * Adds an input PerformanceResult for the analysis operation.
	 * @param input A PerformanceResult to be added to the list of
	 * PerformanceResults used as input for this operation.
	 * @see PerformanceResult
	 */
	public void addInput(PerformanceResult input) ;
	
	/**
	 * Sets the input PerformanceResult for the analysis operation. 
	 * @param input A PerformanceResult to be added to the list of
	 * PerformanceResults used as input for this operation.
	 * @see PerformanceResult
	 */
	public void setInput(PerformanceResult input) ;
	
	/**
	 * Sets a List of input PerformanceResults for the analysis operation. 
	 * @param inputs A List of PerformanceResult objects to be used
	 * as input for this operation.
	 * @see PerformanceResult
	 * @see java.util.List
	 */
	public void setInputs(List<PerformanceResult> inputs) ;
	
	/**
	 * Returns the List of PerformanceResult inputs for the analysis operation.
	 * @return the List of inputs
	 * @see PerformanceResult
	 * @see java.util.List
	 */
	public List<PerformanceResult> getInputs() ;

	/**
	 * Performs the analysis operation.
	 * @return a List of PerformanceResult outputs
	 * @see PerformanceResult
	 * @see java.util.List
	 */
	public List<PerformanceResult> processData() ;

	/**
	 * Returns a List of PerformanceResult objects, the output from the
	 * analysis operation.
	 * @return a List of PerformanceResult outputs
	 * @see PerformanceResult
	 * @see java.util.List
	 */
	public List<PerformanceResult> getOutputs() ;

	/**
	 * Sets the List of output PerformanceResult objects.
	 * @param outputs A List of PerformanceResult objects.
	 * @see PerformanceResult
	 * @see java.util.List
	 */
	public void setOutputs(List<PerformanceResult> outputs) ;

	/**
	 * Returns a particular PerformanceResult object from the list of
	 * PerfomanceResult output.
	 * @param index The index of the PerformanceResult
	 * @return the PerformanceResult output
	 * @see PerformanceResult
	 */
	public PerformanceResult getOutputAtIndex(int index) ;
	
	/**
	 * Returns a printable string of this analysis operation.
	 * @return The string
	 */
	public String toString();
	
	/**
	 * Resets the analysis operation to it's initial state.
	 */
	public void reset();

}
