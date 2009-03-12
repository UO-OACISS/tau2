/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;

import edu.uoregon.tau.perfdmf.Trial;


/**
 * This class is used as an abstract implementation of the PerformanceAnalysisOperation
 * interface.  This class has all the member data fields for the plethora
 * of anticipated subclasses.
 * 
 * <P>CVS $Id: AbstractPerformanceOperation.java,v 1.5 2009/03/12 20:49:52 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0 */

public abstract class AbstractPerformanceOperation implements PerformanceAnalysisOperation, Serializable {

	private Long id;
	
	protected List<PerformanceResult> inputs = null;
	protected List<PerformanceResult> outputs = null;
	
	/**
	 * Default constructor
	 */
	protected AbstractPerformanceOperation() {
		Provenance.addOperation(this);
	}

	/**
	 * Constructor which includes the inputData object
	 * @param input
	 */
	protected AbstractPerformanceOperation(PerformanceResult input) {
		if (input == null) {
			System.err.println("\n\n *** ERROR: Input Trial is null. ***\n\n");
		}
		this.setInput(input);
		this.outputs = new ArrayList<PerformanceResult>();
		Provenance.addOperation(this);
	}

	/**
	 * Constructor which includes the inputData object
	 * @param input
	 */
	protected AbstractPerformanceOperation(Trial trial) {
		PerformanceResult input = new TrialResult(trial);
		this.setInput(input);
		this.outputs = new ArrayList<PerformanceResult>();
		Provenance.addOperation(this);
	}

	/**
	 * Constructor which includes the inputData object
	 * @param inputs
	 */
	protected AbstractPerformanceOperation(List<PerformanceResult> inputs) {
		this.setInputs(inputs);
		this.outputs = new ArrayList<PerformanceResult>();
		Provenance.addOperation(this);
	}

	public List<PerformanceResult> getInputs() {
		return inputs;
	}

	public void setInputs(List<PerformanceResult> inputs) {
		this.inputs = inputs;
	}

	public void setInputsTrials(List<Trial> trials) {
		this.inputs = new ArrayList<PerformanceResult>();
		for (Trial trial : trials) {
			PerformanceResult input = new TrialResult(trial);
			this.addInput(input);
		}
	}

	public void setInput(PerformanceResult input) {
		this.inputs = new ArrayList<PerformanceResult>();
		this.inputs.add(input);
	}
	
	public void setInput(Trial trial) {
		this.inputs = new ArrayList<PerformanceResult>();
		PerformanceResult input = new TrialResult(trial);
		this.inputs.add(input);
	}
	
	public void addInput(PerformanceResult input) {
		this.inputs.add(input);
	}
	
	public void addInput(Trial trial) {
		PerformanceResult input = new TrialResult(trial);
		this.inputs.add(input);
	}
	
	public List<PerformanceResult> getOutputs() {
		return outputs;
	}

	public void setOutputs(List<PerformanceResult> outputs) {
		this.outputs = outputs;
	}

	public PerformanceResult getOutputAtIndex(int index) {
		return outputs.get(index);
	}

	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public String toString() {
		return this.getClass().getName();
	}

	public void reset() {
		outputs.clear();
	}
	
}
