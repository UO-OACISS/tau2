/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import edu.uoregon.tau.perfdmf.Trial;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

//import javax.persistence.GeneratedValue;
//import javax.persistence.Id;

/**
 * @author khuck
 *
 */
public abstract class AbstractPSLOperation implements PerformanceAnalysisOperation {

	//@Id @GeneratedValue
	private Long id;
	
	protected List<PerformanceResult> inputs = null;

	/**
	 * 
	 */
	public AbstractPSLOperation() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * Constructor which includes the inputData object
	 * @param input
	 */
	protected AbstractPSLOperation(PerformanceResult input) {
		this.setInput(input);
		Provenance.addOperation(this);
	}

	/**
	 * Constructor which includes the inputData object
	 * @param input
	 */
	protected AbstractPSLOperation(Trial trial) {
		PerformanceResult input = new TrialResult(trial);
		this.setInput(input);
		Provenance.addOperation(this);
	}

	/**
	 * Constructor which includes the inputData object
	 * @param inputs
	 */
	protected AbstractPSLOperation(List<PerformanceResult> inputs) {
		this.setInputs(inputs);
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
	
	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

}
