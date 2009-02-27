/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * For this class, do we know what we want to store, or do
 * we just store reflection information?   For example, should
 * we store the name of the class, and some information for
 * re-constructing it?
 * 
 * <P>CVS $Id: Provenance.java,v 1.8 2009/02/27 23:22:16 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0 
 */

//@Entity
public class Provenance {
	private static Provenance current = null;
	private static boolean enabled = false;

	//@Id @GeneratedValue
	private Long id = null;
	
	private Date date = null;

	//@AnyToMany(cascade = CascadeType.ALL)
	private List<PerformanceAnalysisOperation> operations = null;
	
	private Provenance(boolean empty) {
		if (!empty) {
			this.date = new Date();
			this.operations = new ArrayList<PerformanceAnalysisOperation>();
			current = this;
		}
	}
	
	public static Provenance getCurrent() {
		if (current == null) {
			current = new Provenance(false);
		}
		return current;
	}
	
	// make this package-private, so only glue objects can do this.
	static void addOperation(PerformanceAnalysisOperation operation) {
		Provenance current = getCurrent();
		if (!current.operations.contains(operation) && enabled) {
			current.operations.add(operation);
		}
	}

	public Long getId() {
		return id;
	}

	public List<PerformanceAnalysisOperation> getOperations() {
		return operations;
	}

	public Date getDate() {
		return date;
	}

	public void setDate(Date date) {
		this.date = date;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public void setOperations(List<PerformanceAnalysisOperation> operations) {
		this.operations = operations;
	}

	public static void save() {
		// TODO - this should get a list of operations, and all the options of how
		// to recreate it, and save it to the trial as XML metadata.
	}
	
	public static void listAll() {
		//DB4OUtilities.listAll(new Provenance(true));
	}
	
	public String toString() {
		StringBuilder buf = new StringBuilder();
		if (id == null) {
			buf.append(date.toString());
		} else {
			buf.append(date.toString() + " " + id.toString());
		}
		
		for (PerformanceAnalysisOperation operation : operations) {
			buf.append("\n\toperation: " + operation.toString());
			
			for (PerformanceResult input : operation.getInputs()) {
				buf.append("\n\t\tinput: " + input.toString());
			}
			for (PerformanceResult output : operation.getOutputs()) {
				buf.append("\n\t\toutput: " + output.toString());
			}
		}
		return buf.toString();
	}
	
	public static PerformanceAnalysisOperation getLastOperation() {
		PerformanceAnalysisOperation output = null;
		Provenance provenance = Provenance.getCurrent();
		int size = provenance.getOperations().size();
		if (size > 0) {
			output = provenance.getOperations().get(size - 1);
		}
		return output;
	}

	public static List<PerformanceResult> getLastOutput() {
		List<PerformanceResult> outputs = null;
		PerformanceAnalysisOperation last = Provenance.getLastOperation();
		if (last != null) {
			outputs = last.getOutputs();
		}
		return outputs;
	}

	public static void setEnabled (boolean enabled) {
		Provenance.enabled = enabled;
	}

}
