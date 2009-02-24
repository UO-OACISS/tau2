/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.common.PythonInterpreterFactory;

/**
 * @author khuck
 *
 */
public class Script {
	private static Script theInstance = null;
	private List<Object> parameters = new ArrayList<Object>();
	
	private Script() {
		super();
	}
	
	public static Script getInstance() {
		if (theInstance == null) {
			theInstance = new Script();
		}
		
		return theInstance;
	}
	
	public void addParameter(Object obj) {
		parameters.add(obj);
	}

	public List<Object> getParameters() {
		return parameters;
	}
	
	public void clearParameters() {
		parameters = new ArrayList<Object>();
	}
	
	public static void execute(String scriptName) {
		PythonInterpreterFactory.defaultfactory.getPythonInterpreter().execfile(scriptName);
	}
}
