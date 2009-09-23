package edu.uoregon.tau.perfexplorer.common;

import java.io.InputStream;

import edu.uoregon.tau.common.PythonInterpreterFactory;
import org.python.util.PythonInterpreter;

/**
 * This is the main server thread which processes long-executing analysis 
 * requests.  It is created by the PerfExplorerServer object, and 
 * checks the queue every 1 seconds to see if there are any new requests.
 *
 * <P>CVS $Id: ScriptThread.java,v 1.4 2009/09/23 19:31:27 smillst Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 * @see     PerfExplorerServer
 */
public class ScriptThread extends Thread {

	/**
	 *  retain some stuff
	 */
	private String scriptName = null;
	private String script = null;

	/**
	 * Constructor.  Expects a reference to a PerfExplorerServer.
	 * @param server
	 */
	public ScriptThread (String scriptName) {
		super();
		this.scriptName = scriptName;
		start();
	}

	/**
	 * Constructor.  Expects a reference to a PerfExplorerServer.
	 * @param server
	 */
	public ScriptThread (String script, boolean notFile) {
		super();
		this.script = script;
		start();
	}
	/**
	 * run method.  When the thread wakes up, this method is executed.
	 * This method creates an PythonInterpreter object, and runs
	 * the specified script.
	 */
	public void run() {

		String translate;
		try {
			if(script == null){
				translate = TranslateScript.translate(scriptName);
				PythonInterpreterFactory.defaultfactory.getPythonInterpreter().exec(translate);
			}else{
				PythonInterpreterFactory.defaultfactory.getPythonInterpreter().exec(script);
			}
				
		} catch (EquationParseException e) {
			// TODO Auto-generated catch block
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	
		
	}
}
