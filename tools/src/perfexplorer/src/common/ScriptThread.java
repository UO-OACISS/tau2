package edu.uoregon.tau.perfexplorer.common;

import edu.uoregon.tau.common.PythonInterpreterFactory;

/**
 * This is the main server thread which processes long-executing analysis 
 * requests.  It is created by the PerfExplorerServer object, and 
 * checks the queue every 1 seconds to see if there are any new requests.
 *
 * <P>CVS $Id: ScriptThread.java,v 1.2 2009/02/24 00:53:37 khuck Exp $</P>
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
	 * run method.  When the thread wakes up, this method is executed.
	 * This method creates an PythonInterpreter object, and runs
	 * the specified script.
	 */
	public void run() {
		PythonInterpreterFactory.defaultfactory.getPythonInterpreter().execfile(scriptName);
	}
}
