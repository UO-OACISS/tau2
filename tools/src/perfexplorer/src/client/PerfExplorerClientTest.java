package edu.uoregon.tau.perfexplorer.client;

import junit.framework.TestCase;

public class PerfExplorerClientTest extends TestCase {

	public final void testPerfExplorerClient() {
		String home = System.getProperty("user.home");
		String slash = System.getProperty("file.separator");
		String configFile = home + slash + ".ParaProf" + slash + "perfdmf.cfg";

		PerfExplorerClient frame = new PerfExplorerClient(true, configFile, false);
		frame.pack();
		frame.setVisible(true);
		
		PerfExplorerActionListener listener = (PerfExplorerActionListener)frame.getListener();
		listener.setScriptName("rules/scdemo2.py");
		listener.runScript();
	}

}
