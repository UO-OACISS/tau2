package edu.uoregon.tau.perfexplorer.client;

import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;

public class PerfExplorerNoGUI {
//	private static String tauHome;
//	private static String tauArch;
//	private final PerfExplorerConnection connection;
//	private final PerfExplorerModel model;

	public PerfExplorerNoGUI (String configFile, boolean quiet) {
		PerfExplorerOutput.setQuiet(quiet);
		PerfExplorerConnection.setStandalone(true);
		PerfExplorerConnection.setConfigFile(configFile);
		//connection = 
			PerfExplorerConnection.getConnection();
		//model = 
			PerfExplorerModel.getModel();
	}
}
