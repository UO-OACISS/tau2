package edu.uoregon.tau.perfexplorer.client;

import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorerModel;

public class PerfExplorerModel extends RMIPerfExplorerModel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6644066993143334180L;
	private static PerfExplorerModel theModel = null;

	public static PerfExplorerModel getModel() {
		if (theModel == null) {
			theModel = new PerfExplorerModel();
		}
		return theModel;
	}

	public RMIPerfExplorerModel copy() {
		return new RMIPerfExplorerModel(theModel);
	}

}
