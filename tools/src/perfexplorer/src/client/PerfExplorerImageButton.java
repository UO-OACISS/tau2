package edu.uoregon.tau.perfexplorer.client;

import javax.swing.Icon;
import javax.swing.JButton;

public class PerfExplorerImageButton extends JButton {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3741062834668426286L;
	int id = 0;
	String description = null;

	public PerfExplorerImageButton (Icon icon, int id, String description) {
		super(icon);
		this.id = id;
		this.description = description;
		this.setToolTipText(description);
		this.setActionCommand("" + id);
	}

}
