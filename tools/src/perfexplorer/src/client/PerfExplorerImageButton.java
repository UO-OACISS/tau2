package edu.uoregon.tau.perfexplorer.client;

import javax.swing.JButton;
import javax.swing.Icon;

public class PerfExplorerImageButton extends JButton {

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
