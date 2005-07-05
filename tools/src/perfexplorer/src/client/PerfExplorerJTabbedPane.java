package client;

import javax.swing.*;
import java.awt.*;
//import java.awt.event.*;

/**
 Borrowed from http://java.sun.com/docs/books/tutorial/uiswing/components/example-1dot4/TabbedPaneDemo.java
 */
public class PerfExplorerJTabbedPane extends JTabbedPane {

	private static PerfExplorerJTabbedPane thePane = null;

	public static PerfExplorerJTabbedPane getPane () {
		if (thePane == null) {
			thePane = new PerfExplorerJTabbedPane();
			thePane.addChangeListener(new PerfExplorerJTabbedPaneListener());
		}
		return thePane;
	}

	private PerfExplorerJTabbedPane () {
		super();
		ImageIcon icon = null;
		icon = createImageIcon("red-ball.gif");

		JComponent panel1 = AnalysisManagementPane.getPane();
		panel1.setPreferredSize(new Dimension(500, 500));
		this.addTab("Analysis Management", icon, panel1, "Request Cluster Analysis");

		JComponent panel2 = PerformanceExplorerPane.getPane();
		panel2.setPreferredSize(new Dimension(500, 500));
		this.addTab("Cluster Results", icon, panel2, "View Cluster Results");

		JComponent panel3 = PerfExplorerCorrelationPane.getPane();
		panel3.setPreferredSize(new Dimension(500, 500));
		this.addTab("Correlation Results", icon, panel3, "View Correlation Results");
	}

    /** Returns an ImageIcon, or null if the path was invalid. */
    protected static ImageIcon createImageIcon(String path) {
        java.net.URL imgURL = PerfExplorerJTabbedPane.class.getResource(path);
        if (imgURL != null) {
            return new ImageIcon(imgURL);
        } else {
            System.err.println("Couldn't find file: " + path);
            return null;
        }
    }
	
	public void update() {
		fireStateChanged();
	}

}
