package edu.uoregon.tau.perfexplorer.client;

import javax.swing.*;

import java.awt.*;
//import java.awt.event.*;

/**
 Borrowed from http://java.sun.com/docs/books/tutorial/uiswing/components/example-1dot4/TabbedPaneDemo.java
 */
public class PerfExplorerJTabbedPane extends JTabbedPane {

	private static PerfExplorerJTabbedPane thePane = null;
	private JComponent panel1 = null;
	private JComponent panel2 = null;
	private JComponent panel3 = null;
	private JComponent panel4 = null;

	public static PerfExplorerJTabbedPane getPane () {
		if (thePane == null) {
			thePane = new PerfExplorerJTabbedPane();
			thePane.addChangeListener(new PerfExplorerJTabbedPaneListener());
		}
		return thePane;
	}
	
	public JComponent getTab(int index) {
		JComponent comp = null;
		switch(index) {
			case (0): {
				comp = this.panel1;
				break;
			}
			case (1): {
				comp = this.panel2;
				break;
			}
			case (2): {
				comp = this.panel3;
				break;
			}
			case (3): {
				comp = this.panel4;
				break;
			}
		}
		return comp;
	}
	
	private PerfExplorerJTabbedPane () {
		super();
		ImageIcon icon = null;
		icon = createImageIcon("red-ball.gif");

		// Create a split pane for the tree view and tabbed pane
		JSplitPane splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
		JScrollPane treeView = new JScrollPane(AnalysisManagementPane.getPane());
		JScrollBar jScrollBar = treeView.getVerticalScrollBar();
		jScrollBar.setUnitIncrement(35);

		splitPane.setTopComponent(treeView);
		splitPane.setBottomComponent(null);
		splitPane.setDividerLocation(200);
		this.panel1 = splitPane;
		panel1.setPreferredSize(new Dimension(600, 500));
		this.addTab("Analysis Management", icon, panel1, "Request Cluster Analysis");

		this.panel2 = PerformanceExplorerPane.getPane();
		panel2.setPreferredSize(new Dimension(600, 500));
		this.addTab("Cluster Results", icon, panel2, "View Cluster Results");

		this.panel3 = PerfExplorerCorrelationPane.getPane();
		panel3.setPreferredSize(new Dimension(600, 500));
		this.addTab("Correlation Results", icon, panel3, "View Correlation Results");

		this.panel4 = ChartPane.getPane();
		panel4.setPreferredSize(new Dimension(600, 500));
		this.addTab("Custom Charts", icon, panel4, "Custom Performance Charts");
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
