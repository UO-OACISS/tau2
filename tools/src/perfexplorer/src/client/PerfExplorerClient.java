package client;

import javax.swing.*;

import server.AnalysisTaskWrapper;

import java.awt.*;
import java.awt.event.*;

public class PerfExplorerClient extends JFrame {

	private ActionListener listener = null;

	private JSplitPane splitPane = null;

	private static PerfExplorerClient mainFrame = null;

	public static PerfExplorerClient getMainFrame() {
		return mainFrame;
	}

	private PerfExplorerClient (String hostname, boolean standalone, String configFile, int analysisEngine) {
		super("PerfExplorer Client");
		PerfExplorerConnection.setHostname(hostname);
		PerfExplorerConnection.setStandalone(standalone);
		PerfExplorerConnection.setConfigFile(configFile);
		PerfExplorerConnection.setAnalysisEngine(analysisEngine);
		listener = new PerfExplorerActionListener(this);
		// create a tree
        PerfExplorerJTree tree = PerfExplorerJTree.getTree();
        // Create a scroll pane for the tree
        JScrollPane treeView = new JScrollPane(tree);
        treeView.setPreferredSize(new Dimension(300, 400));
		// Create a tabbed pane
		PerfExplorerJTabbedPane tabbedPane = PerfExplorerJTabbedPane.getPane();
		// Create a split pane for the tree view and tabbed pane
		JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        splitPane.setLeftComponent(treeView);
        splitPane.setRightComponent(tabbedPane);
		// add the split pane to the main frame
		getContentPane().add(splitPane, BorderLayout.CENTER);
		setJMenuBar(new PerfExplorerMainJMenuBar(listener));

		// exit when the user closes the main window.
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				System.exit(0);
			}
		});
	}

	public void actionPerformed (ActionEvent event) {
		try {
			Object EventSrc = event.getSource();
			if(EventSrc instanceof JMenuItem) {
				String arg = event.getActionCommand();
				if(arg.equals("Quit")) {
					System.exit(0);
				}
			}
		} catch (Exception e) {
			System.err.println("actionPerformed Exception: " + e.getMessage());
			e.printStackTrace();
		} 
	}

	public static void main (String[] args) {
	/*
		try {
			UIManager.setLookAndFeel(
				UIManager.getCrossPlatformLookAndFeelClassName());
		} catch (Exception e) { }
		*/

		String usage = "Usage: PerfExplorerClient [--standalone config_file analysis_engine]\n  where analysis_engine = R or Weka";
		
		boolean standalone = false;
		String configFile = null;
		int analysisEngine = AnalysisTaskWrapper.WEKA_ENGINE;
		if (args.length > 1) {
			if (args[1].equalsIgnoreCase("--standalone"))
				standalone = true;
			configFile = args[2];
			if (args[3].equalsIgnoreCase("R")) {
				analysisEngine = AnalysisTaskWrapper.RPROJECT_ENGINE;
			} else if (args[3].equalsIgnoreCase("weka")) {
				analysisEngine = AnalysisTaskWrapper.WEKA_ENGINE;
			} else {
				System.out.println(usage);
				System.exit(0);
			}

		}
		JFrame frame = new PerfExplorerClient(args[0], standalone, configFile, analysisEngine);
		frame.pack();
		frame.setVisible(true);
	}

}
