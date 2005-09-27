package client;

import server.AnalysisTaskWrapper;
import common.PerfExplorerOutput;
import jargs.gnu.CmdLineParser;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class PerfExplorerClient extends JFrame {
	private static String USAGE = "Usage: PerfExplorerClient [{-h,--help}] {-c,--configfile}=<config_file> [{-s,--standalone}] [{-e,--engine}=<analysis_engine>]\n  where analysis_engine = R or Weka";

	private ActionListener listener = null;
	private JSplitPane splitPane = null;
	private static PerfExplorerClient mainFrame = null;

	public static PerfExplorerClient getMainFrame() {
		return mainFrame;
	}

	private PerfExplorerClient (boolean standalone, String configFile,
	int analysisEngine, boolean quiet) {
		super("PerfExplorer Client");
		PerfExplorerOutput.initialize(quiet);
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

		// window stuff
		int windowWidth = 800;
		int windowHeight = 515;

        //Grab the screen size.
        Toolkit tk = Toolkit.getDefaultToolkit();
        Dimension screenDimension = tk.getScreenSize();
        int screenHeight = screenDimension.height;
        int screenWidth = screenDimension.width;

        Point savedPosition = null; //ParaProf.preferences.getManagerWindowPosition();

        if (savedPosition == null || (savedPosition.x + windowWidth) > screenWidth
                || (savedPosition.y + windowHeight > screenHeight)) {

            //Find the center position with respect to this window.
            int xPosition = (screenWidth - windowWidth) / 2;
            int yPosition = (screenHeight - windowHeight) / 2;

            //Offset a little so that we do not interfere too much with
            //the
            //main window which comes up in the center of the screen.
            if (xPosition > 50)
                xPosition = xPosition - 50;
            if (yPosition > 50)
                yPosition = yPosition - 50;

            this.setLocation(xPosition, yPosition);
        } else {
            this.setLocation(savedPosition);
        }
		mainFrame = this;
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
		CmdLineParser parser = new CmdLineParser();
		CmdLineParser.Option helpOpt = parser.addBooleanOption('h',"help");
		CmdLineParser.Option standaloneOpt = parser.addBooleanOption('s',"standalone");
		CmdLineParser.Option configfileOpt = parser.addStringOption('c',"configfile");
		CmdLineParser.Option engineOpt = parser.addStringOption('e',"engine");
		CmdLineParser.Option quietOpt = parser.addBooleanOption('q',"quiet");

		try {
			parser.parse(args);
		} catch (CmdLineParser.OptionException e) {
			System.err.println(e.getMessage());
			System.err.println(USAGE);
			System.exit(-1);
		}   
		
		Boolean help = (Boolean) parser.getOptionValue(helpOpt);
		Boolean standalone = (Boolean) parser.getOptionValue(standaloneOpt);
		String configFile = (String) parser.getOptionValue(configfileOpt);
		String engine = (String) parser.getOptionValue(engineOpt);
		Boolean quiet = (Boolean) parser.getOptionValue(quietOpt);

		int analysisEngine = AnalysisTaskWrapper.WEKA_ENGINE;

		if (help != null && help.booleanValue()) {
			System.err.println(USAGE);
			System.exit(-1);
		}

		if (quiet == null) 
			quiet = new Boolean(false);

		if (standalone == null) 
			standalone = new Boolean(false);

		if (standalone.booleanValue()) {
			if (configFile == null) {
				System.err.println("Please enter a valid config file.");
				System.err.println(USAGE);
				System.exit(-1);
			}
			if (engine == null) {
				System.err.println("Please enter a valid engine type.");
				System.err.println(USAGE);
				System.exit(-1);
			} else if (engine.equalsIgnoreCase("R")) {
				analysisEngine = AnalysisTaskWrapper.RPROJECT_ENGINE;
			} else if (engine.equalsIgnoreCase("weka")) {
				analysisEngine = AnalysisTaskWrapper.WEKA_ENGINE;
			} else {
				System.err.println(USAGE);
				System.exit(-1);
			}
		}


	/*
		try {
			UIManager.setLookAndFeel(
				UIManager.getCrossPlatformLookAndFeelClassName());
		} catch (Exception e) { }
	*/

		JFrame frame = new PerfExplorerClient(standalone.booleanValue(),
			configFile, analysisEngine, quiet.booleanValue());
		frame.pack();
		frame.setVisible(true);
	}

}
