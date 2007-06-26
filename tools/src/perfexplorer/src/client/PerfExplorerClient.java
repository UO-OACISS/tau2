package client;

import common.EngineType;
import common.PerfExplorerOutput;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.common.ImageExport;
import jargs.gnu.CmdLineParser;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.net.URL;
import edu.uoregon.tau.perfdmf.database.DBConnector;
import edu.uoregon.tau.perfdmf.database.PasswordCallback;

public class PerfExplorerClient extends JFrame implements ImageExport {
	private static String USAGE = "Usage: PerfExplorerClient [{-h,--help}] {-c,--configfile}=<config_file> [{-s,--standalone}] [{-e,--engine}=<analysis_engine>]\n  where analysis_engine = R or Weka";

	private ActionListener listener = null;
	private static PerfExplorerClient mainFrame = null;
	private JComponent mainComponent = null;

	public static PerfExplorerClient getMainFrame() {
		return mainFrame;
	}

	public ActionListener getListener() {
		return listener;
	}

	PerfExplorerClient (boolean standalone, String configFile,
	EngineType analysisEngine, boolean quiet) {
		
		super("TAU: PerfExplorer Client");
		
		DBConnector.setPasswordCallback(PasswordCallback.guiPasswordCallback);
		
		PerfExplorerOutput.setQuiet(quiet);
		PerfExplorerConnection.setStandalone(standalone);
		PerfExplorerConnection.setConfigFile(configFile);
		PerfExplorerConnection.setAnalysisEngine(analysisEngine);
		listener = new PerfExplorerActionListener(this);
		// create a tree
		PerfExplorerJTree tree = PerfExplorerJTree.getTree();
		// Create a scroll pane for the tree
		JScrollPane treeView = new JScrollPane(tree);
		JScrollBar jScrollBar = treeView.getVerticalScrollBar();
		jScrollBar.setUnitIncrement(35);
		treeView.setPreferredSize(new Dimension(300, 400));
		// Create a tabbed pane
		PerfExplorerJTabbedPane tabbedPane = PerfExplorerJTabbedPane.getPane();
		// Create a split pane for the tree view and tabbed pane
		JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
		splitPane.setLeftComponent(treeView);
		splitPane.setRightComponent(tabbedPane);
		mainComponent = splitPane;
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
		int windowWidth = 1000;
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
    	URL url = Utility.getResource("tau32x32.gif");
    	if (url != null)
    		setIconImage(Toolkit.getDefaultToolkit().getImage(url));
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

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        return this.mainComponent.getSize();
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        this.mainComponent.setDoubleBuffered(false);
        this.mainComponent.paintAll(g2D);
        this.mainComponent.setDoubleBuffered(true);
    }

	public static void main (String[] args) {

		// set the tooltip delay to 20 seconds
		ToolTipManager.sharedInstance().setDismissDelay(20000);

		// Process the command line
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

		EngineType analysisEngine = EngineType.WEKA;

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
			try {
				analysisEngine = EngineType.getType(engine);
			} catch (Exception e) {
				System.err.println("Please enter a valid engine type.");
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
