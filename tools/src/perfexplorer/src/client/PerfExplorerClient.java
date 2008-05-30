package client;

import common.EngineType;
import common.Console;
import common.PerfExplorerOutput;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.PythonInterpreterFactory;
import jargs.gnu.CmdLineParser;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.net.URL;
import java.io.IOException;
import java.io.OutputStream;
import edu.uoregon.tau.perfdmf.database.DBConnector;
import edu.uoregon.tau.perfdmf.database.PasswordCallback;
import java.util.List;
import java.util.ArrayList;

public class PerfExplorerClient extends JFrame implements ImageExport {
	private static String USAGE = "\nPerfExplorer\n****************************************************************************\nUsage: perfexplorer [OPTIONS]\nwhere [OPTIONS] are:\n[{-h,--help}]  ............................................ print this help.\n[{-g,--configfile}=<config_file>] .. specify one PerfDMF configuration file.\n[{-c,--config}=<config_name>] ........... specify one PerfDMF configuration.\n[{-e,--engine}=<analysis_engine>] ......  where analysis_engine = R or Weka.\n[{-n,--nogui}] ..................................................... no GUI.\n[{-i,--script}=<script_name>] ................ execute script <script_name>.\n";

	private ActionListener listener = null;
	private static PerfExplorerClient mainFrame = null;
	private JComponent mainComponent = null;

	private static String tauHome;
	private static String tauArch;

	public static PerfExplorerClient getMainFrame() {
		return mainFrame;
	}

	public ActionListener getListener() {
		return listener;
	}

	public PerfExplorerClient (boolean standalone, String configFile,
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
		treeView.setPreferredSize(new Dimension(300, 650));
		// Create a tabbed pane
		PerfExplorerJTabbedPane tabbedPane = PerfExplorerJTabbedPane.getPane();
		tabbedPane.setPreferredSize(new Dimension(890, 650));
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
		//this.setPreferredSize(new Dimension(1190, 700));    // only works in java 5+
		int windowWidth = 1190;
		int windowHeight = 620;

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

	public void refreshDatabases() {
		System.out.println("update the database tree!");
		PerfExplorerJTree.refreshDatabases();
	}

	public static final OutputStream DEV_NULL = new OutputStream() {
		public void write(int b) { return; }
	};

	public static void main (String[] args) {
		// DO THIS FIRST!
		// -Dderby.stream.error.field=client.PerfExplorerClient.DEV_NULL
		// doesn't work with JNLP, so do it here!
		System.setProperty("derby.stream.error.field", "client.PerfExplorerClient.DEV_NULL");
		
		// set the tooltip delay to 20 seconds
		ToolTipManager.sharedInstance().setDismissDelay(20000);

		// Process the command line
		CmdLineParser parser = new CmdLineParser();
		CmdLineParser.Option helpOpt = parser.addBooleanOption('h',"help");
		// this is the new default... have to specify client if you want it
		CmdLineParser.Option standaloneOpt = parser.addBooleanOption('s',"standalone");
		CmdLineParser.Option clientOnlyOpt = parser.addBooleanOption('l',"clientonly");
		// no longer required!
		CmdLineParser.Option configfileOpt = parser.addStringOption('g',"configfile");
		CmdLineParser.Option configOpt = parser.addStringOption('c',"config");
		// assume weka if not specified.
		CmdLineParser.Option engineOpt = parser.addStringOption('e',"engine");
		CmdLineParser.Option quietOpt = parser.addBooleanOption('v',"verbose");
        CmdLineParser.Option tauHomeOpt = parser.addStringOption('t', "tauhome");
        CmdLineParser.Option tauArchOpt = parser.addStringOption('a', "tauarch");
        CmdLineParser.Option noGUIOpt = parser.addBooleanOption('n', "nogui");
        CmdLineParser.Option scriptOpt = parser.addStringOption('i', "script");
        CmdLineParser.Option consoleOpt = parser.addBooleanOption('w', "consoleWindow");
        
		try {
			parser.parse(args);
		} catch (CmdLineParser.OptionException e) {
			System.err.println(e.getMessage());
			System.err.println(USAGE);
			System.exit(-1);
		}   
		
		Boolean help = (Boolean) parser.getOptionValue(helpOpt);
		Boolean standalone = (Boolean) parser.getOptionValue(standaloneOpt);
		Boolean clientOnly = (Boolean) parser.getOptionValue(clientOnlyOpt);
		String configFile = (String) parser.getOptionValue(configfileOpt);
		String config = (String) parser.getOptionValue(configOpt);
		String engine = (String) parser.getOptionValue(engineOpt);
		Boolean quiet = (Boolean) parser.getOptionValue(quietOpt);
        PerfExplorerClient.tauHome = (String) parser.getOptionValue(tauHomeOpt);
        PerfExplorerClient.tauArch = (String) parser.getOptionValue(tauArchOpt);
        Boolean noGUI = (Boolean) parser.getOptionValue(noGUIOpt);
        String scriptName = (String) parser.getOptionValue(scriptOpt);
        Boolean console = (Boolean) parser.getOptionValue(consoleOpt);
        //Boolean console = new Boolean(true);

		EngineType analysisEngine = EngineType.WEKA;

		if (help != null && help.booleanValue()) {
			System.err.println(USAGE);
			System.exit(-1);
		}

		if (quiet == null) 
			quiet = new Boolean(true);

		// standalone is the new default!
		if (standalone == null)  {
			standalone = new Boolean(true);
		}
		if (clientOnly != null && clientOnly.booleanValue()) 
			standalone = new Boolean(false);

		if (noGUI == null)  {
			noGUI = new Boolean(false);
		}

		if (console == null)  {
			console = new Boolean(true);
		}

		if (standalone.booleanValue()) {
			if (configFile == null) {
				if (config != null) {
					String home = System.getProperty("user.home");
					String slash = System.getProperty("file.separator");
					configFile = home + slash + ".ParaProf" + slash + "perfdmf.cfg." + config;
				}
			}
			try {
				analysisEngine = EngineType.getType(engine);
			} catch (Exception e) {
				analysisEngine = EngineType.WEKA;
			}
		}

/*		try {
			UIManager.setLookAndFeel(
				UIManager.getCrossPlatformLookAndFeelClassName());
		} catch (Exception e) { }
*/
		// make sure the Jython interpreter knows about our packages
		// because if they aren't in the classpath, it can't find them.
		// this is necessary when running from JNLP.

		List<String> packages = new ArrayList<String>();
		packages.add("client");
		packages.add("glue");
		packages.add("rules");
		packages.add("edu.uoregon.tau.perfdmf");
		packages.add("edu.uoregon.tau.perfdmf.database");
		edu.uoregon.tau.common.PythonInterpreterFactory.defaultfactory.addPackagesFromList(packages);

		if (noGUI.booleanValue()) {
			//System.out.println("no gui");
			PerfExplorerNoGUI test = new PerfExplorerNoGUI(
				configFile, analysisEngine, quiet.booleanValue());
			if (scriptName != null) {
				//System.out.println("running script, no gui");
				PythonInterpreterFactory.defaultfactory.getPythonInterpreter().execfile(scriptName);
				System.exit(0);
			}
		} else {
			if (!console.booleanValue()) {
				// send all output to a console window
				try {
					new Console();
				} catch (IOException e) {
					System.err.println("An error occurred setting up the console:");
					System.err.println(e.getMessage());
					e.printStackTrace();
					System.exit(-1);
				} 
			} 
			//System.out.println("gui");
			PerfExplorerClient frame = new PerfExplorerClient(standalone.booleanValue(),
				configFile, analysisEngine, quiet.booleanValue());
			frame.pack();
			frame.setVisible(true);
			frame.toFront();

			if (scriptName != null) {
				//System.out.println("running script, gui");
				PythonInterpreterFactory.defaultfactory.getPythonInterpreter().execfile(scriptName);
				PerfExplorerActionListener listener = (PerfExplorerActionListener)frame.getListener();
				listener.setScriptName(scriptName);
			}
		}
	}

	public static String getTauHome() {
		return tauHome;
	}

	public static void setTauHome(String tauHome) {
		PerfExplorerClient.tauHome = tauHome;
	}

	public static String getTauArch() {
		return tauArch;
	}

	public static void setTauArch(String tauArch) {
		PerfExplorerClient.tauArch = tauArch;
	}

}
