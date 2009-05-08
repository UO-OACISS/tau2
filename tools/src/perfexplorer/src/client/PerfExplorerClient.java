package edu.uoregon.tau.perfexplorer.client;

import jargs.gnu.CmdLineParser;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.ToolTipManager;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.PythonInterpreterFactory;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.database.DBConnector;
import edu.uoregon.tau.perfdmf.database.PasswordCallback;
import edu.uoregon.tau.perfexplorer.common.Console;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;
import edu.uoregon.tau.perfexplorer.common.ScriptThread;

public class PerfExplorerClient extends JFrame implements ImageExport {
	private static String USAGE = "\nPerfExplorer\n****************************************************************************\nUsage: perfexplorer [OPTIONS]\nwhere [OPTIONS] are:\n[{-h,--help}]  ............................................ print this help.\n[{-g,--configfile}=<config_file>] .. specify one PerfDMF configuration file.\n[{-c,--config}=<config_name>] ........... specify one PerfDMF configuration.\n[{-n,--nogui}] ..................................................... no GUI.\n[{-i,--script}=<script_name>] ................ execute script <script_name>.\n";

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

	public PerfExplorerClient (boolean standalone, String configFile,boolean quiet) {
		super("TAU: PerfExplorer Client");
		
		DBConnector.setPasswordCallback(PasswordCallback.guiPasswordCallback);
		
		PerfExplorerOutput.setQuiet(quiet);
		PerfExplorerConnection.setStandalone(standalone);
		PerfExplorerConnection.setConfigFile(configFile);
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

		int windowWidth = 1190;
		int windowHeight = 620;
		PerfExplorerWindowUtility.centerWindow(this, windowWidth, windowHeight, 0,0, false);
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
		CmdLineParser.Option clientOnlyOpt = parser.addBooleanOption('l',"client");
		// no longer required!
		CmdLineParser.Option configfileOpt = parser.addStringOption('g',"configfile");
		CmdLineParser.Option configOpt = parser.addStringOption('c',"config");
		// assume weka if not specified.
		CmdLineParser.Option quietOpt = parser.addBooleanOption('v',"verbose");
        CmdLineParser.Option tauHomeOpt = parser.addStringOption('t', "tauhome");
        CmdLineParser.Option tauArchOpt = parser.addStringOption('a', "tauarch");
        CmdLineParser.Option noGUIOpt = parser.addBooleanOption('n', "nogui");
        CmdLineParser.Option scriptOpt = parser.addStringOption('i', "script");
        CmdLineParser.Option paramOpt = parser.addStringOption('p', "scriptparams");
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
		Boolean quiet = (Boolean) parser.getOptionValue(quietOpt);
        PerfExplorerClient.tauHome = (String) parser.getOptionValue(tauHomeOpt);
        PerfExplorerClient.tauArch = (String) parser.getOptionValue(tauArchOpt);
        Boolean noGUI = (Boolean) parser.getOptionValue(noGUIOpt);
        String scriptName = (String) parser.getOptionValue(scriptOpt);
        String paramsName = (String) parser.getOptionValue(paramOpt);
        Boolean console = (Boolean) parser.getOptionValue(consoleOpt);
        //Boolean console = new Boolean(true);

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
			console = new Boolean(false);
		}

		if (standalone.booleanValue()) {
			if (configFile == null) {
				if (config != null) {
					String home = System.getProperty("user.home");
					String slash = System.getProperty("file.separator");
					configFile = home + slash + ".ParaProf" + slash + "perfdmf.cfg." + config;
				}
			}
		}

		if (paramsName != null) {
			// create the model a little early, but it's ok.
			PerfExplorerModel model = PerfExplorerModel.getModel();
			// extract the script parameters
			StringTokenizer st = new StringTokenizer(paramsName, ",");
			while(st.hasMoreTokens()) {
				String next = st.nextToken();
				StringTokenizer st2 = new StringTokenizer(next, "=");
				String name = st2.nextToken();
				String value = st2.nextToken();
			    model.addScriptParameter(name, value);
			}
			if (config != null)
			    model.addScriptParameter("config", config);
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
		packages.add("edu.uoregon.tau.perfexplorer.client");
		packages.add("edu.uoregon.tau.perfexplorer.glue");
		packages.add("edu.uoregon.tau.perfexplorer.rules");
		packages.add("edu.uoregon.tau.perfdmf");
		packages.add("edu.uoregon.tau.perfdmf.database");
		edu.uoregon.tau.common.PythonInterpreterFactory.defaultfactory.addPackagesFromList(packages);

		/*
		 * No UI, used for scripting only
		 */
		if (noGUI.booleanValue()) {
			//System.out.println("no gui");
			PerfExplorerNoGUI test = new PerfExplorerNoGUI(
				configFile, quiet.booleanValue());
			if (scriptName != null) {
				//System.out.println("running script, no gui");
				PythonInterpreterFactory.defaultfactory.getPythonInterpreter().execfile(scriptName);
				System.exit(0);
			}
		} else {
			if (console.booleanValue()) {
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
				configFile, quiet.booleanValue());
			frame.pack();
			frame.setVisible(true);
			frame.toFront();

			if (scriptName != null) {
				//System.out.println("running script, gui");
				PerfExplorerActionListener listener = (PerfExplorerActionListener)frame.getListener();
				listener.setScriptName(scriptName);
				//PythonInterpreterFactory.defaultfactory.getPythonInterpreter().execfile(scriptName);
				ScriptThread scripter = new ScriptThread(scriptName);
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
