package edu.uoregon.tau.paraprof;

import jargs.gnu.CmdLineParser;

import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import javax.swing.ToolTipManager;
import javax.swing.UIManager;

import edu.uoregon.tau.common.TauScripter;
import edu.uoregon.tau.paraprof.interfaces.EclipseHandler;
import edu.uoregon.tau.paraprof.script.ParaProfScript;
import edu.uoregon.tau.paraprof.sourceview.SourceManager;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DataSourceExport;
import edu.uoregon.tau.perfdmf.FileList;
import edu.uoregon.tau.perfdmf.Node;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * ParaProf This is the 'main' for paraprof
 * 
 * <P>
 * CVS $Id: ParaProf.java,v 1.90 2009/12/12 01:47:40 amorris Exp $
 * </P>
 * 
 * @author Robert Bell, Alan Morris
 * @version $Revision: 1.90 $
 */
public class ParaProf implements ActionListener {

    // This class handles uncaught throwables on the AWT-EventQueue thread
    static public class XThrowableHandler {

	public XThrowableHandler() {}

	public void handle(Throwable t) throws Throwable {
	    if (t instanceof Exception) {
		ParaProfUtils.handleException((Exception) t);
	    } else {
		System.err.println("Uncaught Throwable: " + t.fillInStackTrace());
	    }
	}
    }

    private final static String VERSION = "Wed Jun  4 02:02:32 PM PDT 2025";

    public static int defaultNumberPrecision = 6;

    public static File paraProfHomeDirectory;
    public static Preferences preferences = new Preferences();
    public static ColorChooser colorChooser;
    public static ColorMap colorMap = new ColorMap();

    public static ParaProfManagerWindow paraProfManagerWindow;
    public static ApplicationManager applicationManager = new ApplicationManager();
    private static HelpWindow helpWindow;
    public static PreferencesWindow preferencesWindow;
    public static Runtime runtime;
    private static int numWindowsOpen = 0;

    private static int fileType = DataSource.TAUPROFILE;
    //If this is set to true then the user has specified snapshot format and we may have a series of snapshots from one trial.
    private static boolean seriesSnap = false;
    private static File sourceFiles[] = new File[0];
    private static boolean fixNames = false;
    private static boolean monitorProfiles;
    private static String configFile;

    public static boolean demoMode;
    public static boolean controlMode;
    public static boolean usePathNameInTrial = false;
    public static FunctionBarChartWindow theComparisonWindow;
    public static boolean JNLP = false;
    public static List<ParaProfScript> scripts = new ArrayList<ParaProfScript>();
    public static String scriptFile;

    public static boolean insideEclipse;
    public static EclipseHandler eclipseHandler;
    public static SourceManager directoryManager;
    public static String jarLocation;
    public static String schemaLocation;
    private static String range;

    // static initializer block
    static {
	ParaProf.runtime = Runtime.getRuntime();
    }

    public static HelpWindow getHelpWindow() {
	if (helpWindow == null) {
	    helpWindow = new HelpWindow();
	}
	return helpWindow;
    }

    public static SourceManager getDirectoryManager() {
	if (directoryManager == null) {
	    directoryManager = new SourceManager(ParaProf.preferences.getSourceLocations());
	}
	return directoryManager;
    }

    public static void registerScript(ParaProfScript pps) {
	scripts.add(pps);
    }

    public ParaProf() {}

    private static void usage() {
	System.err.println("Usage: paraprof [--pack <file>] [--dump] [--dumprank <rank>] [-p] [-m] [-i] [-f <filetype>] <files/directory>\n\n"
		+ "try `paraprof --help` for more information");
    }

    private static void outputHelp() {
	System.err.println("Usage: paraprof [options] <files/directory> \n\n" + "Options:\n\n"
		+ "  -f, --filetype <filetype>       Specify type of performance data, options are:\n"
		+ "                                    profiles (default), pprof, dynaprof, mpip,\n"
		+ "                                    gprof, psrun, hpm, packed, cube, hpc, ompp\n"
		+ "                                    snap, perixml, gptl, ipm, google, darshan\n"
		+ "  --range a-b:c                   Load only profiles from the given range(s) of processes\n"
		+ "                                    Seperate individual ids or dash-defined ranges with colons\n"
		+ "  -h, --help                      Display this help message\n" + "\n"
		+ "The following options will run only from the console (no GUI will launch):\n" + "\n"
		+ "  --merge <file.gz>               Merges snapshot profiles\n"
		+ "  --pack <file>                   Pack the data into packed (.ppk) format\n"
		+ "  --text <file>                   Dump the data into text (.csv) format\n"
		+ "  --dump                          Dump profile data to TAU profile format\n"
		+ "  --dumprank <rank>               Dump profile data for <rank> to TAU profile format\n"
		+ "                                  Specify multiple ranks in with a comma separated list. Use\n"
		+ "                                  a dash between values for a range. Do not use spaces. E.g. 0,4,8-10,16\n"
		+ "  -v, --dumpsummary               Dump derived statistical data to TAU profile format\n"
		+ "  --overwrite                     Allow overwriting of profiles\n"
		+ "  -o, --oss                       Print profile data in OSS style text output\n"
		+ "  -q, --dumpmpisummary            Print high level time and communication summary\n"
		+ "  -d, --metadump                  Print profile metadata (works with --dumpmpisummary)\n"
		+ "  -x, --suppressmetrics           Exclude child calls and exclusive time from --dumpmpisummary\n"		
		+ "  -s, --summary                   Print only summary statistics\n"
		+ "                                    (only applies to OSS output)\n" 
		+ "  -y, --control                   Enter command line control mode for exploring/loading/exporting database trials\n"
		+ "  --writecomm <file.csv>          Write communication matrix data (if any) to the specified csv file.\n"
		+ "\n" + "Notes:\n"
		+ "  For the TAU profiles type, you can specify either a specific set of profile\n"
		+ "files on the commandline, or you can specify a directory (by default the current\n"
		+ "directory).  The specified directory will be searched for profile.*.*.* files,\n"
		+ "or, in the case of multiple counters, directories named MULTI_* containing\n" + "profile data.\n\n");
    }

    public static void incrementNumWindows() {
	numWindowsOpen++;
	//System.out.println ("incrementing: now " + numWindowsOpen);
    }

    public static void decrementNumWindows() {
	numWindowsOpen--;
	//System.out.println ("decrementing: now " + numWindowsOpen);
	if (numWindowsOpen <= 0) {
	    exitParaProf(0);
	}
    }

    public static void loadDefaultTrial(){
	loadDefaultTrial(null);
    }

    public static void loadDefaultTrial(String range) {

	// Create a default application.
	ParaProfApplication app = ParaProf.applicationManager.addApplication();
	app.setName("Default App");

	// Create a default experiment.
	ParaProfExperiment experiment = app.addExperiment();
	experiment.setName("Default Exp");

	ParaProf.paraProfManagerWindow.setVisible(true);
	try {
	    if (fileType == DataSource.PPK) {
		for (int i = 0; i < sourceFiles.length; i++) {
		    File files[] = new File[1];
		    files[0] = sourceFiles[i];
		    paraProfManagerWindow.addTrial(app, experiment, files, fileType, fixNames, monitorProfiles);
		}
	    }else if(sourceFiles.length==0&&fileType == DataSource.GOOGLE){
		//Look in the current directory of *.txt files.
		String currentdir = System.getProperty("user.dir");	
		File dir = new File(currentdir);
		String[] files = dir.list();
		ArrayList<File> filelist = new ArrayList<File>();
		for(String file: files){
		    if(file.endsWith(".txt")){
			filelist.add(new File(file));
		    }
		}
		paraProfManagerWindow.addTrial(experiment, (File[]) filelist.toArray(sourceFiles), fileType, fixNames, monitorProfiles, range);
	    }
	    //If it's not a series but it is a snapshot then open the files successively.
	    else if(!seriesSnap&&fileType==DataSource.SNAP){
	    	for (int i = 0; i < sourceFiles.length; i++) {
			    File files[] = new File[1];
			    files[0] = sourceFiles[i];
			    paraProfManagerWindow.addTrial(app, experiment, files, fileType, fixNames, monitorProfiles);
			}
	    }
	    else {
		paraProfManagerWindow.addTrial(experiment, sourceFiles, fileType, fixNames, monitorProfiles, range);
	    }
	} catch (java.security.AccessControlException ace) {
	    // running as Java Web Start without permission
	}
    }

    public static void initialize() {

	try {
	    if (System.getProperty("jnlp.running") != null) {
		ParaProf.JNLP = true;
	    }
	} catch (java.security.AccessControlException ace) {
	    // if we get this security exception, we are definitely in JNLP
	    ParaProf.JNLP = true;
	}

	if (ParaProf.JNLP == false) {

	    // Establish the presence of a .ParaProf directory. This is located
	    // by default in the user's home directory.
	    ParaProf.paraProfHomeDirectory = new File(System.getProperty("user.home") + "/.ParaProf");
	    if (paraProfHomeDirectory.exists()) {

		// try to load preferences
		try {
		    FileInputStream savedPreferenceFIS = new FileInputStream(ParaProf.paraProfHomeDirectory.getPath()
			    + "/ParaProf.conf");

		    ObjectInputStream inSavedPreferencesOIS = new ObjectInputStream(savedPreferenceFIS);
		    ParaProf.preferences = (Preferences) inSavedPreferencesOIS.readObject();
		    colorChooser = new ColorChooser(ParaProf.preferences);
		} catch (Exception e) {
		    if (e instanceof FileNotFoundException) {
			System.out.println("No preference file present, using defaults!");
		    } else {
			System.out.println("Error while trying to read the ParaProf preferences file, using defaults");
			//System.out.println("Please delete this file, or replace it with a valid one!");
			//System.out.println("Note: Deleting the file will cause ParaProf to restore the default preferences");
		    }
		}

		ParaProf.colorMap.setMap(preferences.getAssignedColors());
		ParaProf.preferences.setDatabasePassword(null);

		// try to load perfdmf.cfg.
		File perfDMFcfg = new File(ParaProf.paraProfHomeDirectory.getPath() + "/perfdmf.cfg");
		if (perfDMFcfg.exists()) {
		    ParaProf.preferences.setDatabaseConfigurationFile(ParaProf.paraProfHomeDirectory.getPath() + "/perfdmf.cfg");
		}

	    } else {
		System.out.println("Did not find ParaProf home directory... creating...");
		paraProfHomeDirectory.mkdir();
		System.out.println("Done creating ParaProf home directory!");
	    }
	} else { // ParaProf.JNLP == true
	    // Java Web Start
	    //URL url = ParaProf.class.getResource("/perfdmf.cfg");
	    //throw new ParaProfException("URL = " + url);

	    //            URL url = ParaProf.class.getResource("/perfdmf.cfg");
	    //            String path = URLDecoder.decode(url.getPath());
	    //            ParaProf.preferences.setDatabaseConfigurationFile(path);
	    preferences = new Preferences();
	}
	
	setUIFont (new javax.swing.plaf.FontUIResource(new Font(preferences.getFontName(),preferences.getFontStyle(), preferences.getFontSize())));

	if (colorChooser == null) {
	    // we create one if ParaProf.conf wasn't properly read
	    ParaProf.colorChooser = new ColorChooser(null);
	}

	ParaProf.preferencesWindow = new PreferencesWindow(preferences);

	DataSource.setMeanIncludeNulls(!preferences.getComputeMeanWithoutNulls());

	// Set the default exception handler for AWT
	// This avoids the mess of having to put a try/catch around every AWT entry point
	try {
	    System.setProperty("sun.awt.exception.handler", XThrowableHandler.class.getName());
	} catch (java.security.AccessControlException ace) {
	    // running as Java Web Start without permission
	}

	// Initialize, but do not show the manager window
	//System.out.println("creating Manager window with: " + configFile);
	ParaProf.paraProfManagerWindow = new ParaProfManagerWindow(configFile);
    }

    public static void loadScripts() {
	if (ParaProf.JNLP == false) {
	    ParaProf.scripts.clear();
	    ParaProf.scriptFile = System.getProperty("user.home") + "/.ParaProf/ParaProf.py";
	    if (new File(scriptFile).exists()) {
		try {
		    TauScripter.execfile(System.getProperty("user.home") + "/.ParaProf/ParaProf.py");
		} catch (Exception e) {
		    new ParaProfErrorDialog("Exception while executing script: ", e);
		}
	    }
	}
    }

    public void actionPerformed(ActionEvent evt) {
	Object EventSrc = evt.getSource();
	if (EventSrc instanceof javax.swing.Timer) {
	    System.out.println("------------------------");
	    System.out.println("The amount of memory used by the system is: " + runtime.totalMemory());
	    System.out.println("The amount of memory free to the system is: " + runtime.freeMemory());
	}
    }

    public static String getInfoString() {
	//long memUsage = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024;

	DecimalFormat f = new DecimalFormat("#.## MB");

	String memUsage = "Free: " + f.format(java.lang.Runtime.getRuntime().freeMemory() / 1000000.0) + "\nTotal: "
	+ f.format(java.lang.Runtime.getRuntime().totalMemory() / 1000000.0) + "\nMax: "
	+ f.format(java.lang.Runtime.getRuntime().maxMemory() / 1000000.0);

	String message = "ParaProf\n" + getVersionString() + "\nJVM Memory stats:\n" + memUsage + "\n"
	+ "http://www.cs.uoregon.edu/research/tau\n";

	return message;
    }

    public static String getVersionString() {
	return new String(VERSION);
    }

    public static void loadPreferences(File file) throws FileNotFoundException, IOException, ClassNotFoundException {

	FileInputStream savedPreferenceFIS = new FileInputStream(file);

	//If here, means that no exception was thrown, and there is a preference file present.
	//Create ObjectInputStream and try to read it in.
	ObjectInputStream inSavedPreferencesOIS = new ObjectInputStream(savedPreferenceFIS);
	ParaProf.preferences = (Preferences) inSavedPreferencesOIS.readObject();
	colorChooser = new ColorChooser(ParaProf.preferences);

	ParaProf.colorMap.setMap(ParaProf.preferences.getAssignedColors());

	List<ParaProfTrial> trials = ParaProf.paraProfManagerWindow.getLoadedTrials();
	for (Iterator<ParaProfTrial> it = trials.iterator(); it.hasNext();) {
	    ParaProfTrial ppTrial = it.next();
	    ParaProf.colorChooser.setColors(ppTrial, -1);
	    ppTrial.updateRegisteredObjects("colorEvent");
	    ppTrial.updateRegisteredObjects("prefEvent");
	}

    }

    // This method is reponsible for any cleanup required in ParaProf 
    // before an exit takes place.
    public static void exitParaProf(int exitValue) {
	try {
	    savePreferences(new File(ParaProf.paraProfHomeDirectory.getPath() + "/ParaProf.conf"));
	} catch (Exception e) {
	    System.err.println("An error occured while trying to save ParaProf preferences.");
	    e.printStackTrace();
	}
	if (!insideEclipse && !controlMode) {
	    // never call System.exit when invoked by the eclipse plugin, it will close the whole JVM, including the user's eclipse!
	    // also, don't call when in control mode, which is the new mechanism used by eclipse
	    System.exit(exitValue);
	}
    }

    public static boolean savePreferences(File file) {

	ParaProf.colorChooser.setSavedColors();
	ParaProf.preferences.setAssignedColors(ParaProf.colorMap.getMap());
	//ParaProf.preferences.setManagerWindowPosition(ParaProf.paraProfManagerWindow.getLocation());
	ParaProf.preferences.setSourceLocations(getDirectoryManager().getCurrentElements());
	try {
	    ObjectOutputStream prefsOut = new ObjectOutputStream(new FileOutputStream(file));
	    prefsOut.writeObject(ParaProf.preferences);
	    prefsOut.close();
	} catch (Exception e) {
	    System.err.println("An error occured while trying to save ParaProf preferences.");
	    //e.printStackTrace();
	    return false;
	}
	return true;
    }

    // Main entry point
    static public void main(String[] args) {

	// Set the tooltip delay to 20 seconds
	ToolTipManager.sharedInstance().setDismissDelay(20000);

	// Process command line arguments
	CmdLineParser parser = new CmdLineParser();

	CmdLineParser.Option mergeOpt = parser.addStringOption('a', "merge");
	CmdLineParser.Option packOpt = parser.addStringOption('a', "pack");
	CmdLineParser.Option textOpt = parser.addStringOption('t', "text");
	CmdLineParser.Option schemaLocationOpt = parser.addStringOption('c', "schemadir");
	CmdLineParser.Option metadumpOpt = parser.addBooleanOption('d', "metadump");
	CmdLineParser.Option typeOpt = parser.addStringOption('f', "filetype");
	CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
	CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
	CmdLineParser.Option fixOpt = parser.addBooleanOption('i', "fixnames");
	CmdLineParser.Option jarLocationOpt = parser.addStringOption('j', "jardir");
	CmdLineParser.Option monitorOpt = parser.addBooleanOption('m', "monitor");
	CmdLineParser.Option rangeOpt = parser.addStringOption('n', "range");
	CmdLineParser.Option ossOpt = parser.addBooleanOption('o', "oss");
	CmdLineParser.Option unpackMPISummOpt = parser.addBooleanOption('q', "dumpmpisummary");
	CmdLineParser.Option unpackRankOpt = parser.addStringOption('r', "dumprank");
	CmdLineParser.Option summaryOpt = parser.addBooleanOption('s', "summary");
	CmdLineParser.Option unpackOpt = parser.addBooleanOption('u', "dump");
	CmdLineParser.Option unpackSummOpt = parser.addBooleanOption('v', "dumpsummary");
	CmdLineParser.Option overwriteOpt = parser.addBooleanOption('w', "overwrite");
	CmdLineParser.Option suppressOpt = parser.addBooleanOption('x', "suppressmetrics");
	CmdLineParser.Option controlOpt = parser.addBooleanOption('y', "control");
	CmdLineParser.Option demoOpt = parser.addBooleanOption('z', "demo");
	CmdLineParser.Option writeCommOpt = parser.addStringOption('a', "writecomm");
	
	

	try {
	    parser.parse(args);
	} catch (CmdLineParser.OptionException e) {
	    System.err.println("paraprof: " + e.getMessage());
	    ParaProf.usage();
	    System.exit(-1);
	}

	configFile = (String) parser.getOptionValue(configfileOpt);
	Boolean help = (Boolean) parser.getOptionValue(helpOpt);
	String fileTypeString = (String) parser.getOptionValue(typeOpt);
	Boolean fixNames = (Boolean) parser.getOptionValue(fixOpt);
	String merge = (String) parser.getOptionValue(mergeOpt);
	String pack = (String) parser.getOptionValue(packOpt);
	String text = (String) parser.getOptionValue(textOpt);
	String writeComm = (String) parser.getOptionValue(writeCommOpt);
	Boolean unpack = (Boolean) parser.getOptionValue(unpackOpt);
	Boolean unpackSumm = (Boolean) parser.getOptionValue(unpackSummOpt);
	
	Boolean unpackMPISumm = (Boolean) parser.getOptionValue(unpackMPISummOpt);
	Boolean suppress = (Boolean) parser.getOptionValue(suppressOpt);
	Boolean metadump = (Boolean) parser.getOptionValue(metadumpOpt);
	
	Boolean overwrite = (Boolean) parser.getOptionValue(overwriteOpt);
	String unpackrank = (String) parser.getOptionValue(unpackRankOpt);
	if (unpackrank != null||unpackSumm!=null||unpackMPISumm!=null||metadump!=null) {
	    unpack = Boolean.TRUE;
	}
	Boolean oss = (Boolean) parser.getOptionValue(ossOpt);
	Boolean summary = (Boolean) parser.getOptionValue(summaryOpt);
	Boolean monitor = (Boolean) parser.getOptionValue(monitorOpt);
	Boolean demo = (Boolean) parser.getOptionValue(demoOpt);
	Boolean control = (Boolean) parser.getOptionValue(controlOpt);
	ParaProf.jarLocation = (String) parser.getOptionValue(jarLocationOpt);
	ParaProf.schemaLocation = (String) parser.getOptionValue(schemaLocationOpt);
	range = (String) parser.getOptionValue(rangeOpt);//TODO: Implement range value for profile input
	if(range!=null&&fileTypeString==null){
	    fileTypeString="profiles";
	}

	controlMode = control != null && control.booleanValue();
	demoMode = demo != null && demo.booleanValue();

	if (configFile != "") {
	    //System.out.println("commandline db config: " + configFile);
	    ParaProf.preferences.setDatabaseConfigurationFile(configFile);
	}

	if (monitor != null) {
	    monitorProfiles = monitor.booleanValue();
	}

	if (pack != null && unpack != null) {
	    System.err.println("--pack and --dump are mutually exclusive");
	    System.exit(-1);
	}

	if (help != null && help.booleanValue()) {
	    ParaProf.outputHelp();
	    System.exit(-1);
	}

	String sourceFilenames[] = parser.getRemainingArgs();
	sourceFiles = new File[sourceFilenames.length];
	for (int i = 0; i < sourceFilenames.length; i++) {
	    sourceFiles[i] = new File(sourceFilenames[i]);
//	    if(!sourceFiles[i].exists()){
//	    	sourceFiles= new File[0];
//	    	fileTypeString="profiles";
//	    	break;
//	    }
	}
	
	if(sourceFiles!=null&&sourceFiles.length>0){
		if(sourceFiles[0]!=null&&sourceFiles[0].exists()){
			if(sourceFiles[0].isDirectory()){
				try {
					LoadTrialWindow.lastDirectory=sourceFiles[0].getCanonicalPath();
				} catch (IOException e) {
				}
			}
			else{
				try {
					File pf=sourceFiles[0].getParentFile();
					if(pf!=null&&pf.exists())
						LoadTrialWindow.lastDirectory=pf.getCanonicalPath();
				} catch (IOException e) {
				}
			}
		}
	}

	if (fixNames != null)
	    ParaProf.fixNames = fixNames.booleanValue();

	if (fileTypeString != null) {
	    if (fileTypeString.equals("profiles")) {
		ParaProf.fileType = DataSource.TAUPROFILE;
	    } else if (fileTypeString.equals("pprof")) {
		ParaProf.fileType = DataSource.PPROF;
	    } else if (fileTypeString.equals("dynaprof")) {
		ParaProf.fileType = DataSource.DYNAPROF;
	    } else if (fileTypeString.equals("mpip")) {
		ParaProf.fileType = DataSource.MPIP;
	    } else if (fileTypeString.equals("hpm")) {
		ParaProf.fileType = DataSource.HPM;
	    } else if (fileTypeString.equals("gprof")) {
		ParaProf.fileType = DataSource.GPROF;
	    } else if (fileTypeString.equals("psrun")) {
		ParaProf.fileType = DataSource.PSRUN;
	    } else if (fileTypeString.equals("packed")) {
		ParaProf.fileType = DataSource.PPK;
	    } else if (fileTypeString.equals("cube")) {
		ParaProf.fileType = DataSource.CUBE;
	    } else if (fileTypeString.equals("hpc")) {
		ParaProf.fileType = DataSource.HPCTOOLKIT;
	    } else if (fileTypeString.equals("snapshot")) {
		ParaProf.fileType = DataSource.SNAP;
	    } else if (fileTypeString.equals("snap")) {
	    	seriesSnap=true;
	    	ParaProf.fileType = DataSource.SNAP;
	    } else if (fileTypeString.equals("ompp")) {
		ParaProf.fileType = DataSource.OMPP;
	    } else if (fileTypeString.equals("perixml")) {
		ParaProf.fileType = DataSource.PERIXML;
	    } else if (fileTypeString.equals("gptl")) {
		ParaProf.fileType = DataSource.GPTL;
	    } else if (fileTypeString.equals("paraver")) {
		ParaProf.fileType = DataSource.PARAVER;
	    } else if (fileTypeString.equals("ipm")) {
		ParaProf.fileType = DataSource.IPM;
	    } else if (fileTypeString.equals("google")) {
		ParaProf.fileType = DataSource.GOOGLE;
	    } else if (fileTypeString.equals("darshan")){
	    ParaProf.fileType = DataSource.DARSHAN;
	    }
	    else {
		System.err.println("Please enter a valid file type.");
		ParaProf.usage();
		System.exit(-1);
	    }
	} else {
	    if (sourceFilenames.length >= 1) {
		ParaProf.fileType = UtilFncs.identifyData(sourceFiles[0]);
	    }
	}

	if (merge != null) {
	    try {

		if (sourceFiles.length == 0) {
		    FileList fl = new FileList();
		    sourceFiles = fl.helperFindSnapshots(System.getProperty("user.dir"));
		}

		if (sourceFiles.length == 0) {
		    System.err.println("No snapshots found\n");
		    System.exit(-1);
		}

		for (int i = 0; i < sourceFiles.length; i++) {
		    if (DataSource.SNAP != UtilFncs.identifyData(sourceFiles[i])) {
			System.err.println("Error: File '" + sourceFiles[i] + "' is not a snapshot profile\n");
			System.exit(-1);
		    }
		}

		// merge and write
		UtilFncs.mergeSnapshots(sourceFiles, merge);

	    } catch (Exception e) {
		e.printStackTrace();
	    }
	    System.exit(0);
	}

	if (oss != null) {
	    try {

		boolean doSummary = false;
		if (summary != null) {
		    doSummary = true;
		}

		DataSource dataSource = UtilFncs.initializeDataSource(sourceFiles, fileType, ParaProf.fixNames);
		dataSource.load();
		OSSWriter.writeOSS(dataSource, doSummary);

	    } catch (Exception e) {
		e.printStackTrace();
	    }
	    System.exit(0);
	}

	if (pack != null) {
	    try {

	    File outPPK = new File(pack);
	    if(outPPK.exists()){
	    	System.out.println("The file "+pack+" already exists. Do you want to overwrite?");
	    	Scanner scin = new Scanner(System.in);
	    	
	    	
	    	boolean valid = false;
	    	boolean write = false;
	    	while(!valid){
	    	System.out.print("[yes/no]: ");
	    	String response = scin.nextLine();
	    	response = response.toLowerCase().trim();
	    	
	    	if(response.equals("yes")||response.equals("y")){
	    		write=true;
	    		valid=true;
	    	}
	    	else if(response.equals("no")||response.equals("n")){
	    		valid=true;
	    	}
	    	}
	    	scin.close();
	    	if(!write){
	    		System.exit(0);
	    	}
	    }
		DataSource dataSource = UtilFncs.initializeDataSource(sourceFiles, fileType, ParaProf.fixNames);
		System.out.println("Loading data...");
		dataSource.load();
		System.out.println("Packing data...");
		DataSourceExport.writePacked(dataSource, outPPK);

	    } catch (Exception e) {
		e.printStackTrace();
	    }
	    System.exit(0);
	}
	
	if(writeComm!=null) {
		try {
			DataSource dataSource = UtilFncs.initializeDataSource(sourceFiles, fileType, ParaProf.fixNames);
			System.out.println("Loading data...");
			dataSource.load();
			System.out.println("Writing data...");
			CommunicationMatrixWindow.writeCommCSV(dataSource, new FileOutputStream(new File(writeComm)),-1);
		}catch (Exception e) {
		e.printStackTrace();
	    }
		System.exit(0);
	}

	if (text != null) {
	    try {

		DataSource dataSource = UtilFncs.initializeDataSource(sourceFiles, fileType, ParaProf.fixNames);
		System.out.println("Loading data...");
		dataSource.load();
		System.out.println("Writing data...");
		DataSourceExport.writeDelimited(dataSource, new FileOutputStream(new File(text)));

	    } catch (Exception e) {
		e.printStackTrace();
	    }
	    System.exit(0);
	}

	if (unpack != null && unpack.booleanValue()) {
	    try {

		FileList fl = new FileList();
		List<File[]> v = fl.helperFindProfiles(".");

		if (overwrite == null&&(unpackMPISumm==null||!unpackMPISumm)&&(metadump==null||!metadump)) {
		    if (v.size() != 0) {
			System.err.println("Error: profiles found in current directory, please remove first");
			return;
		    }
		}

		DataSource dataSource = UtilFncs.initializeDataSource(sourceFiles, fileType, ParaProf.fixNames);
		Set<Integer> ranks = null;
		System.out.println("Loading data...");
		if (unpackrank != null) {
		    ranks= getRankList(unpackrank);// = Integer.parseInt(unpackrank);
		   dataSource.setSelectedRank(ranks);
		}
		
			dataSource.load();
		
		System.out.println("Creating TAU Profile data...");
		if (unpackrank != null) {
		    //int rank = Integer.parseInt(unpackrank);
			for(Integer i:ranks) {
				 Node node = dataSource.getNode(i);
				 if(node==null) {
					 System.out.println("No profile for rank "+i);
				 }
				 else{
					 DataSourceExport.writeProfiles(dataSource, new File("."), node.getThreads());
				 }
			}
		   
		}else if(unpackSumm !=null){
			DataSourceExport.writeAggProfiles(dataSource, new File("."));
		}else if(unpackMPISumm !=null||metadump!=null){
			boolean dumpMetadata = false;
			if(metadump!=null)
				dumpMetadata=metadump.booleanValue();
			boolean sup = false;
			if(suppress!=null)
				sup = suppress.booleanValue();
			boolean unpackmpi = false;
			if(unpackMPISumm!=null)
				unpackmpi=unpackMPISumm.booleanValue();
			if(unpackmpi)
				DataSourceExport.writeAggMPISummary(dataSource,sup,dumpMetadata);
			else if(dumpMetadata){
				DataSourceExport.writeMetaDataSummary(dataSource);
			}
				
		}else {
		    DataSourceExport.writeProfiles(dataSource, new File("."));
		}

	    } catch (Exception e) {
		e.printStackTrace();
	    }
	    System.exit(0);
	}

	if (controlMode) {
	    ParaProf.initialize();
	    ParaProf.loadScripts();
	    ExternalController.runController();
	} else {
		
		
	    javax.swing.SwingUtilities.invokeLater(new Runnable() {
		public void run() {
		    try {
			ParaProf.initialize();
			ParaProf.loadScripts();
			ParaProf.loadDefaultTrial(range);
		    } catch (Exception e) {
			ParaProfUtils.handleException(e);
		    }
		}
	    });
	}
    }
    
    private static Set<Integer> getRankList(String ranks) {
    	
    	ranks=ranks.trim();
    	String[] rankstrings = ranks.split(",");
    	Set<Integer> rankSet=new LinkedHashSet<Integer>();
    	for(String s:rankstrings) {
    		s=s.trim();
    		if(s.contains("-")) {
    			String[] range = s.split("-");
    			if(range.length!=2) {
    				System.out.println("Invalid range specification: "+s);
    				return null;
    			}
    			int from = Integer.parseInt(range[0]);
    			int to = Integer.parseInt(range[1]);
    			if(from>to) {
    				int tmp=from;
    				from=to;
    				to=tmp;
    			}
    			for(int i=from;i<=to;i++) {
    				rankSet.add(i);
    			}
    		}else {
    			rankSet.add(Integer.parseInt(s));
    		}
    			
    	}
    	
    	return rankSet;
    }
    
    private static void setUIFont(javax.swing.plaf.FontUIResource f)
    {
        Enumeration<Object> keys = UIManager.getDefaults().keys();
        while (keys.hasMoreElements())
        {
            Object key = keys.nextElement();
            Object value = UIManager.get(key);
            if (value instanceof javax.swing.plaf.FontUIResource)
            {
                UIManager.put(key, f);
            }
        }
    }
}
