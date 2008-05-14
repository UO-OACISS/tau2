package edu.uoregon.tau.paraprof;

import jargs.gnu.CmdLineParser;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.net.URL;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import javax.swing.ToolTipManager;

import edu.uoregon.tau.common.TauScripter;
import edu.uoregon.tau.paraprof.interfaces.EclipseHandler;
import edu.uoregon.tau.paraprof.script.ParaProfScript;
import edu.uoregon.tau.paraprof.sourceview.SourceManager;
import edu.uoregon.tau.perfdmf.*;

/**
 * ParaProf This is the 'main' for paraprof
 * 
 * <P>
 * CVS $Id: ParaProf.java,v 1.22 2008/05/14 23:34:58 amorris Exp $
 * </P>
 * 
 * @author Robert Bell, Alan Morris
 * @version $Revision: 1.22 $
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

    private final static String VERSION = "Wed May 14 16:34:22 PDT 2008";

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

    //Command line options related.
    private static int fileType = 0; //0:profile, 1:pprof, 2:dynaprof, 3:mpip, 4:hpmtoolkit, 5:gprof, 6:psrun, 7:ppk, 8:cube
    private static File sourceFiles[] = new File[0];
    private static boolean fixNames = false;
    private static boolean monitorProfiles;
    private static String args[];
    //End - Command line options related.

    public static boolean demoMode;
    public static boolean usePathNameInTrial = false;
    public static FunctionBarChartWindow theComparisonWindow;
    public static boolean JNLP = false;
    public static List scripts = new ArrayList();
    public static String scriptFile;

    public static boolean insideEclipse;
    public static EclipseHandler eclipseHandler;
    public static SourceManager directoryManager;
    public static String tauHome;
    public static String tauArch;
    
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
        System.err.println("Usage: paraprof [--pack <file>] [--dump] [-p] [-m] [-i] [-f <filetype>] <files/directory>\n\n"
                + "try `paraprof --help` for more information");
    }

    private static void outputHelp() {
        System.err.println("Usage: paraprof [options] <files/directory> \n\n" + "Options:\n\n"
                + "  -f, --filetype <filetype>       Specify type of performance data, options are:\n"
                + "                                    profiles (default), pprof, dynaprof, mpip,\n"
                + "                                    gprof, psrun, hpm, packed, cube, hpc, ompp\n"
                + "                                    snap, perixml, gptl\n"
                + "  -h, --help                      Display this help message\n"
                + "  -p                              Use `pprof` to compute derived data\n"
                + "  -i, --fixnames                  Use the fixnames option for gprof\n"
                + "  -m, --monitor                   Perform runtime monitoring of profile data\n" 
                + "\n"
                + "The following options will run only from the console (no GUI will launch):\n"
                + "\n"
                + "  --pack <file>                   Pack the data into packed (.ppk) format\n"
                + "  --dump                          Dump profile data to TAU profile format\n"
                + "  -o, --oss                       Print profile data in OSS style text output\n"
                + "  -s, --summary                   Print only summary statistics\n" 
                + "                                    (only applies to OSS output)\n" 
                + "\n" 
                + "Notes:\n"
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

    public static void loadDefaultTrial() {

        // Create a default application.
        ParaProfApplication app = ParaProf.applicationManager.addApplication();
        app.setName("Default App");

        // Create a default experiment.
        ParaProfExperiment experiment = app.addExperiment();
        experiment.setName("Default Exp");

        ParaProf.paraProfManagerWindow.setVisible(true);

        try {

            if (fileType == 7) {
                for (int i = 0; i < sourceFiles.length; i++) {
                    File files[] = new File[1];
                    files[0] = sourceFiles[i];
                    paraProfManagerWindow.addTrial(app, experiment, files, fileType, fixNames, monitorProfiles);
                }
            } else {
                paraProfManagerWindow.addTrial(app, experiment, sourceFiles, fileType, fixNames, monitorProfiles);
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
                    ParaProf.preferences.setLoaded(true);
                    colorChooser = new ColorChooser(ParaProf.preferences);
                } catch (Exception e) {
                    if (e instanceof FileNotFoundException) {
                        //System.out.println("No preference file present, using defaults!");
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
                    //System.out.println("Found db configuration file: "
                    //        + ParaProf.paraProfHomeDirectory.getPath() + "/perfdmf.cfg");
                    ParaProf.preferences.setDatabaseConfigurationFile(ParaProf.paraProfHomeDirectory.getPath() + "/perfdmf.cfg");
                } else {
                    System.out.println("Did not find db configuration file ... load manually");
                }

            } else {
                System.out.println("Did not find ParaProf home directory ... creating ...");
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
        }

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
        ParaProf.paraProfManagerWindow = new ParaProfManagerWindow();
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
        long memUsage = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024;

        return new String("ParaProf\n" + getVersionString() + "\nJVM Heap Size: " + memUsage + "kb\n"
                + "http://www.cs.uoregon.edu/research/tau\n");
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
        ParaProf.preferences.setLoaded(true);
        colorChooser = new ColorChooser(ParaProf.preferences);

        ParaProf.colorMap.setMap(ParaProf.preferences.getAssignedColors());

        List trials = ParaProf.paraProfManagerWindow.getLoadedTrials();
        for (Iterator it = trials.iterator(); it.hasNext();) {
            ParaProfTrial ppTrial = (ParaProfTrial) it.next();
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
            // we'll get an exception here if running under Java Web Start
        }
        if (!insideEclipse) {
            // never call System.exit when invoked by the eclipse plugin, it will close the whole JVM, including the user's eclipse!
            System.exit(exitValue);
        }
    }

    public static boolean savePreferences(File file) {

        ParaProf.colorChooser.setSavedColors();
        ParaProf.preferences.setAssignedColors(ParaProf.colorMap.getMap());
        ParaProf.preferences.setManagerWindowPosition(ParaProf.paraProfManagerWindow.getLocation());
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
        ParaProf.args = args;

        // Set the tooltip delay to 20 seconds
        ToolTipManager.sharedInstance().setDismissDelay(20000);

        // Process command line arguments
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option typeOpt = parser.addStringOption('f', "filetype");
        CmdLineParser.Option fixOpt = parser.addBooleanOption('i', "fixnames");
        CmdLineParser.Option packOpt = parser.addStringOption('a', "pack");
        CmdLineParser.Option unpackOpt = parser.addBooleanOption('u', "dump");
        CmdLineParser.Option ossOpt = parser.addBooleanOption('o', "oss");
        CmdLineParser.Option summaryOpt = parser.addBooleanOption('s', "summary");
        CmdLineParser.Option monitorOpt = parser.addBooleanOption('m', "monitor");
        CmdLineParser.Option demoOpt = parser.addBooleanOption('z', "demo");
        CmdLineParser.Option tauHomeOpt = parser.addStringOption('t', "tauhome");
        CmdLineParser.Option tauArchOpt = parser.addStringOption('a', "tauarch");

        try {
            parser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.err.println("paraprof: " + e.getMessage());
            ParaProf.usage();
            System.exit(-1);
        }

        Boolean help = (Boolean) parser.getOptionValue(helpOpt);
        String fileTypeString = (String) parser.getOptionValue(typeOpt);
        Boolean fixNames = (Boolean) parser.getOptionValue(fixOpt);
        String pack = (String) parser.getOptionValue(packOpt);
        Boolean unpack = (Boolean) parser.getOptionValue(unpackOpt);
        Boolean oss = (Boolean) parser.getOptionValue(ossOpt);
        Boolean summary = (Boolean) parser.getOptionValue(summaryOpt);
        Boolean monitor = (Boolean) parser.getOptionValue(monitorOpt);
        Boolean demo = (Boolean) parser.getOptionValue(demoOpt);
        ParaProf.tauHome = (String) parser.getOptionValue(tauHomeOpt);
        ParaProf.tauArch = (String) parser.getOptionValue(tauArchOpt);

        demoMode = demo != null && demo.booleanValue();

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
            } else if (fileTypeString.equals("snap")) {
                ParaProf.fileType = DataSource.SNAP;
            } else if (fileTypeString.equals("ompp")) {
                ParaProf.fileType = DataSource.OMPP;
            } else if (fileTypeString.equals("perixml")) {
                ParaProf.fileType = DataSource.PERIXML;
            } else if (fileTypeString.equals("gptl")) {
                ParaProf.fileType = DataSource.GPTL;
            } else {
                System.err.println("Please enter a valid file type.");
                ParaProf.usage();
                System.exit(-1);
            }
        } else {
            if (sourceFilenames.length >= 1) {
                String filename = sourceFiles[0].getName();
                if (filename.toLowerCase().endsWith(".ppk")) {
                    ParaProf.fileType = 7;
                }
                if (filename.toLowerCase().endsWith(".cube")) {
                    ParaProf.fileType = 8;
                }
                if (filename.toLowerCase().endsWith(".mpip")) {
                    ParaProf.fileType = 3;
                }
            }
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

                DataSource dataSource = UtilFncs.initializeDataSource(sourceFiles, fileType, ParaProf.fixNames);
                System.out.println("Loading data...");
                dataSource.load();
                System.out.println("Packing data...");
                DataSourceExport.writePacked(dataSource, new File(pack));

            } catch (Exception e) {
                e.printStackTrace();
            }
            System.exit(0);
        }

        if (unpack != null && unpack.booleanValue()) {
            try {

                FileList fl = new FileList();
                List v = fl.helperFindProfiles(".");
                if (v.size() != 0) {
                    System.err.println("Error: profiles found in current directory, please remove first");
                    return;
                }

                DataSource dataSource = UtilFncs.initializeDataSource(sourceFiles, fileType, ParaProf.fixNames);
                System.out.println("Loading data...");
                dataSource.load();
                System.out.println("Creating TAU Profile data...");
                DataSourceExport.writeProfiles(dataSource, new File("."));

            } catch (Exception e) {
                e.printStackTrace();
            }
            System.exit(0);
        }

        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                try {
                    ParaProf.initialize();
                    ParaProf.loadScripts();
                    ParaProf.loadDefaultTrial();
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }
        });
    }
}
