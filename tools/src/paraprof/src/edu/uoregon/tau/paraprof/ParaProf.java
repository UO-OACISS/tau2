package edu.uoregon.tau.paraprof;

import java.util.*;
import java.io.*;
import java.awt.event.*;
import jargs.gnu.CmdLineParser;
import edu.uoregon.tau.dms.dss.*;
import javax.swing.*;

/**
 * ParaProf This is the 'main' for paraprof
 * 
 * <P>
 * CVS $Id: ParaProf.java,v 1.33 2005/04/04 22:26:00 amorris Exp $
 * </P>
 * 
 * @author Robert Bell, Alan Morris
 * @version $Revision: 1.33 $
 */
public class ParaProf implements ActionListener {

    static public class XThrowableHandler {
        
        public XThrowableHandler() {
        }
        
        public void handle(Throwable t) throws Throwable {
            if (t instanceof Exception) {
                ParaProfUtils.handleException((Exception) t);
            } else {
                System.err.println("Uncaught Throwable: " + t.fillInStackTrace());
            }
        }
    }

    private final static String VERSION = "2.1 (with TAU 2.14.1) (01/21/2005)";

    static ColorMap colorMap = new ColorMap();

    //System wide stuff.
    static String homeDirectory = null;
    static File paraProfHomeDirectory = null;
    static String profilePathName = null; //This contains the path to the
    // currently loaded profile data.
    static int defaultNumberPrecision = 6;
    static boolean dbSupport = false;
    //static ParaProfLisp paraProfLisp = null;
    static Preferences preferences = null;
    static ColorChooser colorChooser;

    static ParaProfManagerWindow paraProfManager = null;
    static ApplicationManager applicationManager = null;
    static HelpWindow helpWindow = null;
    static PreferencesWindow preferencesWindow;
    static Runtime runtime = null;
    static private int numWindowsOpen = 0;
    //End - System wide stuff.

    //Command line options related.
    private static int fileType = 0; //0:profile, 1:pprof, 2:dynaprof, 3:mpip, 4:hpmtoolkit, 5:gprof, 6:psrun
    private static File sourceFiles[] = new File[0];
    private static boolean fixNames = false;
    //End - Command line options related.

    private ParaProfTrial pptrial = null;

    public ParaProf() {

        try {
            //            //UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); }
            //            UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
            //
            //            
            //            UIManager.setLookAndFeel(
            //                                     "com.sun.java.swing.plaf.gtk.GTKLookAndFeel");
            //     
            //            UIManager.setLookAndFeel(
            //            "javax.swing.plaf.metal.MetalLookAndFeel");
            //
            //            UIManager.setLookAndFeel(
            //            "com.sun.java.swing.plaf.motif.MotifLookAndFeel");

        } catch (Exception e) {

        }

    }

    private static void usage() {
        System.err.println("Usage: paraprof [-d] [-p] [-i] [-f <filetype>] <files/directory>\n\n"
                + "try `paraprof --help` for more information");
    }

    private static void outputHelp() {
        System.err.println("Usage: paraprof [options] <files/directory> \n\n" + "Options:\n\n"
                + "  -f, --filetype <filetype>        Specify type of performance data, options are:\n"
                + "                                   profiles (default), pprof, dynaprof, mpip,\n"
                + "                                   gprof, psrun, hpm\n"
                + "  -h, --help                       Display this help message\n"
                + "  -p                               Use `pprof` to compute derived data\n"
                + "  -i, --fixnames                   Use the fixnames option for gprof\n\n" + "Notes:\n"
                + "  For the TAU profiles type, you can specify either a specific set of profile\n"
                + "files on the commandline, or you can specify a directory (by default the current\n"
                + "directory).  The specified directory will be searched for profile.*.*.* files,\n"
                + "or, in the case of multiple counters, directories named MULTI_* containing\n"
                + "profile data.\n\n");
    }

    static void incrementNumWindows() {
        //        System.out.println ("incrementing");
        numWindowsOpen++;
    }

    static void decrementNumWindows() {
        //        System.out.println ("decrementing");
        numWindowsOpen--;
        if (numWindowsOpen <= 0) {
            exitParaProf(0);
        }
    }

    public void loadDefaultTrial() {
        ParaProf.applicationManager = new ApplicationManager();

        // Create a default application.
        ParaProfApplication app = ParaProf.applicationManager.addApplication();
        app.setName("Default App");

        // Create a default experiment.
        ParaProfExperiment experiment = app.addExperiment();
        experiment.setName("Default Exp");

        ParaProf.helpWindow = new HelpWindow();
        ParaProf.paraProfManager = new ParaProfManagerWindow();

        paraProfManager.addTrial(app, experiment, sourceFiles, fileType, fixNames);
    }

    public void startSystem() {
        try {
            // Initialization of static objects takes place on a need basis.
            // This helps prevent the creation of a graphical system unless it is absolutely
            // necessary. Static initializations are marked with "Static Initialization" 
            // to make them easy to find.

            ParaProf.preferences = new Preferences();

            //Establish the presence of a .ParaProf directory. This is located
            // by default in the user's home directory.
            ParaProf.paraProfHomeDirectory = new File(homeDirectory + "/.ParaProf");
            if (paraProfHomeDirectory.exists()) {

                //Try and load a preference file ... ParaProfPreferences.dat
                try {
                    FileInputStream savedPreferenceFIS = new FileInputStream(
                            ParaProf.paraProfHomeDirectory.getPath() + "/ParaProf.conf");

                    //If here, means that no exception was thrown, and there is a preference file present.
                    //Create ObjectInputStream and try to read it in.
                    ObjectInputStream inSavedPreferencesOIS = new ObjectInputStream(savedPreferenceFIS);
                    ParaProf.preferences = (Preferences) inSavedPreferencesOIS.readObject();
                    ParaProf.preferences.setLoaded(true);
                    colorChooser = new ColorChooser(ParaProf.preferences);
                } catch (Exception e) {
                    if (e instanceof FileNotFoundException) {
                        //System.out.println("No preference file present, using defaults!");
                    } else {
                        //Print some kind of error message, and quit the system.
                        System.out.println("Error while trying to read the ParaProf preferences file, using defaults");
                        //                        System.out.println("Please delete this file, or replace it with a valid one!");
                        //                        System.out.println("Note: Deleting the file will cause ParaProf to restore the default preferences");
                    }
                }

                ParaProf.colorMap.setMap(preferences.getAssignedColors());

                //Try and find perfdmf.cfg.
                File perfDMFcfg = new File(ParaProf.paraProfHomeDirectory.getPath() + "/perfdmf.cfg");
                if (perfDMFcfg.exists()) {
                    //System.out.println("Found db configuration file: "
                    //        + ParaProf.paraProfHomeDirectory.getPath() + "/perfdmf.cfg");
                    ParaProf.preferences.setDatabaseConfigurationFile(ParaProf.paraProfHomeDirectory.getPath()
                            + "/perfdmf.cfg");
                } else
                    System.out.println("Did not find db configuration file ... load manually");
            } else {
                System.out.println("Did not find ParaProf home directory ... creating ...");
                paraProfHomeDirectory.mkdir();
                System.out.println("Done creating ParaProf home directory!");
            }

            if (colorChooser == null) {
                ParaProf.colorChooser = new ColorChooser(null);
            }

            ParaProf.preferencesWindow = new PreferencesWindow(preferences);

            javax.swing.SwingUtilities.invokeLater(new Runnable() {
                public void run() {
                    System.setProperty("sun.awt.exception.handler", XThrowableHandler.class.getName());
                    loadDefaultTrial();
                }
            });

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
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
        return new String("ParaProf Version " + getVersionString() + "\n Java Heap Size: " + memUsage + "kb");
    }

    public static String getVersionString() {
        return new String(VERSION);
    }

    public static void loadPreferences(File file) throws FileNotFoundException, IOException,
            ClassNotFoundException {

        FileInputStream savedPreferenceFIS = new FileInputStream(file);

        //If here, means that no exception was thrown, and there is a preference file present.
        //Create ObjectInputStream and try to read it in.
        ObjectInputStream inSavedPreferencesOIS = new ObjectInputStream(savedPreferenceFIS);
        ParaProf.preferences = (Preferences) inSavedPreferencesOIS.readObject();
        ParaProf.preferences.setLoaded(true);
        colorChooser = new ColorChooser(ParaProf.preferences);

        ParaProf.colorMap.setMap(ParaProf.preferences.getAssignedColors());

        Vector trials = ParaProf.paraProfManager.getLoadedTrials();
        for (Iterator it = trials.iterator(); it.hasNext();) {
            ParaProfTrial ppTrial = (ParaProfTrial) it.next();
            ParaProf.colorChooser.setColors(ppTrial, -1);
            ppTrial.getSystemEvents().updateRegisteredObjects("colorEvent");
            ppTrial.getSystemEvents().updateRegisteredObjects("prefEvent");
        }

    }

    // This method is reponsible for any cleanup required in ParaProf 
    // before an exit takes place.
    public static void exitParaProf(int exitValue) {
        //try {
        //   throw new Exception();
        //} catch (Exception e) {
        //   e.printStackTrace();
        //}

        //        if (
        //        File file = new File(ParaProf.paraProfHomeDirectory.getPath() + "/ParaProf.prefs");
        //
        //        try {
        //            ObjectOutputStream prefsOut = new ObjectOutputStream(new FileOutputStream(file));
        //            this.setSavedPreferences();
        //            prefsOut.writeObject(ParaProf.savedPreferences);
        //            prefsOut.close();
        //        } catch (Exception e) {
        //            //Display an error
        //            JOptionPane.showMessageDialog(this,
        //                    "An error occured while trying to save ParaProf preferences.", "Error!",
        //                    JOptionPane.ERROR_MESSAGE);
        //        }

        savePreferences(new File(ParaProf.paraProfHomeDirectory.getPath() + "/ParaProf.conf"));

        System.exit(exitValue);
    }

    public static boolean savePreferences(File file) {

        ParaProf.colorChooser.setSavedColors();
        ParaProf.preferences.setAssignedColors(ParaProf.colorMap.getMap());
        ParaProf.preferences.setManagerWindowPosition(ParaProf.paraProfManager.getLocation());

        //        System.out.println ("saving manager position = " + preferences.getManagerWindowPosition());

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

        ParaProf.homeDirectory = System.getProperty("user.home");

        //Bring ParaProf into being!
        ParaProf paraProf = new ParaProf();

        //######
        //Process command line arguments.
        //ParaProf has numerous modes of operation. A number of these mode
        //can be specified on the command line.
        //######
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option debugOpt = parser.addBooleanOption('d', "debug");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option typeOpt = parser.addStringOption('f', "filetype");
        CmdLineParser.Option fixOpt = parser.addBooleanOption('i', "fixnames");
        try {
            parser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.err.println("paraprof: " + e.getMessage());
            ParaProf.usage();
            exitParaProf(-1);
        }

        Boolean help = (Boolean) parser.getOptionValue(helpOpt);
        Boolean debug = (Boolean) parser.getOptionValue(debugOpt);
        String fileTypeString = (String) parser.getOptionValue(typeOpt);
        Boolean fixNames = (Boolean) parser.getOptionValue(fixOpt);

        if (help != null && help.booleanValue()) {
            ParaProf.outputHelp();
            exitParaProf(-1);
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
                ParaProf.fileType = 0;
            } else if (fileTypeString.equals("pprof")) {
                ParaProf.fileType = 1;
            } else if (fileTypeString.equals("dynaprof")) {
                ParaProf.fileType = 2;
            } else if (fileTypeString.equals("mpip")) {
                ParaProf.fileType = 3;
            } else if (fileTypeString.equals("hpm")) {
                ParaProf.fileType = 4;
            } else if (fileTypeString.equals("gprof")) {
                ParaProf.fileType = 5;
            } else if (fileTypeString.equals("psrun")) {
                ParaProf.fileType = 6;
            } else {
                System.err.println("Please enter a valid file type.");
                ParaProf.usage();
                exitParaProf(-1);
            }
        }

        ParaProf.runtime = Runtime.getRuntime();

        //        if (UtilFncs.debug) {
        //            //Create and start the a timer, and then add paraprof to it.
        //            javax.swing.Timer jTimer = new javax.swing.Timer(8000, paraProf);
        //            jTimer.start();
        //        }

        paraProf.startSystem();
    }
}