/**
 * ParaProf This is the 'main' for paraprof
 * 
 * <P>
 * CVS $Id: ParaProf.java,v 1.20 2004/12/21 00:44:00 amorris Exp $
 * </P>
 * 
 * @author Robert Bell, Alan Morris
 * @version $Revision: 1.20 $
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.io.*;
import java.awt.event.*;
import jargs.gnu.CmdLineParser;
import edu.uoregon.tau.dms.dss.*;

public class ParaProf implements ActionListener {

    //System wide stuff.
    static String homeDirectory = null;
    static File paraProfHomeDirectory = null;
    static String profilePathName = null; //This contains the path to the
    // currently loaded profile data.
    static int defaultNumberPrecision = 6;
    static boolean dbSupport = false;
    static ParaProfLisp paraProfLisp = null;
    static SavedPreferences savedPreferences = null;
    static ParaProfManager paraProfManager = null;
    static ApplicationManager applicationManager = null;
    static HelpWindow helpWindow = null;
    static Runtime runtime = null;
    static boolean runHasBeenOpened = false;
    static private int numWindowsOpen = 0;
    //End - System wide stuff.

    //Command line options related.
    private static int fileType = 0; //0:profile, 1:pprof, 2:dynaprof, 3:mpip,
    // 4:hpmtoolkit, 5:gprof, 6:psrun
    private static File sourceFiles[] = new File[0];
    private static boolean fixNames = false;
    //End - Command line options related.

    private ParaProfTrial pptrial = null;

    public ParaProf() {
        /*
         * try {
         * UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); }
         * catch (Exception e) { }
         */
        //End uncomment!
    }

    
    private static void usage() {
        System.err.println("Usage: paraprof [-d] [-p] [-i] [-f <filetype>] <files/directory>\n\n" +
                	"try `paraprof --help` for more information");
    }
    
    private static void outputHelp() {
        System.err.println("Usage: paraprof [options] <files/directory> \n\n"
                + "Options:\n\n"
                + "  -f, --filetype <filetype>        Specify type of performance data, options are:\n"
                + "                                   profiles (default), pprof, dynaprof, mpip,\n"
                + "                                   gprof, psrun, hpm\n"
                + "  -h, --help                       Display this help message\n"
                + "  -p                               Use `pprof` to compute derived data\n"
                + "  -d, --debug                      Turn debug statements on\n"
                + "  -i, --fixnames                   Use the fixnames option for gprof\n\n"
                + "Notes:\n"
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

        //Create a default application.
        ParaProfApplication app = ParaProf.applicationManager.addApplication();
        app.setName("Default App");

        //Create a default experiment.
        ParaProfExperiment experiment = app.addExperiment();
        experiment.setName("Default Exp");
        DataSource dataSource = null;
        FileList fl = new FileList();
        Vector v = null;

        ParaProf.helpWindow = new HelpWindow(UtilFncs.debug);
        ParaProf.paraProfManager = new ParaProfManager();

        paraProfManager.addTrial(app, experiment, sourceFiles, fileType, fixNames);
    }

    public void startSystem() {
        try {

            // Initialization of static objects takes place on a need basis.
            // This helps prevent the creation of a graphical system unless it is absolutely
            // necessary. Static initializations are marked with "Static Initialization" 
            // to make them easy to find.

            //Static Initialization
            ParaProf.savedPreferences = new SavedPreferences();
            //End - Static Initialization

            //Establish the presence of a .ParaProf directory. This is located
            // by default in the user's home
            //directory.
            ParaProf.paraProfHomeDirectory = new File(homeDirectory + "/.ParaProf");
            if (paraProfHomeDirectory.exists()) {
                //Try and load a preference file ... ParaProfPreferences.dat
                try {
                    FileInputStream savedPreferenceFIS = new FileInputStream(
                            ParaProf.paraProfHomeDirectory.getPath() + "/ParaProf.dat");

                    //If here, means that no exception was thrown, and there is
                    // a preference file present.
                    //Create ObjectInputStream and try to read it in.
                    ObjectInputStream inSavedPreferencesOIS = new ObjectInputStream(
                            savedPreferenceFIS);
                    ParaProf.savedPreferences = (SavedPreferences) inSavedPreferencesOIS.readObject();
                } catch (Exception e) {
                    if (e instanceof FileNotFoundException) {
                        System.out.println("No preference file present, using defaults!");
                    } else {
                        //Print some kind of error message, and quit the
                        // system.
                        System.out.println("There was an internal error while trying to read the ParaProf preference");
                        System.out.println("file.  Please delete this file, or replace it with a valid one!");
                        System.out.println("Note: Deleting the file will cause ParaProf to restore the default preferences");
                    }
                }

                //Try and find perfdmf.cfg.
                File perfDMFcfg = new File(ParaProf.paraProfHomeDirectory.getPath()
                        + "/perfdmf.cfg");
                if (perfDMFcfg.exists()) {
                    System.out.println("Found db configuration file: "
                            + ParaProf.paraProfHomeDirectory.getPath() + "/perfdmf.cfg");
                    ParaProf.savedPreferences.setDatabaseConfigurationFile(ParaProf.paraProfHomeDirectory.getPath()
                            + "/perfdmf.cfg");
                } else
                    System.out.println("Did not find db configuration file ... load manually");
            } else {
                System.out.println("Did not find ParaProf home directory ... creating ...");
                paraProfHomeDirectory.mkdir();
                System.out.println("Done creating ParaProf home directory!");
            }

            //######
            //Static Initialization
            //######
            //ParaProf.paraProfLisp = new ParaProfLisp(UtilFncs.debug);
            //######
            //End - Static Initialization
            //######
            //Register lisp primatives in ParaProfLisp.
            //ParaProf.paraProfLisp.registerParaProfPrimitives();

            /*
             * //See if the user has defined any lisp code to run. try{
             * FileInputStream file = new FileInputStream("ParaProfLisp.lp");
             * //If here, means that no exception was thrown, and there is a
             * lisp file present. InputStreamReader isr = new
             * InputStreamReader(file); BufferedReader br = new
             * BufferedReader(isr);
             * 
             * String inputString = null;
             * 
             * while((inputString = br.readLine()) != null){
             * System.out.println("Expression: " + inputString);
             * System.out.println(ParaProf.paraProfLisp.eval(inputString)); } }
             * catch(Exception e){ if(e instanceof FileNotFoundException){
             * System.out.println("No ParaProfLisp.lp file present!"); } else{
             * //Print some kind of error message, and quit the system.
             * System.out.println("There was an internal error while trying to
             * read the ParaProfLisp.pl"); System.out.println("Please delete
             * this file, or replace it with a valid one!"); } }
             */

            loadDefaultTrial();

        } catch (Exception e) {
            System.out.println("An un-caught exception has occurred within the program!");
            System.out.println("The details of this exception have been stored in a file named: exception.err");
            System.out.println("Please email this file to Robert Bell at bertie@cs.uoregon.edu ");
            e.printStackTrace();
        }
    }

    public void actionPerformed(ActionEvent evt) {
        Object EventSrc = evt.getSource();
        if (EventSrc instanceof javax.swing.Timer) {
            System.out.println("------------------------");
            System.out.println("The amount of memory used by the system is: "
                    + runtime.totalMemory());
            System.out.println("The amount of memory free to the system is: "
                    + runtime.freeMemory());
        }
    }

    public static String getInfoString() {
        long memUsage = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024;
        return new String("ParaProf Version 2.0 ... The Tau Group!\n Java Heap Size: " + memUsage
                + "kb");
    }

    // This method is reponsible for any cleanup required in ParaProf 
    // before an exit takes place.
    public static void exitParaProf(int exitValue) {

        //try {
        //   throw new Exception();
        //} catch (Exception e) {
        //   e.printStackTrace();
        //}

        if (UtilFncs.objectDebug != null) {
            UtilFncs.objectDebug.outputToFile("ParaProf exiting!!");
            UtilFncs.objectDebug.flushDebugFileBuffer();
            UtilFncs.objectDebug.closeDebugFile();
        }
        System.exit(exitValue);
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

        if (debug != null) {
            //######
            //Static Initialization
            //######
            UtilFncs.objectDebug = new Debug();
            //######
            //End - Static Initialization
            //######

            UtilFncs.debug = debug.booleanValue();
            Debug.debug = debug.booleanValue();
        }

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
                /*
                 * } else if (fileTypeString.equals("sppm")) { ParaProf.fileType =
                 * 101; } else if (fileTypeString.equals("xprof")) {
                 * ParaProf.fileType = 0; } else if
                 * (fileTypeString.equals("sddf")) { ParaProf.fileType = 0;
                 */
            } else {
                System.err.println("Please enter a valid file type.");
                ParaProf.usage();
                exitParaProf(-1);
            }
        }

        ParaProf.runtime = Runtime.getRuntime();

        if (UtilFncs.debug) {
            //Create and start the a timer, and then add paraprof to it.
            //javax.swing.Timer jTimer = new javax.swing.Timer(8000, paraProf);
            //jTimer.start();
        }

        paraProf.startSystem();
    }
}