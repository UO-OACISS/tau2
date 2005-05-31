package edu.uoregon.tau.dms.loader;

import jargs.gnu.CmdLineParser;

import java.io.*;
import java.util.List;
import java.util.Vector;

import edu.uoregon.tau.dms.dss.*;

public class LoadTrial {

    //    public static String USAGE = "usage: perfdmf_loadtrial {-f, --filetype}
    // file_type {-e,--experimentid} experiment_id\n [{-t, --trialid} trial_id]
    // [{-n,--name} trial_name] [{-i --fixnames}] <files>\n Where:\n file_type =
    // profiles (TAU), pprof (TAU), dynaprof, mpip, gprof, psrun, hpm\n\n
    // (example: perfdmf_loadtrial -f profiles -e 12 profile.*)";
    //private File readPprof;

    public static void usage() {
        System.err.println("Usage: perfdmf_loadtrial -e <experiment id> -n <name> [options] <files>\n\n"
                + "try `perfdmf_loadtrial --help' for more information");
    }

    public static void outputHelp() {

        System.err.println("Usage: perfdmf_loadtrial -e <experiment id> -n <name> [options] <files>\n\n"
                + "Required Arguments:\n\n"
                + "  -e, --experimentid <number>    Specify associated experiment ID for this trial\n"
                + "  -n, --name <text>              Specify the name of the trial\n\n"
                + "Optional Arguments:\n\n"
                + "  -f, --filetype <filetype>      Specify type of performance data, options are:\n"
                + "                                   profiles (default), pprof, dynaprof, mpip,\n"
                + "                                   gprof, psrun, hpm\n"
                + "  -t, --trialid <number>         Specify trial ID\n"
                + "  -i, --fixnames                 Use the fixnames option for gprof\n\n" + "Notes:\n"
                + "  For the TAU profiles type, you can specify either a specific set of profile\n"
                + "files on the commandline, or you can specify a directory (by default the current\n"
                + "directory).  The specified directory will be searched for profile.*.*.* files,\n"
                + "or, in the case of multiple counters, directories named MULTI_* containing\n"
                + "profile data.\n\n" + "Examples:\n\n" + "  perfdmf_loadtrial -e 12 -n \"Batch 001\"\n"
                + "    This will load profile.* (or multiple counters directories MULTI_*) into\n"
                + "    experiment 12 and give the trial the name \"Batch 001\"\n\n"
                + "  perfdmf_loadtrial -e 12 -n \"HPM data 01\" perfhpm*\n"
                + "    This will load perfhpm* files of type HPMToolkit into experiment 12 and give\n"
                + "    the trial the name \"HPM data 01\"\n");
    }

    private File writeXml;
    private String trialTime;
    private String sourceFiles[];
    private Application app;
    private Experiment exp;
    private boolean fixNames = false;
    private int expID = 0;
    public int trialID = 0;
    private int fileType = 0;
    private DataSource dataSource = null;
    public String trialName = new String();
    public String problemFile = new String();

    /*
     * This variable connects translator to DB in order to check whether the
     * app. and exp. associated with the trial data do exist there.
     */
    DatabaseAPI databaseAPI = null;
    Trial trial = null;

    //constructor
    public LoadTrial(String configFileName, String sourceFiles[]) {
        this.sourceFiles = sourceFiles;

        // check for the existence of file
        //readPprof = new File(sourcename);

        databaseAPI = new DatabaseAPI();
        try {
            databaseAPI.initialize(configFileName, true);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

    }

    public boolean checkForExp(String expid) {
        this.expID = Integer.parseInt(expid);

        try {
            exp = databaseAPI.setExperiment(this.expID);
        } catch (Exception e) {

        }
        if (exp == null) {
            System.err.println("Experiment id " + expid + " not found,  please enter a valid experiment ID.");
            System.exit(-1);
            return false;
        } else
            return true;
    }

    public boolean checkForTrial(String trialid) {
        Trial tmpTrial = databaseAPI.setTrial(Integer.parseInt(trialid));
        if (tmpTrial == null)
            return false;
        else
            return true;
    }

    public void loadTrial(int fileType) {

        trial = null;
        this.fileType = fileType;

        List v = null;
        File[] inFile = new File[1];
        File filelist[];
        switch (fileType) {
        case 0:
            if (sourceFiles.length != 1) {
                System.err.println("pprof type: you must specify exactly one file");
                System.exit(-1);
            }
            if (isDirectory(sourceFiles[0])) {
                System.err.println("pprof type: you must specify a file, not a directory");
                System.exit(-1);
            }
            inFile[0] = new File(sourceFiles[0]);
            v = new Vector();
            v.add(inFile);
            dataSource = new TauPprofDataSource(v);
            break;
        case 1:

            FileList fl = new FileList();

            if (sourceFiles.length < 1) {
                v = fl.helperFindProfiles(System.getProperty("user.dir"));
            } else {
                if (isDirectory(sourceFiles[0])) {

                    if (sourceFiles.length > 1) {
                        System.err.println("profiles type: you can only specify one directory");
                        System.exit(-1);
                    }

                    v = fl.helperFindProfiles(sourceFiles[0]);

                } else {

                    v = new Vector();
                    filelist = new File[sourceFiles.length];
                    for (int i = 0; i < sourceFiles.length; i++) {
                        filelist[i] = new File(sourceFiles[i]);
                    }
                    v.add(filelist);
                }

            }

            //fl = new FileList();
            //v = fl.getFileList(new File(System.getProperty("user.dir")),
            // null, fileType, "profile", false);
            //v = helperFindFiles(".", "\\Aprofile\\..*\\..*\\..*\\z");

            dataSource = new TauDataSource(v);
            break;
        case 2:
            filelist = new File[sourceFiles.length];
            for (int i = 0; i < sourceFiles.length; i++) {
                filelist[i] = new File(sourceFiles[i]);
            }
            dataSource = new DynaprofDataSource(filelist);
            break;
        case 3:
            if (sourceFiles.length != 1) {
                System.err.println("MpiP type: you must specify exactly one file");
                System.exit(-1);
            }
            if (isDirectory(sourceFiles[0])) {
                System.err.println("MpiP type: you must specify a file, not a directory");
                System.exit(-1);
            }
            inFile[0] = new File(sourceFiles[0]);
            dataSource = new MpiPDataSource(inFile[0]);
            break;
        case 4:
            v = new Vector();
            filelist = new File[sourceFiles.length];
            for (int i = 0; i < sourceFiles.length; i++) {
                filelist[i] = new File(sourceFiles[i]);
            }
            v.add(filelist);
            dataSource = new HPMToolkitDataSource(v);
            break;
        case 5:
            filelist = new File[sourceFiles.length];
            for (int i = 0; i < sourceFiles.length; i++) {
                filelist[i] = new File(sourceFiles[i]);
            }
            dataSource = new GprofDataSource(filelist, fixNames);
            break;
        case 6:
            v = new Vector();
            filelist = new File[sourceFiles.length];
            for (int i = 0; i < sourceFiles.length; i++) {
                filelist[i] = new File(sourceFiles[i]);
            }
            v.add(filelist);
            dataSource = new PSRunDataSource(v);
            break;
        /*
         * case 101: if (fileExists()) { inFile[0] = new File (sourceFile); v =
         * new Vector(); v.add(inFile); } else { fl = new FileList(); String[]
         * sourcePath = extractSourcePath(); if (sourcePath[0] != null) v =
         * fl.getFileList(new File(sourcePath[0]), null, fileType,
         * sourcePath[1], false); else v = fl.getFileList(new
         * File(System.getProperty("user.dir")), null, fileType, sourceFile,
         * false); } dataSession = new SPPMDataSource(); break;
         */
        default:
            break;
        }

        trial = new Trial();
        trial.setDataSource(dataSource);

        try {
            dataSource.load();
        } catch (Exception e) {
            System.err.println("Error Loading Trial:");
            e.printStackTrace();
        }

        if (trialID == 0)
            saveTrial();
        else
            appendToTrial();

    }

    //    public void writeTrial() {
    //        XMLSupport xmlWriter = new XMLSupport(trial);
    //        xmlWriter.writeXmlFiles(0, writeXml);
    //    }

    public void saveTrial() {
        // if (fileType == 101) return;
        // set some things in the trial
        // 	int[] maxNCT = dataSession.getMaxNCTNumbers();
        // 	trial.setNodeCount(maxNCT[0]+1);
        // 	trial.setNumContextsPerNode(maxNCT[1]+1);
        // 	trial.setNumThreadsPerContext(maxNCT[2]+1);
        trial.setName(trialName);
        //trial.setProblemDefinition(getProblemString());

        System.out.println("TrialName: " + trialName);
        trial.setExperimentID(expID);
        try {
            databaseAPI.saveParaProfTrial(trial, -1);
        } catch (DatabaseException e) {
            e.printStackTrace();
            Exception e2 = e.getException();
            System.out.println ("from: ");
            e2.printStackTrace();
            System.exit(-1);
        }
        System.out.println("Done saving trial!");
    }

    public void appendToTrial() {
        // set some things in the trial
        trial.setID(this.trialID);
        try {
            databaseAPI.saveParaProfTrial(trial, 0);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        System.out.println("Done adding metric to trial!");
    }

    public String getProblemString() {
        // if the file wasn't passed in, this is an existing trial.
        if (problemFile == null)
            return new String("");

        // open the file
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(problemFile));
        } catch (Exception e) {
            System.out.println("Problem file not found!  Exiting...");
            System.exit(0);
        }
        // read the file, one line at a time, and do some string
        // substitution to make sure that we don't blow up our
        // SQL statement. ' characters aren't allowed...
        StringBuffer problemString = new StringBuffer();
        String line;
        while (true) {
            try {
                line = reader.readLine();
            } catch (Exception e) {
                line = null;
            }
            if (line == null)
                break;
            problemString.append(line.replaceAll("'", "\'"));
        }

        // close the problem file
        try {
            reader.close();
        } catch (Exception e) {
        }

        // return the string
        return problemString.toString();
    }

    private boolean isDirectory(String name) {
        File f = new File(name);
        return f.isDirectory();
    }

    private boolean fileExists() {
        boolean rc = false;
        try {
            FileInputStream fileIn = new FileInputStream(sourceFiles[0]);
            if (fileIn != null) {
                InputStreamReader inReader = new InputStreamReader(fileIn);
                if (inReader != null) {
                    BufferedReader br = new BufferedReader(inReader);
                    if (br != null) {
                        rc = true;
                        br.close();
                    }
                }
            }
        } catch (IOException e) {
            // do nothing but return false
        }
        return rc;
    }

    private String[] extractSourcePath() {
        //StringTokenizer st = new StringTokenizer(sourceFile, "/");
        File inFile = new File(sourceFiles[0]);
        String[] newPath = new String[2];
        newPath[0] = new String(inFile.getParent());
        if (newPath[0] != null) {
            newPath[1] = new String(inFile.getName());
        }
        return newPath;
    }

    //******************************
    //End - Helper functionProfiles for buildStatic data.
    //******************************

    static public void main(String[] args) {
        // 	for (int i=0; i<args.length; i++) {
        // 	    System.out.println ("args[" + i + "]: " + args[i]);
        // 	}

        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option experimentidOpt = parser.addStringOption('e', "experimentid");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");
        //CmdLineParser.Option problemOpt = parser.addStringOption('p',
        // "problemfile");
        CmdLineParser.Option trialOpt = parser.addStringOption('t', "trialid");
        CmdLineParser.Option typeOpt = parser.addStringOption('f', "filetype");
        CmdLineParser.Option fixOpt = parser.addBooleanOption('i', "fixnames");

        try {
            parser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.err.println(e.getMessage());
            LoadTrial.usage();
            System.exit(-1);
        }

        Boolean help = (Boolean) parser.getOptionValue(helpOpt);
        String configFile = (String) parser.getOptionValue(configfileOpt);
        //String sourceFile = (String)parser.getOptionValue(sourcefileOpt);
        String experimentID = (String) parser.getOptionValue(experimentidOpt);
        String trialName = (String) parser.getOptionValue(nameOpt);

        //String problemFile = (String)parser.getOptionValue(problemOpt);
        String trialID = (String) parser.getOptionValue(trialOpt);
        String fileTypeString = (String) parser.getOptionValue(typeOpt);
        Boolean fixNames = (Boolean) parser.getOptionValue(fixOpt);

        if (help != null && help.booleanValue()) {
            LoadTrial.outputHelp();
            System.exit(-1);
        }

        if (configFile == null) {
            System.err.println("Error: Missing config file (perfdmf_loadtrial should supply it)\n");
            LoadTrial.usage();
            System.exit(-1);
        } else if (trialName == null) {
            System.err.println("Error: Missing trial name\n");
            LoadTrial.usage();
            System.exit(-1);

            // 	} else if (sourceFile == null) {
            // 	    System.err.println("Please enter a valid source file.");
            // 	    System.err.println(LoadTrial.USAGE);
            // 	    System.exit(-1);
        } else if (experimentID == null) {
            System.err.println("Error: Missing experiment id\n");
            LoadTrial.usage();
            System.exit(-1);
        }

        String sourceFiles[] = parser.getRemainingArgs();

        int fileType = 1;
        String filePrefix = null;
        if (fileTypeString != null) {
            if (fileTypeString.equals("pprof")) {
                fileType = 0;
            } else if (fileTypeString.equals("profiles")) {
                fileType = 1;
            } else if (fileTypeString.equals("dynaprof")) {
                fileType = 2;
            } else if (fileTypeString.equals("mpip")) {
                fileType = 3;
            } else if (fileTypeString.equals("hpm")) {
                fileType = 4;
            } else if (fileTypeString.equals("gprof")) {
                fileType = 5;
            } else if (fileTypeString.equals("psrun")) {
                fileType = 6;
                /*
                 * } else if (fileTypeString.equals("sppm")) { fileType = 101; }
                 * else if (fileTypeString.equals("xprof")) { fileType = 0; }
                 * else if (fileTypeString.equals("sddf")) { fileType = 0;
                 */
            } else {
                System.err.println("Error: unknown type '" + fileTypeString + "'\n");
                LoadTrial.usage();
                System.exit(-1);
            }
        }

        if (trialName == null) {
            trialName = new String("");
        }

        if (fixNames == null) {
            fixNames = new Boolean(false);
        }

        LoadTrial trans = new LoadTrial(configFile, sourceFiles);
        trans.checkForExp(experimentID);
        if (trialID != null) {
            trans.checkForTrial(trialID);
            trans.trialID = Integer.parseInt(trialID);
        }
        trans.trialName = trialName;
        //trans.problemFile = problemFile;
        trans.fixNames = fixNames.booleanValue();
        trans.loadTrial(fileType);
        // the trial will be saved when the load is finished (update is called)
    }

}