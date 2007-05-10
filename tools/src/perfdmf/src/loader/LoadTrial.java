package edu.uoregon.tau.perfdmf.loader;

import jargs.gnu.CmdLineParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import edu.uoregon.tau.perfdmf.*;

public class LoadTrial {

    private File writeXml;
    private String trialTime;
    private String sourceFiles[];
    private String metadataFile;
    private Application app;
    private Experiment exp;
    private boolean fixNames = false;
    private int expID = 0;
    public int trialID = 0;
    private int fileType = 0;
    private DataSource dataSource = null;
    public String trialName = new String();
    public String problemFile = new String();
    public String configuration;
    
    public static void usage() {
        System.err.println("Usage: perfdmf_loadtrial -c <config name> -e <experiment id> -n <name> [options] <files>\n\n"
                + "try `perfdmf_loadtrial --help' for more information");
    }

    public static void outputHelp() {

        System.err.println("Usage: perfdmf_loadtrial -c <config name> -e <experiment id> -n <name> [options] <files>\n\n"
                + "Required Arguments:\n\n"
                + "  -c, --config <file>             Specify the name of the configuration to use\n"
                + "  -n, --name <text>               Specify the name of the trial\n"
                + "  -e, --experimentid <number>     Specify associated experiment ID\n"
				+ "                                    for this trial\n"
				+ "               ...or...\n"
                + "  -n, --name <text>               Specify the name of the trial\n"
                + "  -a, --applicationname <string>  Specify associated application name\n"
				+ "                                    for this trial\n"
                + "  -x, --experimentname <string>   Specify associated experiment name\n"
				+ "                                    for this trial\n"
				+ "\n" + "Optional Arguments:\n\n"
                + "  -f, --filetype <filetype>       Specify type of performance data, options are:\n"
                + "                                    profiles (default), pprof, dynaprof, mpip,\n"
                + "                                    gprof, psrun, hpm, packed, cube, hpc\n"
                + "  -t, --trialid <number>          Specify trial ID\n"
                + "  -i, --fixnames                  Use the fixnames option for gprof\n\n"
                + "  -m, --metadata <filename>       XML metadata for the trial\n" + "Notes:\n"
                + "  For the TAU profiles type, you can specify either a specific set of profile\n"
                + "files on the commandline, or you can specify a directory (by default the current\n"
                + "directory).  The specified directory will be searched for profile.*.*.* files,\n"
                + "or, in the case of multiple counters, directories named MULTI_* containing\n" + "profile data.\n\n"
                + "Examples:\n\n" + "  perfdmf_loadtrial -e 12 -n \"Batch 001\"\n"
                + "    This will load profile.* (or multiple counters directories MULTI_*) into\n"
                + "    experiment 12 and give the trial the name \"Batch 001\"\n\n"
                + "  perfdmf_loadtrial -e 12 -n \"HPM data 01\" -f hpm perfhpm*\n"
                + "    This will load perfhpm* files of type HPMToolkit into experiment 12 and give\n"
                + "    the trial the name \"HPM data 01\"\n"
                + "  perfdmf_loadtrial -an \"NPB2.3\" -en \"parametric\" -n \"64\"\n"
                + "    This will load profile.* (or multiple counters directories MULTI_*) into\n"
				+ "    the experiment named \"parametric\" under the application named \"NPB2.3\" \n"
				+ "    and give the trial the name \"64\"\n");
    }

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
            //System.out.println(configFileName + ", " + configuration);
        	databaseAPI.initialize(configFileName, true);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

    }

    public boolean checkForExp(String expid, String appName, String expName) {
		if (expid != null) {
        	this.expID = Integer.parseInt(expid);

        	try {
            	exp = databaseAPI.setExperiment(this.expID);
        	} catch (Exception e) {
        	}
        	if (exp == null) {
            	System.err.println("Experiment id " + expid + 
						" not found,  please enter a valid experiment ID.");
            	System.exit(-1);
            	return false;
        	} else
            	return true;
		} else {
			Experiment exp = databaseAPI.getExperiment (appName, expName, true);
			this.expID = exp.getID();
			return true;
		}
    }

    public boolean checkForTrial(String trialid) {
        Trial tmpTrial = databaseAPI.setTrial(Integer.parseInt(trialid));
        if (tmpTrial == null)
            return false;
        else
            return true;
    }

    public void loadTrial(int fileType) {

        File[] files = new File[sourceFiles.length];
        for (int i = 0; i < sourceFiles.length; i++) {
            files[i] = new File(sourceFiles[i]);
        }

        try {
            dataSource = UtilFncs.initializeDataSource(files, fileType, fixNames);
        } catch (DataSourceException e) {

            if (files == null || files.length != 0) // We don't output an error message if paraprof was just invoked with no parameters.
                e.printStackTrace();
            return;
        }

        trial = null;
        this.fileType = fileType;

        trial = new Trial();
        trial.setDataSource(dataSource);

		// set the metadata file name before loading the data, because
		// aggregateData() is called at the end of the dataSource.load()
		// and this file has to be set before then.
        try {
            if (metadataFile != null) {
                dataSource.setMetadataFile(metadataFile);
            }
        } catch (Exception e) {
            System.err.println("Error Loading metadata:");
            e.printStackTrace();
            System.exit(1);
        }

        try {
            dataSource.load();
        } catch (Exception e) {
            System.err.println("Error Loading Trial:");
            e.printStackTrace();
        }

        // set the meta data from the datasource
        trial.setMetaData(dataSource.getMetaData());
        
        if (trialID == 0) {
            saveTrial();
        } else {
            appendToTrial();
        }

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
            //databaseAPI.saveTrial(trial, -1);
            databaseAPI.uploadTrial(trial);
        } catch (DatabaseException e) {
            e.printStackTrace();
            Exception e2 = e.getException();
            System.out.println("from: ");
            e2.printStackTrace();
            System.exit(-1);
        }
        System.out.println("Done saving trial!");
    }

    public void appendToTrial() {
        // set some things in the trial
        trial.setID(this.trialID);
        try {
            databaseAPI.saveTrial(trial, 0);
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
        } catch (Exception e) {}

        // return the string
        return problemString.toString();
    }

    static public void main(String[] args) {
        // 	for (int i=0; i<args.length; i++) {
        // 	    System.out.println ("args[" + i + "]: " + args[i]);
        // 	}

        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option experimentidOpt = parser.addStringOption('e', "experimentid");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");
        //CmdLineParser.Option problemOpt = parser.addStringOption('p',
        // "problemfile");
        //CmdLineParser.Option gopt = parser.addStringOption('g', "g");
        CmdLineParser.Option configOpt = parser.addStringOption('c', "config");
        CmdLineParser.Option trialOpt = parser.addStringOption('t', "trialid");
        CmdLineParser.Option typeOpt = parser.addStringOption('f', "filetype");
        CmdLineParser.Option fixOpt = parser.addBooleanOption('i', "fixnames");
        CmdLineParser.Option metadataOpt = parser.addStringOption('m', "metadata");
        CmdLineParser.Option appNameOpt = parser.addStringOption('a', "applicationname");
        CmdLineParser.Option expNameOpt = parser.addStringOption('x', "experimentname");

        try {
            parser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.err.println(e.getMessage());
            LoadTrial.usage();
            System.exit(-1);
        }

        Boolean help = (Boolean) parser.getOptionValue(helpOpt);
        //String sourceFile = (String)parser.getOptionValue(sourcefileOpt);
        String configFile = (String) parser.getOptionValue(configOpt);
        String experimentID = (String) parser.getOptionValue(experimentidOpt);
        String trialName = (String) parser.getOptionValue(nameOpt);
        String appName = (String) parser.getOptionValue(appNameOpt);
        String expName = (String) parser.getOptionValue(expNameOpt);

        //String problemFile = (String)parser.getOptionValue(problemOpt);
        String trialID = (String) parser.getOptionValue(trialOpt);
        String fileTypeString = (String) parser.getOptionValue(typeOpt);
        Boolean fixNames = (Boolean) parser.getOptionValue(fixOpt);
        String metadataFile = (String) parser.getOptionValue(metadataOpt);

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
            //        } else if (sourceFiles == null) {
            //            System.err.println("Please enter a valid source file.");
            //            LoadTrial.usage();
            //            System.exit(-1);
        } else if (experimentID == null && expName == null) {
            System.err.println("Error: Missing experiment id or name\n");
            LoadTrial.usage();
            System.exit(-1);
        } else if (expName != null && appName == null) {
            System.err.println("Error: Missing application name\n");
            LoadTrial.usage();
            System.exit(-1);
        }

        String sourceFiles[] = parser.getRemainingArgs();

        int fileType = 0;
        String filePrefix = null;
        if (fileTypeString != null) {
            if (fileTypeString.equals("profiles")) {
                fileType = 0;
            } else if (fileTypeString.equals("pprof")) {
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
            } else if (fileTypeString.equals("packed")) {
                fileType = 7;
            } else if (fileTypeString.equals("cube")) {
                fileType = 8;
            } else if (fileTypeString.equals("hpc")) {
                fileType = 9;
            } else if (fileTypeString.equals("gyro")) {
                fileType = 100;
            } else {
                System.err.println("Please enter a valid file type.");
                LoadTrial.usage();
                System.exit(-1);
            }
        } else {
            if (sourceFiles.length == 1) {
                String filename = sourceFiles[0];
                if (filename.toLowerCase().endsWith(".ppk")) {
                    fileType = 7;
                }
                if (filename.toLowerCase().endsWith(".cube")) {
                    fileType = 8;
                }
                if (filename.toLowerCase().endsWith(".mpip")) {
                    fileType = 3;
                }
            }
        }

        if (trialName == null) {
            trialName = new String("");
        }

        if (fixNames == null) {
            fixNames = new Boolean(false);
        }
        configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg." + configFile;
        //System.out.println(System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg." + configFile);
        LoadTrial trans = new LoadTrial(configFile, sourceFiles);
        trans.checkForExp(experimentID, appName, expName);
        if (trialID != null) {
            trans.checkForTrial(trialID);
            trans.trialID = Integer.parseInt(trialID);
        }
        trans.trialName = trialName;
        //trans.problemFile = problemFile;
        trans.fixNames = fixNames.booleanValue();
        trans.metadataFile = metadataFile;
        trans.loadTrial(fileType);
        // the trial will be saved when the load is finished (update is called)
    }

}
