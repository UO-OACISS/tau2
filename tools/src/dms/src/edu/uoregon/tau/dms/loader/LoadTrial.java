package edu.uoregon.tau.dms.loader;

import edu.uoregon.tau.dms.database.*;
import edu.uoregon.tau.dms.dss.*;
import jargs.gnu.CmdLineParser;
import java.io.File;
import java.io.Serializable;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Date;
import java.util.Vector;
import java.awt.Component;
import java.io.*;

public class LoadTrial{

    public static String USAGE = "usage: perfdmf_loadtrial {-f, --filetype} file_type {-s,--sourcefile} sourcefilename\n {-e,--experimentid} experiment_id [{-t, --trialid} trial_id] [{-n,--name} trial_name]\n [{-p,--problemfile} problem_file] [{-i --fixnames}]\n Where:\n    file_type = profiles (TAU), pprof (TAU), dynaprof, mpip, gprof, psrun, sddf (svpablo)\n\n (example: perfdmf_loadtrial -f profiles -e 12 -s \"profile.*\")";
    private File readPprof;
    private File writeXml;
    private String trialTime;
    private String sourceFile;
    private Application app;
    private Experiment exp;
    private boolean fixNames = false;
    private int expID = 0;
    public int trialID = 0;
    private int fileType = 0;
    private ParaProfDataSession dataSession = null;
    public String trialName = new String();
    public String problemFile = new String();

    /* This variable connects translator to DB in order to check whether
       the app. and exp. associated with the trial data do exist there. */
    PerfDMFSession dbSession = null;
    Trial trial = null;

    //constructor
    public LoadTrial(String configFileName, String sourcename) {
	this.sourceFile = sourcename;

	// check for the existence of file
	readPprof = new File(sourcename);

	dbSession = new PerfDMFSession();
	dbSession.initialize(configFileName);
    }

    public boolean checkForExp(String expid) {
	this.expID = Integer.parseInt(expid);
	exp = dbSession.setExperiment(this.expID);
	if (exp == null) {
	    System.err.println("Please enter a valid experiment ID.");
	    System.err.println(USAGE);
	    System.exit(-1);
	    return false;
	} else
	    return true;
    }

    public boolean checkForTrial(String trialid) {
	Trial tmpTrial = dbSession.setTrial(Integer.parseInt(trialid));
	if (tmpTrial == null)
	    return false;
	else
	    return true;
    }

    public void loadTrial(int fileType) {
	trial = null;
	this.fileType = fileType;

	Vector v = null;
	File[] inFile = new File[1];
	FileList fl = null;
	switch (fileType) {
	case 0:
	    inFile[0] = new File (sourceFile);
	    v = new Vector();
	    v.add(inFile);
	    dataSession = new TauPprofOutputSession();
	    break;
	case 1:
	    fl = new FileList();
	    v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, "profile", false);
	    dataSession = new TauOutputSession();
	    break;
	case 2:
	    inFile[0] = new File (sourceFile);
	    v = new Vector();
	    v.add(inFile);
	    dataSession = new DynaprofOutputSession();
	    break;
	case 3:
	    inFile[0] = new File (sourceFile);
	    v = new Vector();
	    v.add(inFile);
	    dataSession = new MpiPOutputSession();
	    break;
	case 4:
	    if (fileExists()) {
		inFile[0] = new File (sourceFile);
		v = new Vector();
		v.add(inFile);
	    } else {
		fl = new FileList();
		String[] sourcePath = extractSourcePath();
		if (sourcePath[0] != null)
		    v = fl.getFileList(new File(sourcePath[0]), null, fileType, sourcePath[1], false);
		else
		    v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, sourceFile, false);
	    }
	    dataSession = new HPMToolkitDataSession();
	    break;
	case 5:
	    if (fileExists()) {
		inFile[0] = new File (sourceFile);
		v = new Vector();
		v.add(inFile);
	    } else {
		fl = new FileList();
		String[] sourcePath = extractSourcePath();
		if (sourcePath[0] != null)
		    v = fl.getFileList(new File(sourcePath[0]), null, fileType, sourcePath[1], false);
		else
		    v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, sourceFile, false);
	    }
	    dataSession = new GprofOutputSession(fixNames);
	    break;
	case 6:
	    if (fileExists()) {
		inFile[0] = new File (sourceFile);
		v = new Vector();
		v.add(inFile);
	    } else {
		fl = new FileList();
		String[] sourcePath = extractSourcePath();
		if (sourcePath[0] != null)
		    v = fl.getFileList(new File(sourcePath[0]), null, fileType, sourcePath[1], false);
		else
		    v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, sourceFile, false);
	    }
	    dataSession = new PSRunDataSession();
	    break;
	    /*
	      case 101:
	      if (fileExists()) {
	      inFile[0] = new File (sourceFile);
	      v = new Vector();
	      v.add(inFile);
	      } else {
	      fl = new FileList();
	      String[] sourcePath = extractSourcePath();
	      if (sourcePath[0] != null)
	      v = fl.getFileList(new File(sourcePath[0]), null, fileType, sourcePath[1], false);
	      else
	      v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, sourceFile, false);
	      }
	      dataSession = new SPPMOutputDataSession();
	      break;
	    */
	default:
	    break;
	}

	trial = new Trial();
	trial.setDataSession(dataSession);
	dataSession.initialize(v);
	if(trialID==0)
	    saveTrial();
	else
	    appendToTrial();
    }

    public void writeTrial() {
	XMLSupport xmlWriter = new XMLSupport(trial);
	xmlWriter.writeXmlFiles(0, writeXml);
    }

    public void saveTrial() {
	// if (fileType == 101) return;
	// set some things in the trial
	int[] maxNCT = dataSession.getMaxNCTNumbers();
	trial.setNodeCount(maxNCT[0]+1);
	trial.setNumContextsPerNode(maxNCT[1]+1);
	trial.setNumThreadsPerContext(maxNCT[2]+1);
	trial.setName(trialName);
	trial.setProblemDefinition(getProblemString());
	trial.setExperimentID(expID);
	dbSession.saveParaProfTrial(trial, -1);
	System.out.println("Done saving trial!");
    }

    public void appendToTrial() {
	// set some things in the trial
	trial.setID(this.trialID);
	dbSession.saveParaProfTrial(trial, 0);
	System.out.println("Done adding metric to trial!");
    }

    public String getProblemString() {
	// if the file wasn't passed in, this is an existing trial.
	if (problemFile == null)
	    return new String("");

	// open the file
	BufferedReader reader = null;
	try {
	    reader = new BufferedReader (new FileReader (problemFile));
	} catch (Exception e) {
	    System.out.println("Problem file not found!  Exiting...");
	    System.exit(0);
	}
	// read the file, one line at a time, and do some string
	// substitution to make sure that we don't blow up our
	// SQL statement.  ' characters aren't allowed...
	StringBuffer problemString = new StringBuffer();
	String line;
	while (true) {
	    try {
		line = reader.readLine();
	    } catch (Exception e) {
		line = null;
	    }
	    if (line == null) break;
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

    private boolean fileExists() {
	boolean rc = false;
	try {
	    FileInputStream fileIn = new FileInputStream(sourceFile);
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
	File inFile = new File(sourceFile);
	String[] newPath = new String[2];
	newPath[0] = new String(inFile.getParent());
	if (newPath[0] != null) {
	    newPath[1] = new String(inFile.getName());
	}
	return newPath;
    }

    //******************************
    //End - Helper functions for buildStatic data.
    //******************************

    static public void main(String[] args){

        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option sourcefileOpt = parser.addStringOption('s', "sourcefile");
        CmdLineParser.Option experimentidOpt = parser.addStringOption('e', "experimentid");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");
        CmdLineParser.Option problemOpt = parser.addStringOption('p', "problemfile");
        CmdLineParser.Option trialOpt = parser.addStringOption('t', "trialid");
        CmdLineParser.Option typeOpt = parser.addStringOption('f', "filetype");
        CmdLineParser.Option fixOpt = parser.addBooleanOption('i', "fixnames");

        try {
            parser.parse(args);
        }
        catch ( CmdLineParser.OptionException e ) {
            System.err.println(e.getMessage());
	    System.err.println(LoadTrial.USAGE);
	    System.exit(-1);
        }

        Boolean help = (Boolean)parser.getOptionValue(helpOpt);
        String configFile = (String)parser.getOptionValue(configfileOpt);
        String sourceFile = (String)parser.getOptionValue(sourcefileOpt);
        String experimentID = (String)parser.getOptionValue(experimentidOpt);
        String trialName = (String)parser.getOptionValue(nameOpt);
        String problemFile = (String)parser.getOptionValue(problemOpt);
        String trialID = (String)parser.getOptionValue(trialOpt);
        String fileTypeString = (String)parser.getOptionValue(typeOpt);
        Boolean fixNames = (Boolean)parser.getOptionValue(fixOpt);

    	if (help != null && help.booleanValue()) {
	    System.err.println(LoadTrial.USAGE);
	    System.exit(-1);
    	}

	if (configFile == null) {
	    System.err.println("Please enter a valid config file.");
	    System.err.println(LoadTrial.USAGE);
	    System.exit(-1);
	} else if (sourceFile == null) {
	    System.err.println("Please enter a valid source file.");
	    System.err.println(LoadTrial.USAGE);
	    System.exit(-1);
	} else if (experimentID == null) {
	    System.err.println("Please enter a valid experiment ID.");
	    System.err.println(LoadTrial.USAGE);
	    System.exit(-1);
	} 
	
	int fileType = 0;
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
		  } else if (fileTypeString.equals("sppm")) {
		  fileType = 101;
		  } else if (fileTypeString.equals("xprof")) {
		  fileType = 0;
		  } else if (fileTypeString.equals("sddf")) {
		  fileType = 0;
		*/
	    } else {
		System.err.println("Please enter a valid file type.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
	    }
	}

	if (trialName == null) {
	    trialName = new String("");
	}

	if (fixNames == null) {
	    fixNames = new Boolean(false);
	}

	LoadTrial trans = new LoadTrial(configFile, sourceFile);
	trans.checkForExp(experimentID);
	if (trialID != null) {
	    trans.checkForTrial(trialID);
	    trans.trialID = Integer.parseInt(trialID);
	}
	trans.trialName = trialName;
	trans.problemFile = problemFile;
	trans.fixNames = fixNames.booleanValue();
	trans.loadTrial(fileType);
	// the trial will be saved when the load is finished (update is called)
    }
} 
