package dms.perfdb;

import jargs.gnu.CmdLineParser;

import java.io.File;
import java.io.Serializable;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Date;
import java.util.Vector;

import paraprof.*;
/*
import paraprof.ParaProfTrial;
import paraprof.ParaProfDataSession;
import paraprof.UtilFncs;
*/
import dms.dss.*;
import java.awt.Component;

public class Translator implements ParaProfObserver {

    private File readPprof;
    private File writeXml;
    private String trialTime;
	private String sourceFile;
	private Application app;
	private Experiment exp;
	private int expID;

    /* This variable connects translator to DB in order to check whether
       the app. and exp. associated with the trial data do exist there. */
	PerfDBSession dbSession = null;
	ParaProfTrial trial = null;

    //constructor
    public Translator(String configFileName, String sourcename, String targetname) {
		this.sourceFile = sourcename;

		// check for the existence of file
		readPprof = new File(sourcename);

		// Get the creation time of pprof.dat
		Date date = new Date(readPprof.lastModified());
		trialTime = date.toString();

		if (readPprof.exists()){
	    	System.out.println("Found "+ sourcename + " ... Loading");
		}
		else {
			System.out.println("Did not find pprof.dat file!"); 
			System.exit(-1);
		}

		// check if the XML file exists
		writeXml = new File(targetname);
	
   		if (!writeXml.exists()){
	    	try {
				if (writeXml.createNewFile()){
		    		System.out.println("Create pprof.xml!");
				}
	    	}
	    	catch(Exception e){
				e.printStackTrace();	
	    	}
		}
	
		dbSession = new PerfDBSession();
		dbSession.initialize(configFileName);
    }

    public boolean checkForApp(String appid) {
		app = dbSession.setApplication(Integer.parseInt(appid));
		if (app == null)
			return false;
		else
			return true;
    }

    public boolean checkForExp(String expid) {
		this.expID = Integer.parseInt(expid);
		exp = dbSession.setExperiment(Integer.parseInt(expid));
		if (exp == null)
			return false;
		else
			return true;
    }

	public void loadTrial(String trialName, String problemFile) {
	    trial = null;
	    Vector v = new Vector();;
		File[] inFile = new File[1];
		inFile[0] = new File (sourceFile);
		v.add(inFile);

	    trial = new ParaProfTrial(null, 0);
		trial.addObserver(this);
	    trial.setName(trialName);
	    trial.setDefaultTrial(true);
	    trial.setPaths(System.getProperty("user.dir"));
	    trial.setLoading(true);
	    trial.initialize(v);

		// register self as a listener to the Trial...

		// finish setting up the trial
		trial.setProblemDefinition(getProblemString(problemFile));
		trial.setExperimentID(expID);
	}

	public void writeTrial() {
		XMLSupport xmlWriter = new XMLSupport(trial);
		xmlWriter.writeXmlFiles(0, writeXml);
	}

	public void saveTrial() {
		DataSession session = trial.getParaProfDataSession();
		dms.dss.Metric metric = (dms.dss.Metric)session.getMetrics().elementAt(0);
		trial.addMetric(metric);
		session.setTrial(trial);
		// set some things in the trial
		int[] maxNCT = trial.getMaxNCTNumbers();
		trial.setNodeCount(maxNCT[0]+1);
		trial.setNumContextsPerNode(maxNCT[1]+1);
		trial.setNumThreadsPerContext(maxNCT[2]+1);
		dbSession.saveParaProfTrial(session, trial);
		System.out.println("Done saving trial!");
	}

	public String getProblemString(String problemFile) {
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

	public void update (Object obj) {
		saveTrial();
	}

	public void update () {
		saveTrial();
	}

	//******************************
	//End - Helper functions for buildStatic data.
	//******************************

    static public void main(String[] args){
		// String USAGE = "USAGE: perfdb_translate [{-s,--sourcefile} sourcefilename] [{-d,destinationfile} destinationname] [{-a,--applicationid} application_id] [{-e,--experimentid} experiment_id] [{-n,--name} trial_name] [{-p,--problemfile} problem_file]";
		String USAGE = "USAGE: perfdb_loadtrial [{-s,--sourcefile} sourcefilename] [{-a,--applicationid} application_id] [{-e,--experimentid} experiment_id] [{-n,--name} trial_name] [{-p,--problemfile} problem_file]";

        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option sourcefileOpt = parser.addStringOption('s', "sourcefile");
        CmdLineParser.Option destinationfileOpt = parser.addStringOption('d', "destinationfile");
        CmdLineParser.Option experimentidOpt = parser.addStringOption('e', "experimentid");
        CmdLineParser.Option applicationidOpt = parser.addStringOption('a', "applicationid");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");
        CmdLineParser.Option problemOpt = parser.addStringOption('p', "problemfile");

        try {
            parser.parse(args);
        }
        catch ( CmdLineParser.OptionException e ) {
            System.err.println(e.getMessage());
	    	System.err.println(USAGE);
	    	System.exit(-1);
        }

        Boolean help = (Boolean)parser.getOptionValue(helpOpt);
        String configFile = (String)parser.getOptionValue(configfileOpt);
        String sourceFile = (String)parser.getOptionValue(sourcefileOpt);
        String destinationFile = (String)parser.getOptionValue(destinationfileOpt);
        String applicationID = (String)parser.getOptionValue(applicationidOpt);
        String experimentID = (String)parser.getOptionValue(experimentidOpt);
        String trialName = (String)parser.getOptionValue(nameOpt);
        String problemFile = (String)parser.getOptionValue(problemOpt);

    	if (help != null && help.booleanValue()) {
			System.err.println(USAGE);
	    	System.exit(-1);
    	}

		if (configFile == null) {
            System.err.println("Please enter a valid config file.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
		} else if (sourceFile == null) {
            System.err.println("Please enter a valid source file.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
		} else if (destinationFile == null) {
            System.err.println("Please enter a valid destination file.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
		} else if (applicationID == null) {
            System.err.println("Please enter a valid application ID.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
		} else if (experimentID == null) {
            System.err.println("Please enter a valid experiment ID.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
		}
		if (trialName == null) {
			trialName = new String("");
		}

		Translator trans = new Translator(configFile, sourceFile, destinationFile);
		trans.checkForApp(applicationID);
		trans.checkForExp(experimentID);
		trans.loadTrial(trialName, problemFile);
		// trans.saveTrial();
		// trans.writeTrial();
    }
} 
