package dms.perfdb;

import jargs.gnu.CmdLineParser;

import java.io.File;
import java.io.Serializable;
import java.util.Date;
import java.util.Vector;

import paraprof.ParaProfTrial;

public class Translator implements Serializable {

    private File readPprof;
    private File writeXml;
    private String trialTime;

    /* This variable connects translator to DB in order to check whether
       the app. and exp. associated with the trial data do exist there. */
    private ConnectionManager connector;

    //constructor
    public Translator(String configFileName, String sourcename, String targetname) {

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
	
		connector = new ConnectionManager(configFileName);
		try {
			connector.connect();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
    }

    public String checkForApp(String appid) {
		String returnVal = null;
	
		StringBuffer buf = new StringBuffer();
        buf.append("select a.id from application a ");
		buf.append("where a.id=" + appid + "; ");
		try {
			returnVal = connector.getDB().getDataItem(buf.toString());               
			if (returnVal == null) {
				System.out.println("Application ID: " + appid + " not found.");
			}
        } catch (Exception ex) {
			ex.printStackTrace();
        }
        return returnVal;
    }

    public String checkForExp(String expid, String appid) {
		String returnVal = null;
	
		StringBuffer buf = new StringBuffer();
		buf.append("select id from experiment ");
		buf.append("where id = " + expid + " and application = " + appid + "; ");
		try {
			returnVal = connector.getDB().getDataItem(buf.toString());
			if (returnVal == null) {
				System.out.println("Experiment ID: " + expid + " with Application ID: " + appid + " not found.");
			}
        }catch (Exception ex) {
			ex.printStackTrace();
        }
        
        return returnVal;
    }

	//******************************
	//End - Helper functions for buildStatic data.
	//******************************

    static public void main(String[] args){
		String USAGE = "USAGE: perfdb_translate [{-s,--sourcefile} sourcefilename] [{-d,destinationfile} destinationname] [{-a,--applicationid} application_id] [{-e,--experimentid} experiment_id] [{-n,--name} trial_name]";

        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option sourcefileOpt = parser.addStringOption('s', "sourcefile");
        CmdLineParser.Option destinationfileOpt = parser.addStringOption('d', "destinationfile");
        CmdLineParser.Option experimentidOpt = parser.addStringOption('e', "experimentid");
        CmdLineParser.Option applicationidOpt = parser.addStringOption('a', "applicationid");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");

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
		trans.checkForExp(experimentID, applicationID);

		/*
	    FileList fl = new FileList();
	    v = fl.getFileList(new File(System.getProperty("user.dir")), null, 0 , "pprof", false);
		if (v.size() == 0) {
	    	v = fl.getFileList(new File(System.getProperty("user.dir")), null, 0 , "profile", false);
		}
		*/

	    ParaProfTrial trial = null;
	    Vector v = new Vector();;
		File[] inFile = new File[1];
		inFile[0] = new File (sourceFile);
		v.add(inFile);

	    trial = new ParaProfTrial(null, 0);
	    trial.setName(trialName);
	    trial.setDefaultTrial(true);
	    trial.setPaths(System.getProperty("user.dir"));
	    trial.setLoading(true);
	    trial.initialize(v);

		System.out.println("Done - Translating pprof.dat into pprof.xml!");
    }
} 
