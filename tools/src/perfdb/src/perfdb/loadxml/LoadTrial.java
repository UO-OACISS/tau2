package perfdb.loadxml;

import jargs.gnu.CmdLineParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.sql.SQLException;

import perfdb.util.dbinterface.DB;

public class LoadTrial {
    private Load load = null;
    private DB db = null;
    
    private static String TRIAL_USAGE = 
        "USAGE: perfdb_loadtrial [{-h,--help}] [{-x,--xmlfile} filename] [{-t,--trialid] trial id] [{-p --problemfile} filename]\n";

    private perfdb.ConnectionManager connector;

    public LoadTrial(String configFileName) {
		super();
		connector = new perfdb.ConnectionManager(configFileName);
    }

    public perfdb.ConnectionManager getConnector(){
		return connector;
    }

    public Load getLoad() {
		if (load == null) {
	    	if (connector.getDB() == null) {
				load = new Load(connector.getParserClass());
	    	} else {
				load = new Load(connector.getDB(), connector.getParserClass());
	    	}
		}
		return load;
    }

    /*** Store a xml document for a trial ***/

    public String storeDocument(String xmlFile, String trialId, String problemFile, boolean bulkLoad) {
		if (trialId.compareTo("0") != 0) {
			String trialIdOut = getLoad().lookupTrial("trial", trialId);
			if (trialIdOut==null){
		    	System.out.println("The trial " + trialId + " was not found.");
		    	System.exit(-1);
			}    
		}

		try {
	    	trialId = getLoad().parse(xmlFile, trialId, problemFile, bulkLoad);
		} catch (Throwable ex) {
	    	System.out.println("Error: " + ex.getMessage());
		}

		if (trialId != null) {
	    	System.out.println("Loaded " + xmlFile + ", the trial id is: " + trialId);
		} else {
	    	System.out.println("Was unable to load document from " + xmlFile);
		}

		return trialId;
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option xmlfileOpt = parser.addStringOption('x', "xmlfile");
        CmdLineParser.Option trialidOpt = parser.addStringOption('t', "trialid");
        CmdLineParser.Option problemfileOpt = parser.addStringOption('p', "problemfile");
        CmdLineParser.Option bulkOpt = parser.addBooleanOption('b', "bulk");

        try {
            parser.parse(args);
        }
        catch ( CmdLineParser.OptionException e ) {
            System.err.println(e.getMessage());
	    	System.err.println(TRIAL_USAGE);
	    	System.exit(-1);
        }

        Boolean help = (Boolean)parser.getOptionValue(helpOpt);
        String configFile = (String)parser.getOptionValue(configfileOpt);
        String xmlFile = (String)parser.getOptionValue(xmlfileOpt);
        String trialID = (String)parser.getOptionValue(trialidOpt);
        String problemFile = (String)parser.getOptionValue(problemfileOpt);
        Boolean bulk = (Boolean)parser.getOptionValue(bulkOpt);

    	if (help != null && help.booleanValue()) {
	    	System.err.println(TRIAL_USAGE);
	    	System.exit(-1);
    	}

		if (configFile == null) {
            System.err.println("Please enter a valid config file.");
	    	System.err.println(TRIAL_USAGE);
	    	System.exit(-1);
		}

		if (trialID == null) {
			trialID = new String("0");
		}
		if (xmlFile == null) {
           	System.err.println("Please enter a valid trial XML file.");
	   		System.err.println(TRIAL_USAGE);
	   		System.exit(-1);
		}
		
	// create a new LoadTrial object, pass in the configuration file name
		LoadTrial loadTrial = new LoadTrial(configFile);
		loadTrial.getConnector().connect();

		int exitval = 0;
	
    	/***** Load a trial into PerfDB *********/
		String trialid = loadTrial.storeDocument(xmlFile, trialID, problemFile, (bulk != null && bulk.booleanValue()));
		if (trialid != null)
			exitval = Integer.parseInt(trialid);

		loadTrial.getConnector().dbclose();
		System.exit(exitval);
    }

}

