package perfdb.loadxml;

import jargs.gnu.CmdLineParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.sql.SQLException;

import perfdb.util.dbinterface.DB;

public class Main {
    private Load load = null;
    private DB db = null;
    
    private static String USAGE = 
        "USAGE: Main [{-h,--help}] [{-g,--configfile} filename] \n"
		+ "    [{-c,--command} loadschema] [{-s,--schemafile} filename] \n"
		+ "  | [{-c,--command} loadapp] [{-x,--xmlfile} filename] \n"
		+ "  | [{-c,--command} loadexp] [{-a,--applicationid} value] [{-x,--xmlfile} filename] \n"
		+ "  | [{-c,--command} loadtrial] [{-x,--xmlfile} filename] [{-t,--trialid] trial id] \n";

    private perfdb.ConnectionManager connector;

    public Main(String configFileName) {
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

    public void errorPrint(String msg) {
	System.err.println(msg);
    }

    /*** Parse and load an application. ***/   

    public String storeApp(String appFile) {
		String appid = null;

		try {	
	    	appid = getLoad().parseApp(appFile);
		} catch (Throwable ex) {
	    	errorPrint("Error: " + ex.getMessage());
	    	return null;
		}

		if ((appid==null) || (appid.trim().length()==0)) {
	    	System.out.println("Loadding application failed");
	    	return null;
		}
		return appid;
    }

    /* Load environemnt information associated with an experiment*/
 
    public String storeExp(String appid, String expFile) {
		String expid = null;

		try {	
	    	expid = getLoad().parseExp(expFile, appid);
		} catch (Throwable ex) {
	    	errorPrint("Error: " + ex.getMessage());
	    	return null;
		}

		if ((expid==null) || (expid.trim().length()==0)) {
	    	System.out.println("Loadding experiment failed");
	    	return null;
		}
		return expid;
    }

    /*** Store a xml document for a trial ***/

    public String storeDocument(String xmlFile, String trialId) {
		if (trialId.compareTo("0") != 0) {
			String trialIdOut = getLoad().lookupTrial("trial", trialId);
			if (trialIdOut==null){
		    	System.out.println("The trial " + trialId + " was not found.");
		    	System.exit(-1);
			}    
		}

		try {
	    	trialId = getLoad().parse(xmlFile, trialId);		
		} catch (Throwable ex) {
	    	errorPrint("Error: " + ex.getMessage());
		}

		if (trialId != null) {
	    	System.out.println("Loaded " + xmlFile + ", the trial id is: " + trialId);
		} else {
	    	errorPrint("Was unable to load document from " + xmlFile);
		}

		return trialId;
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option commandOpt = parser.addStringOption('c', "command");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option xmlfileOpt = parser.addStringOption('x', "xmlfile");
        CmdLineParser.Option trialidOpt = parser.addStringOption('t', "trialid");
        CmdLineParser.Option applicationidOpt = parser.addStringOption('a', "applicationid");
        CmdLineParser.Option schemafileOpt = parser.addStringOption('s', "schemafile");

        try {
            parser.parse(args);
        }
        catch ( CmdLineParser.OptionException e ) {
            System.err.println(e.getMessage());
	    	System.err.println(USAGE);
	    	System.exit(-1);
        }

        Boolean help = (Boolean)parser.getOptionValue(helpOpt);
        String command = (String)parser.getOptionValue(commandOpt);
        String configFile = (String)parser.getOptionValue(configfileOpt);
        String xmlFile = (String)parser.getOptionValue(xmlfileOpt);
        String trialID = (String)parser.getOptionValue(trialidOpt);
        String applicationID = (String)parser.getOptionValue(applicationidOpt);
        String schemaFile = (String)parser.getOptionValue(schemafileOpt);

    	if (help != null && help.booleanValue()) {
			System.err.println(USAGE);
	    	System.exit(-1);
    	}

		if (command == null) {
            System.err.println("Please enter a valid command.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
		}

		if (configFile == null) {
            System.err.println("Please enter a valid config file.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
		}

	// create a new Main object, pass in the configuration file name
		Main demo = new Main(configFile);
		demo.getConnector().connect();

		int exitval = 0;
	
    	/***** Load database schema to establish PerfDB, invoke at most one time. ******/
		if (command.equalsIgnoreCase("LOADSCHEMA")) {
			demo.getConnector().genParentSchema(schemaFile);
    	}
    	/***** Load appliation into PerfDB *********/
		else if (command.equalsIgnoreCase("LOADAPP")) {
			String appid = demo.storeApp(xmlFile);
			if (appid != null)
				exitval = Integer.parseInt(appid);
    	}
    	/***** Load experiment into PerfDB ********/
		else if (command.equalsIgnoreCase("LOADEXP")) {
			String expid = demo.storeExp(applicationID, xmlFile);
			if (expid != null)
				exitval = Integer.parseInt(expid);
    	}
    	/***** Load a trial into PerfDB *********/
		else if (command.equalsIgnoreCase("LOADXML") || command.equalsIgnoreCase("LOADTRIAL")) {
			String trialid = demo.storeDocument(xmlFile, trialID);
			if (trialid != null)
				exitval = Integer.parseInt(trialid);
    	}

		demo.getConnector().dbclose();
		System.exit(exitval);
    }

}

