package perfdb.loadxml;

import jargs.gnu.CmdLineParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.sql.SQLException;

import perfdb.util.dbinterface.DB;

public class LoadExperiment {
    private Load load = null;
    private DB db = null;
    
    private static String EXP_USAGE = 
        "USAGE: LoadExperiment [{-h,--help}] [{-a,--applicationid} value] [{-x,--xmlfile} filename]\n";

    private perfdb.ConnectionManager connector;

    public LoadExperiment(String configFileName) {
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

    /* Load environemnt information associated with an experiment*/
    public String storeExp(String appid, String expFile) {
		String expid = null;

		try {	
	    	expid = getLoad().parseExp(expFile, appid);
		} catch (Throwable ex) {
	    	System.out.println("Error: " + ex.getMessage());
	    	return null;
		}

		if ((expid==null) || (expid.trim().length()==0)) {
	    	System.out.println("Loadding experiment failed");
	    	return null;
		}
		return expid;
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option xmlfileOpt = parser.addStringOption('x', "xmlfile");
        CmdLineParser.Option applicationidOpt = parser.addStringOption('a', "applicationid");

        try {
            parser.parse(args);
        }
        catch ( CmdLineParser.OptionException e ) {
            System.err.println(e.getMessage());
	    	System.err.println(EXP_USAGE);
	    	System.exit(-1);
        }

        Boolean help = (Boolean)parser.getOptionValue(helpOpt);
        String configFile = (String)parser.getOptionValue(configfileOpt);
        String xmlFile = (String)parser.getOptionValue(xmlfileOpt);
        String applicationID = (String)parser.getOptionValue(applicationidOpt);

    	if (help != null && help.booleanValue()) {
	    	System.err.println(EXP_USAGE);
	    	System.exit(-1);
    	}

		if (configFile == null) {
            System.err.println("Please enter a valid config file.");
	    	System.err.println(EXP_USAGE);
	    	System.exit(-1);
		}

		// validate the command line options...
		if (applicationID == null) {
           	System.err.println("Please enter a valid application ID.");
	    	System.err.println(EXP_USAGE);
	    	System.exit(-1);
		}
		if (xmlFile == null) {
           	System.err.println("Please enter a valid experiment XML file.");
	    	System.err.println(EXP_USAGE);
	    	System.exit(-1);
		}

	// create a new LoadExperiment object, pass in the configuration file name
		LoadExperiment loadExp = new LoadExperiment(configFile);
		loadExp.getConnector().connect();
		int exitval = 0;
		String expid = loadExp.storeExp(applicationID, xmlFile);
		if (expid != null) 
			exitval = Integer.parseInt(expid);
		loadExp.getConnector().dbclose();
		System.exit(exitval);
    }

}

