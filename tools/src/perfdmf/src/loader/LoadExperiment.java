package edu.uoregon.tau.perfdmf.loader;

import jargs.gnu.CmdLineParser;
import edu.uoregon.tau.perfdmf.database.ConnectionManager;
import edu.uoregon.tau.perfdmf.database.DB;

public class LoadExperiment {
    private Load load = null;
    private DB db = null;
    
    private static String EXP_USAGE = 
        "USAGE: LoadExperiment [{-h,--help}] -c configure {-a,--applicationid} applicationID {-x,--xmlfile} filename\n";

    private ConnectionManager connector;

    public LoadExperiment(String configFileName) {
	super();
	connector = new ConnectionManager(configFileName);
    }

    public ConnectionManager getConnector(){
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
        CmdLineParser.Option configOpt = parser.addStringOption('g', "configfile");
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
        String configName = (String) parser.getOptionValue(configOpt);
        String xmlFile = (String)parser.getOptionValue(xmlfileOpt);
        String applicationID = (String)parser.getOptionValue(applicationidOpt);

    	if (help != null && help.booleanValue()) {
	    System.err.println(EXP_USAGE);
	    System.exit(-1);
    	}

    	if (configFile == null) {
        	if (configName == null)
        		configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg";
        	else
        		configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg." + configName;
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
	try {
	    loadExp.getConnector().connect();
	} catch (Exception e) {
	    e.printStackTrace();
	    System.exit(0);
	}
	int exitval = 0;
	String expid = loadExp.storeExp(applicationID, xmlFile);
	if (expid != null) 
	    exitval = Integer.parseInt(expid);
	loadExp.getConnector().dbclose();
	// System.exit(exitval);
    }

}

