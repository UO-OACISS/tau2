package edu.uoregon.tau.perfdmf.loader;

import jargs.gnu.CmdLineParser;
import edu.uoregon.tau.perfdmf.database.ConnectionManager;
import edu.uoregon.tau.perfdmf.database.DB;

public class LoadSchema {
    private Load load = null;
    private DB db = null;
    
    private static String SCHEMA_USAGE = 
        "usage: perfdmf_loadschema [{-h,--help}] {-s,--schemafile} [{-g, --configfile} configFile] -c configuration filename\n";

    private ConnectionManager connector;

    public LoadSchema(String configFileName) {
	super();
	connector = new ConnectionManager(configFileName);
    }

    public ConnectionManager getConnector(){
	return connector;
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configFile");
        CmdLineParser.Option configOpt = parser.addStringOption('c', "config");
        CmdLineParser.Option schemafileOpt = parser.addStringOption('s', "schemafile");

        try {
            parser.parse(args);
        }
        catch ( CmdLineParser.OptionException e ) {
            System.err.println(e.getMessage());
	    System.err.println(SCHEMA_USAGE);
	    System.exit(-1);
        }

        Boolean help = (Boolean)parser.getOptionValue(helpOpt);
        String configFile = (String)parser.getOptionValue(configfileOpt);
        String configName = (String)parser.getOptionValue(configOpt);
        String schemaFile = (String)parser.getOptionValue(schemafileOpt);
            	if (help != null && help.booleanValue()) {
	    System.err.println(SCHEMA_USAGE);
	    System.exit(-1);
    	}

        if (configFile == null) {
        	if (configName == null)
        		configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg";
            else
              	configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg." + configName;
        }

	// validate the command line options...
	if (schemaFile == null) {
	    System.err.println("Please enter a valid schema file.");
	    System.err.println(SCHEMA_USAGE);
	    System.exit(-1);
	}

	// create a new LoadSchema object, pass in the configuration file name
	configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg." + configFile;
	LoadSchema loadSchema = new LoadSchema(configFile);
	try {
	    loadSchema.getConnector().connect();
	} catch (Exception e) {
	    e.printStackTrace();
	    System.exit(0);
	}

	int exitval = 0;
	
	/*** Load database schema to establish PerfDMF, invoke at most one time. ****/
	loadSchema.getConnector().genParentSchema(schemaFile);
	loadSchema.getConnector().dbclose();
	System.exit(exitval);
    }
}

