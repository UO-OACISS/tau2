package dms.perfdb;

import jargs.gnu.CmdLineParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.sql.SQLException;

public class LoadSchema {
    private Load load = null;
    private DB db = null;
    
    private static String SCHEMA_USAGE = 
        "USAGE: perfdb_loadschema [{-h,--help}] [{-s,--schemafile} filename]\n";

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
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
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
        String schemaFile = (String)parser.getOptionValue(schemafileOpt);

    	if (help != null && help.booleanValue()) {
	    	System.err.println(SCHEMA_USAGE);
	    	System.exit(-1);
    	}

		if (configFile == null) {
            System.err.println("Please enter a valid config file.");
	    	System.err.println(SCHEMA_USAGE);
	    	System.exit(-1);
		}

		// validate the command line options...
		if (schemaFile == null) {
           	System.err.println("Please enter a valid schema file.");
    		System.err.println(SCHEMA_USAGE);
    		System.exit(-1);
		}

		// create a new LoadSchema object, pass in the configuration file name
		LoadSchema loadSchema = new LoadSchema(configFile);
		try {
			loadSchema.getConnector().connect();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}

		int exitval = 0;
	
/*** Load database schema to establish PerfDB, invoke at most one time. ****/
		loadSchema.getConnector().genParentSchema(schemaFile);
		loadSchema.getConnector().dbclose();
		System.exit(exitval);
    }
}

