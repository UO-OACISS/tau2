package edu.uoregon.tau.dms.loader;

import edu.uoregon.tau.dms.database.*;
import edu.uoregon.tau.dms.dss.*;
import jargs.gnu.CmdLineParser;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.sql.SQLException;

public class CreateExperiment {
    
    private static String APP_USAGE = 
        "USAGE: perfdmf_loadapp [{-h,--help}] {-a,--applicationid} applicationID {-n,--name} name\n";

    private PerfDMFSession session;

    public CreateExperiment(String configFileName) {
	super();
	session = new PerfDMFSession();
	session.initialize(configFileName);
    }

    /*** Parse and load an experiment. ***/   

    public int createExp(String name, int appid) {
	int expid = 0;
	Experiment exp = new Experiment();
	exp.setName(name);
	exp.setApplicationID(appid);
	session.setExperiment(exp);
	expid = session.saveExperiment();
	System.out.println("Saved experiment, new ID: " + expid);
	session.terminate();
	return expid;
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");
        CmdLineParser.Option appidOpt = parser.addIntegerOption('a', "applicationid");

        try {
            parser.parse(args);
        }
        catch ( CmdLineParser.OptionException e ) {
            System.err.println(e.getMessage());
	    System.err.println(APP_USAGE);
	    System.exit(-1);
        }

        Boolean help = (Boolean)parser.getOptionValue(helpOpt);
        String configFile = (String)parser.getOptionValue(configfileOpt);
        String name = (String)parser.getOptionValue(nameOpt);
        int appid = ((Integer)parser.getOptionValue(appidOpt)).intValue();

    	if (help != null && help.booleanValue()) {
	    System.err.println(APP_USAGE);
	    System.exit(-1);
    	}

	if (configFile == null) {
            System.err.println("Please enter a valid config file.");
	    System.err.println(APP_USAGE);
	    System.exit(-1);
	}

	// validate the command line options...
	if (name == null) {
	    System.err.println("Please enter a valid experiment name.");
	    System.err.println(APP_USAGE);
	    System.exit(-1);
	}

	// create a new CreateExperiment object, pass in the configuration file name
	CreateExperiment create = new CreateExperiment(configFile);

	int exitval = 0;
	
    	/***** Load appliation into PerfDMF *********/
	int expid = create.createExp(name, appid);

	System.exit(exitval);
    }
}

