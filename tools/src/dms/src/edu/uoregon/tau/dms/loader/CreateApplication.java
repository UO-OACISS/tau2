package edu.uoregon.tau.dms.loader;

import edu.uoregon.tau.dms.database.*;
import edu.uoregon.tau.dms.dss.*;
import jargs.gnu.CmdLineParser;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.sql.SQLException;

public class CreateApplication {
    
    private static String APP_USAGE = 
        "USAGE: perfdmf_loadapp [{-h,--help}] [{-n,--name} application] \n";

    private PerfDMFSession session;

    public CreateApplication(String configFileName) {
		super();
		session = new PerfDMFSession();
		session.initialize(configFileName);
    }

    /*** Parse and load an application. ***/   

    public int createApp(String name) {
		int appid = 0;
		Application app = new Application();
		app.setName(name);
		app.setVersion("");
		app.setLanguage("");
		session.setApplication(app);
		appid = session.saveApplication();
		System.out.println("Saved application, new ID: " + appid);
		session.terminate();
		return appid;
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addStringOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");

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
           	System.err.println("Please enter a valid application name.");
    		System.err.println(APP_USAGE);
    		System.exit(-1);
		}

	// create a new CreateApplication object, pass in the configuration file name
		CreateApplication create = new CreateApplication(configFile);

		int exitval = 0;
	
    	/***** Load appliation into PerfDMF *********/
		int appid = create.createApp(name);

		System.exit(exitval);
    }
}

