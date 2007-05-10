package edu.uoregon.tau.perfdmf.loader;

import jargs.gnu.CmdLineParser;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DatabaseAPI;

public class CreateApplication {

    private static String APP_USAGE = "usage: perfdmf_createapp [{-h,--help}] {-n,--name} name\n";

    private DatabaseAPI session;

    public CreateApplication(String configFileName) {
        super();
        session = new DatabaseAPI();
        try {
            session.initialize(configFileName, true);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    /*** Parse and load an application. ***/

    public int createApp(String name) {
        int appid = -1;
        Application app = new Application();
        app.setName(name);
        //app.setVersion("");
        //app.setLanguage("");
        session.setApplication(app);
        appid = session.saveApplication();
        System.out.println("Created Application, ID: " + appid);
        session.terminate();
        return appid;
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {

        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('c', "config");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");

        try {
            parser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.err.println(e.getMessage());
            System.err.println(APP_USAGE);
            System.exit(-1);
        }

        Boolean help = (Boolean) parser.getOptionValue(helpOpt);
        String configFile = (String) parser.getOptionValue(configfileOpt);
        String name = (String) parser.getOptionValue(nameOpt);

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
            //System.err.println("Please enter a valid application name.");
            System.err.println(APP_USAGE);
            System.exit(-1);
        }

        // create a new CreateApplication object, pass in the configuration file name
        configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg." + configFile;
        
        CreateApplication create = new CreateApplication(configFile);

        int exitval = 0;

        /***** Load appliation into PerfDMF *********/
        int appid = create.createApp(name);

        System.exit(exitval);
    }
}
