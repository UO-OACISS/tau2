package edu.uoregon.tau.perfdmf.loader;

import jargs.gnu.CmdLineParser;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Experiment;

public class CreateExperiment {

    private static String APP_USAGE = "USAGE: perfdmf_loadapp [{-h,--help}] {-a,--applicationid} applicationID {-n,--name} name\n";

    private DatabaseAPI session;

    public CreateExperiment(String configFileName) {
        super();
        session = new DatabaseAPI();
        try {
            session.initialize(configFileName, true);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    /*** Parse and load an experiment. ***/

    public boolean checkForApp(int appid) {
        Application app = session.setApplication(appid);
        if (app == null) {
            System.err.println("Application id " + appid + " not found,  please enter a valid application ID.");
            System.exit(-1);
            return false;
        } else
            return true;
    }

    public int createExp(String name, int appid) {
        int expid = 0;

        checkForApp(appid);

        Experiment exp = new Experiment();
        exp.setName(name);
        exp.setApplicationID(appid);
        session.setExperiment(exp);
        try {
            expid = session.saveExperiment();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Created Experiment, ID: " + expid);
        session.terminate();
        return expid;
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option configfileOpt = parser.addStringOption('c', "config");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");
        CmdLineParser.Option appidOpt = parser.addIntegerOption('a', "applicationid");

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
        Integer app = (Integer) parser.getOptionValue(appidOpt);

        if (help != null && help.booleanValue()) {
            System.err.println(APP_USAGE);
            System.exit(-1);
        }

        if (configFile == null) {
            System.err.println("Please enter a valid config file.");
            System.err.println(APP_USAGE);
            System.exit(-1);
        }

        if (app == null) {
            System.err.println("Please enter a valid application id.");
            System.err.println(APP_USAGE);
            System.exit(-1);
        }
        int appid = ((Integer) parser.getOptionValue(appidOpt)).intValue();

        // validate the command line options...
        if (name == null) {
            System.err.println("Please enter a valid experiment name.");
            System.err.println(APP_USAGE);
            System.exit(-1);
        }

        // create a new CreateExperiment object, pass in the configuration file name
        configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg." + configFile;
        CreateExperiment create = new CreateExperiment(configFile);

        int exitval = 0;

        /***** Load appliation into PerfDMF *********/
        int expid = create.createExp(name, appid);

        System.exit(exitval);
    }
}
