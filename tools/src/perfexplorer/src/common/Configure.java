package common;

import jargs.gnu.CmdLineParser;

import java.io.*;
import java.sql.ResultSet;
import java.sql.SQLException;

import edu.uoregon.tau.dms.database.ConnectionManager;
import edu.uoregon.tau.dms.database.DB;
import edu.uoregon.tau.dms.database.ParseConfig;

public class Configure {

    
    private static String Greeting = "\nNow testing your database connection.\n";

    private String tau_root = "";
    private String db_dbname = "perfdmf";
    private String db_password = "";
    private ParseConfig parser;

    private String configFileName;

    private static String Usage = "Usage: configure [{-h,--help}] [{-g,--configfile} filename] [{-t,--tauroot} path]";

    public Configure(String tauroot, String arch) {
        super();
        this.tau_root = tauroot;
    }

    public void initialize(String configFileNameIn) {
        // Welcome the user to the program
        System.out.println(Greeting);

        try {
            // Check to see if the configuration file exists
            configFileName = configFileNameIn;
            File configFile = new File(configFileName);
            if (configFile.exists()) {
                //System.out.println("Configuration file found...");
                // Parse the configuration file
                parseConfigFile();
            } else {
                System.out.println("Configuration file NOT found...");
                System.out.println("a new configuration file will be created.");
                // If it doesn't exist, explain that the program looks for the 
                // configuration file in ${PerfDMF_Home}/data/perfdmf.cfg
                // Since it isn't there, create a new one.
            }
        } catch (IOException e) {
            // todo - get info from the exception
            System.out.println("I/O Error occurred.");
        }
    }
    
    
    public void parseConfigFile() throws IOException, FileNotFoundException {
        //System.out.println("Parsing config file...");
        parser = new ParseConfig(configFileName);
        db_dbname = parser.getDBName();
        db_password = parser.getDBPasswd();
    }

    
    /* Test that the database exists, and if it doesn't, create it! */
    public void createDB() {
        ConnectionManager connector = null;
        DB db = null;
        try {
            if (db_password != null) {
                connector = new ConnectionManager(configFileName, db_password);
            } else {
                connector = new ConnectionManager(configFileName);
            }
            connector.connect();
            System.out.println();
            db = connector.getDB();
        } catch (Exception e) {
            System.out.println("\nPlease make sure that your DBMS is configured correctly, and");
            System.out.println("the database " + db_dbname + " has been created.");
            System.exit(0);
        }

        try {
            String query = new String("SELECT * FROM " + db.getSchemaPrefix() + "analysis_settings");
            ResultSet resultSet = db.executeQuery(query);
            resultSet.close();
        } catch (SQLException e) {
            // this is our method of determining that no 'application' table exists

            System.out.print("Perfexplorer tables not found.  Would you like to upload the schema? [y/n]: ");

            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

            String input = "";
            try {
                input = reader.readLine();
            } catch (java.io.IOException ioe) {
                ioe.printStackTrace();
                System.exit(-1);
            }
            if (input.equals("y") || input.equals("Y")) {

                String filename="";
                
                if (db.getDBType().compareTo("oracle") == 0) {
                    filename = tau_root + "/tools/src/perfexplorer/etc/dbschema.oracle";
                } else if (db.getDBType().compareTo("mysql") == 0) {
                    filename = tau_root + "/tools/src/perfexplorer/etc/dbschema.mysql";
                } else if (db.getDBType().compareTo("postgresql") == 0) {
                    filename = tau_root + "/tools/src/perfexplorer/etc/dbschema.postgresql";
                } else {
                    System.out.println("Unknown database type: " + db.getDBType());
                    System.exit(-1);
                }
                
                System.out.println("Uploading Schema: " + filename);
                if (connector.genParentSchema(filename) == 0) {
                    System.out.println("Successfully uploaded schema\n");
                } else {
                    System.out.println("Error uploading schema\n");
                    System.exit(-1);
                }
            }
        }

        try {
            if (db.checkSchema() != 0) {
                System.out.println("\nIncompatible schema found.  Please contact us at tau-team@cs.uoregon.edu\nfor a conversion script.");
                System.exit(0);
            }
        } catch (SQLException e) {
            System.out.println("\nError trying to confirm schema:");
            e.printStackTrace();
            System.exit(0);
        }

        connector.dbclose();

        System.out.println("Database connection successful.");
        System.out.println("Configuration complete.");
    }

    
    public static void main(String[] args) {

        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option homeOpt = parser.addStringOption('t', "tauroot");
        CmdLineParser.Option archOpt = parser.addStringOption('a', "arch");
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        try {
            parser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.err.println(e.getMessage());
            System.err.println(Usage);
            System.exit(-1);
        }

        String configFile = (String) parser.getOptionValue(configfileOpt);
        String tauroot = (String) parser.getOptionValue(homeOpt);
        String arch = (String) parser.getOptionValue(archOpt);
        Boolean help = (Boolean) parser.getOptionValue(helpOpt);

        if (help != null && help.booleanValue()) {
            System.err.println(Usage);
            System.exit(-1);
        }

        if (configFile == null)
            configFile = new String("");
        if (tauroot == null)
            tauroot = new String("");
        if (arch == null)
            arch = new String("");

        // Create a new Configure object, which will walk the user through
        // the process of creating/editing a configuration file.
        Configure config = new Configure(tauroot, arch);

        config.tau_root = tauroot;

        config.initialize(configFile);
        config.createDB();

    }

}
