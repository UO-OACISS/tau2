package edu.uoregon.tau.perfdmf.loader;

import jargs.gnu.CmdLineParser;

import java.util.Arrays;
import java.util.List;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.sql.ResultSet;
import java.sql.SQLException;

import edu.uoregon.tau.common.Common;
import edu.uoregon.tau.common.TauRuntimeException;
import edu.uoregon.tau.common.Wget;
import edu.uoregon.tau.common.tar.Tar;
import edu.uoregon.tau.perfdmf.Database;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.database.ConnectionManager;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.DBConnector;
import edu.uoregon.tau.perfdmf.database.ParseConfig;
import edu.uoregon.tau.perfdmf.database.PasswordField;

public class Configure {
    private static String Usage = "Usage: configure [{-h,--help}] --create-default [{-g,--configfile} filename] [{-c --config} configuration_name] [{-t,--tauroot} path]";
    private static final String GREETING =
        "\nWelcome to the configuration program for PerfDMF.\n"
        + "This program will prompt you for some information necessary to ensure\n"
        + "the desired behavior for the PerfDMF tools.\n";

    /** all types of databases */
    private static final List<String> ALL_DB_NAMES =
        Arrays.asList("oracle", "postgresql", "mysql", "derby", "db2", "h2", "sqlite");

    /** databases that use files for storage */
    private static final List<String> FILE_DB_NAMES =
        Arrays.asList("derby", "h2", "sqlite");


    // todo - remove these defaults
    // todo - consider using a hash table!
    private String configuration_name = "";
    private String jardir;
    private String schemadir;
    private String jdbc_db_jarfile = "h2.jar";
    private String jdbc_db_driver = "org.h2.Driver";
    private String jdbc_db_type = "h2";
    private String db_hostname = "";
    private String db_portnum = "";
    private String db_dbname = "perfdmf";
    private String db_username = "";
    private String db_password = "";
    private String db_schemaprefix = "";
    private String db_schemafile = "taudb.sql";
    private String xml_parser = "xerces.jar";

    // Authentication and SSL support
    // Supports either SSL client certs (keys) or SSL connection with password auth.
    private boolean store_db_password = false;
	private boolean db_use_ssl = false;
	private boolean db_use_ssl_keys = false;

	private String db_keystore = "";
	private String db_keystore_password = "";
	private String db_truststore = "";
	private String db_truststore_password = "";

    private ParseConfig parser;
    private boolean configFileFound = false;
    //private String etc = File.separator + "etc" + File.separator;

    private String configFileName;

    public Configure(String jardir, String schemadir) {
        super();
        this.jardir = jardir;
        this.schemadir = schemadir;
    }

    public void errorPrint(String msg) {
        System.err.println(msg);
    }

    /**
     * Initialize method
     *  This method will welcome the user to the program, and prompt them
     *  for some basic information.
     **/

    public void initialize(String configFileNameIn, String configName) {
        if (configName == null) configName = "";

        try {
            // Check to see if the configuration file exists
            configFileName = configFileNameIn;
            File configFile = new File(configFileName);
            if (configFile.exists()) {
                System.out.println("Configuration file found...");
                // Parse the configuration file
                parseConfigFile();
                configuration_name = configName;
                configFileFound = true;
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

    public void initialize(String configFileNameIn) {
        initialize(configFileNameIn, null);
    }


    /** parseConfigFile method
     *  This method opens the configuration file for parsing, and passes
     *  each data line to the ParseConfigField() method.
     **/

    public void parseConfigFile() throws IOException, FileNotFoundException {
        System.out.println("Parsing config file...");
        parser = new ParseConfig(configFileName);
        //perfdmf_home = parser.getPerfDMFHome();
        jdbc_db_jarfile = parser.getJDBCJarFile();
        jdbc_db_driver = parser.getJDBCDriver();
        jdbc_db_type = parser.getDBType();
        db_hostname = parser.getDBHost();
        db_portnum = parser.getDBPort();
        db_schemaprefix = parser.getDBSchemaPrefix();
        db_dbname = parser.getDBName();
        db_username = parser.getDBUserName();
        db_password = parser.getDBPasswd();
        db_schemafile = parser.getDBSchema();
        xml_parser = parser.getXMLSAXParser();

        // various SSL parameters.
        db_use_ssl = parser.getDBUseSSL();
        db_keystore = parser.getDBKeystore();
        db_keystore_password = parser.getDBKeystorePasswd();
        db_truststore = parser.getDBTruststore();
        db_truststore_password = parser.getDBTruststorePasswd();
    }

    /** promptForData method
     *  This method prompts the user for each of the data fields
     *  in the configuration file.
     **/

    private String getUserJarDir() {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.trim().startsWith("windows")) {
            return jardir;
        } else {
            String dir = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator;
            File file = new File(dir);
            file.mkdirs();
            return dir;
        }
    }

    public void useDefaults(String dbname) {
        //String os = System.getProperty("os.name").toLowerCase();
        jdbc_db_jarfile = jardir + File.separator + "h2.jar";
		if (dbname == null) {
          db_dbname = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + "perfdmf" + File.separator + "perfdmf";
		} else {
          db_dbname = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + dbname + File.separator + "perfdmf";
		}
        jdbc_db_driver = "org.h2.Driver";
        db_schemafile = schemadir + File.separator + "taudb.sql";
        db_hostname = "";
        db_portnum = "";
        store_db_password = true;
        db_use_ssl = false;
        db_use_ssl_keys = false;
    }

    /**
     * Ask the user a yes/no question and return the answer as a boolean.
     */
    private boolean promptYN(BufferedReader reader, String prompt) throws IOException {
        while (true) {
            System.out.print(prompt + " (y/n): ");
            String tmp = reader.readLine();
            if (tmp.equalsIgnoreCase("yes") || tmp.equalsIgnoreCase("y")) {
                return true;
            } else if (tmp.equalsIgnoreCase("no") || tmp.equalsIgnoreCase("n")) {
                return false;
            }
        }
    }


    /**
     * Prompt the user to enter a string, and if they enter nothing return a default value.
     */
    private String promptString(BufferedReader reader, String prompt, String default_value) throws IOException {
        System.out.print(prompt + "\n(" + default_value + "): ");
        String tmpString = reader.readLine();
        if (tmpString.length() > 0) {
            return tmpString;
        } else {
            return default_value;
        }
    }


    private String promptPassword(String prompt) throws IOException {
        while (true) {
            String pwd1 = new PasswordField().getPassword(prompt + " ");
            String pwd2 = new PasswordField().getPassword("Confirm password: ");
            if (pwd1.equals(pwd2)) {
                return pwd1;
            } else {
                System.out.println("Passwords do not match.");
                System.out.println();
            }
        }
    }


    /**
     * Join a list of strings with commas and or.
     */
    private String join(List<String> strings, String delim) {
        String loop_delim = "";
        StringBuilder sb = new StringBuilder();
        for (String s:strings) {
            sb.append(loop_delim);
            sb.append(s);
            loop_delim = delim;
        }
        return sb.toString();
    }


    public void promptForData() {
        // Welcome the user to the program
        System.out.println(GREETING);
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        System.out.println("\nYou will now be prompted for new values, if desired.  "
                + "The current or default\nvalues for each prompt are shown "
                + "in parenthesis.\nTo accept the current/default value, " + "just press Enter/Return.\n");
        try {
            configuration_name = promptString(reader, "Please enter the name of this configuration.", configuration_name);

            String old_jdbc_db_type = jdbc_db_type;
            while (true) {
                // Prompt for database type
                jdbc_db_type = promptString(reader, "Please enter the database vendor (" + join(ALL_DB_NAMES, ", ") + ").",
                                         jdbc_db_type);
                if (ALL_DB_NAMES.contains(jdbc_db_type)) break;
            }


            if (configFileFound) {
                if (jdbc_db_type.compareTo("postgresql") == 0 && old_jdbc_db_type.compareTo("postgresql") != 0) {
                    // if the user has chosen postgresql and the config file is not already set for it
                    jdbc_db_jarfile = jardir + File.separator + "postgresql.jar";
                    jdbc_db_driver = "org.postgresql.Driver";
                    db_schemafile = schemadir + File.separator + "taudb.sql";
                    db_portnum = "5432";
                    db_hostname = "localhost";
                    db_dbname = "perfdmf";
                } else if (jdbc_db_type.compareTo("mysql") == 0 && old_jdbc_db_type.compareTo("mysql") != 0) {
                    // if the user has chosen mysql and the config file is not already set for it
                    jdbc_db_jarfile = jardir + File.separator + "mysql.jar";
                    jdbc_db_driver = "org.gjt.mm.mysql.Driver";
                    db_schemafile = schemadir + File.separator + "taudb.mysql.sql";
                    db_portnum = "3306";
                    db_hostname = "localhost";
                    db_dbname = "perfdmf";
                } else if (jdbc_db_type.compareTo("oracle") == 0 && old_jdbc_db_type.compareTo("oracle") != 0) {
                    // if the user has chosen oracle and the config file is not already set for it
                    jdbc_db_jarfile = getUserJarDir() + "ojdbc14.jar";
                    jdbc_db_driver = "oracle.jdbc.OracleDriver";
                    db_schemafile = schemadir + File.separator + "taudb.oracle.sql";
                    db_portnum = "1521";
                    db_hostname = "localhost";
                    db_dbname = "perfdmf";
                } else if (jdbc_db_type.compareTo("derby") == 0 && old_jdbc_db_type.compareTo("derby") != 0) {
                    // if the user has chosen derby and the config file is not already set for it
                    //String os = System.getProperty("os.name").toLowerCase();
                    jdbc_db_jarfile = jardir + File.separator + "derby.jar";
                    jdbc_db_driver = "org.apache.derby.jdbc.EmbeddedDriver";
                    db_schemafile = schemadir + File.separator + "taudb.derby.sql";
                    db_dbname = jardir + File.separator + "perfdmf";
                    db_hostname = "";
                    db_portnum = "";
                } else if (jdbc_db_type.compareTo("sqlite") == 0 && old_jdbc_db_type.compareTo("sqlite") != 0) {
                    // if the user has chosen sqlite and the config file is not already set for it
                    //String os = System.getProperty("os.name").toLowerCase();
                    jdbc_db_jarfile = jardir + File.separator + "sqlite.jar";
                    jdbc_db_driver = "org.sqlite.JDBC";
                    db_schemafile = schemadir + File.separator + "taudb.sqlite.sql";
                    db_dbname = jardir + File.separator + configuration_name + ".db";
                    db_hostname = "";
                    db_portnum = "";
                 } else if (jdbc_db_type.compareTo("h2") == 0 && old_jdbc_db_type.compareTo("h2") != 0) {
                    // if the user has chosen h2 and the config file is not already set for it
                    //String os = System.getProperty("os.name").toLowerCase();
                    jdbc_db_jarfile = jardir + File.separator + "h2.jar";
                    jdbc_db_driver = "org.h2.Driver";
                    db_schemafile = schemadir + File.separator + "taudb.sql";
                    db_dbname = jardir + File.separator + configuration_name;
                    db_hostname = "";
                    db_portnum = "";
                } else if (jdbc_db_type.compareTo("db2") == 0 && old_jdbc_db_type.compareTo("db2") != 0) {
                    // if the user has chosen db2 and the config file is not already set for it

                    jdbc_db_jarfile = ""; // there are 3 jar files...
                    jdbc_db_driver = "com.ibm.db2.jcc.DB2Driver";
                    db_schemafile = schemadir + File.separator + "taudb.db2.sql";
                    db_dbname = "perfdmf";
                    db_schemaprefix = "perfdmf";
                    db_hostname = "localhost";
                    db_portnum = "446";
                } else if (jdbc_db_type.compareTo("db2") == 0 && old_jdbc_db_type.compareTo("db2") == 0) {
                    // if the user has chosen db2 and the config file is already set for it
                    int endIndex = jdbc_db_jarfile.indexOf("java" + File.separator + "db2java.zip:");
                    if (endIndex == -1) {
                        endIndex = jdbc_db_jarfile.indexOf("java" + File.separator + "db2jcc.jar:");
                        if (endIndex == -1) {
                            jdbc_db_jarfile = "";
                        } else {
                            jdbc_db_jarfile = jdbc_db_jarfile.substring(0, endIndex - 1);
                        }
                    } else {
                        jdbc_db_jarfile = jdbc_db_jarfile.substring(0, endIndex - 1);
                    }
                }

            } else {

                if (jdbc_db_type.compareTo("postgresql") == 0) {
                    // if the user has chosen postgresql and the config file is not already set for it
                    jdbc_db_jarfile = "postgresql.jar";
                    jdbc_db_driver = "org.postgresql.Driver";
                    db_schemafile = "taudb.sql";
                    db_hostname = "localhost";
                    db_portnum = "5432";
                } else if (jdbc_db_type.compareTo("mysql") == 0) {
                    // if the user has chosen mysql and the config file is not already set for it
                    jdbc_db_jarfile = "mysql.jar";
                    jdbc_db_driver = "org.gjt.mm.mysql.Driver";
                    db_schemafile = "taudb.mysql.sql";
                    db_hostname = "localhost";
                    db_portnum = "3306";
                } else if (jdbc_db_type.compareTo("oracle") == 0) {
                    // if the user has chosen oracle and the config file is not already set for it
                    jdbc_db_jarfile = "ojdbc14.jar";
                    jdbc_db_driver = "oracle.jdbc.OracleDriver";
                    db_schemafile = "taudb.oracle.sql";
                    db_hostname = "localhost";
                    db_portnum = "1521";
                } else if (jdbc_db_type.compareTo("derby") == 0) {
                    // if the user has chosen derby and the config file is not already set for it
                    jdbc_db_jarfile = "derby.jar";
                    jdbc_db_driver = "org.apache.derby.jdbc.EmbeddedDriver";
                    db_schemafile = "taudb.derby.sql";
                    db_dbname = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + "perfdmf";
                    db_hostname = "";
                    db_portnum = "";
                } else if (jdbc_db_type.compareTo("sqlite") == 0) {
                    // if the user has chosen sqlite and the config file is not already set for it
                    jdbc_db_jarfile = "sqlite.jar";
                    jdbc_db_driver = "org.sqlite.JDBC";
                    db_schemafile = "taudb.sqlite.sql";
                    db_dbname = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + configuration_name + ".db";
                    db_hostname = "";
                    db_portnum = "";
                 } else if (jdbc_db_type.compareTo("h2") == 0) {
                    // if the user has chosen h2 and the config file is not already set for it
                    jdbc_db_jarfile = "h2.jar";
                    jdbc_db_driver = "org.h2.Driver";
                    db_schemafile = "taudb.sql";
                    db_dbname = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + configuration_name;
                    db_hostname = "";
                    db_portnum = "";
                } else if (jdbc_db_type.compareTo("db2") == 0) {
                    jdbc_db_jarfile = ""; // there are 3 jar files...
                    jdbc_db_driver = "com.ibm.db2.jcc.DB2Driver";
                    db_schemafile = "taudb.db2.sql";
                    db_dbname = "perfdmf";
                    db_schemaprefix = "perfdmf";
                    db_hostname = "localhost";
                    db_portnum = "446";
                }
            }

            if (!configFileFound) {
                if (jdbc_db_type.compareTo("derby") == 0) {
                    jdbc_db_jarfile = jardir + File.separator + "derby.jar";
                } else {
                    //jdbc_db_jarfile = getUserJarDir() + jdbc_db_jarfile;
                    jdbc_db_jarfile = jardir + File.separator + jdbc_db_jarfile;
                }
            }

            // Prompt for JDBC jar file
            if (jdbc_db_type.compareTo("db2") == 0) {
                jdbc_db_jarfile = promptString(reader, "Please enter the path to the DB2 sqllib directory,\n" +
                                               "often something like /home/db2_srv/sqllib.",
                                               jdbc_db_jarfile);
            } else {
                jdbc_db_jarfile = promptString(reader, "Please enter the JDBC jar file.", jdbc_db_jarfile);
            }
            jdbc_db_jarfile = jdbc_db_jarfile.replaceAll("~", System.getProperty("user.home"));

            if (!new File(jdbc_db_jarfile).exists()) {
                if (jdbc_db_type.compareToIgnoreCase("oracle") == 0) {
                    System.out.println("\nSorry, can't automatically download drivers for Oracle");
                    System.out.println("Please acquire them manually\n");
                } else if (jdbc_db_type.compareToIgnoreCase("db2") == 0) {
                    System.out.println("\nSorry, can't automatically download drivers for db2");
                    System.out.println("Please acquire them manually\n");
                } else {
                    System.out.println("\n\nCouldn't find jarfile: " + jdbc_db_jarfile);
                    System.out.println("\nJDBC drivers are not distributed with TAU.  You should acquire the JDBC driver");
                    System.out.println("that corresponds to the database you are connecting to.  TAU can now attempt ");
                    System.out.println("to download a JDBC driver that will *probably* work.");


                    System.out.println("\n");

                    if (promptYN(reader, "Would you like to attempt to automatically download a JDBC driver?")) {
                        try {

                            (new File(".perfdmf_tmp")).mkdirs();
                            System.setProperty("tar.location", ".perfdmf_tmp");

                            if (jdbc_db_type.equalsIgnoreCase("postgresql")) {
                                Wget.wget("http://www.cs.uoregon.edu/research/paracomp/tau/postgresql-redirect.html",
                                        ".perfdmf_tmp" + File.separator + "postgresql-redirect.html", false);
                                BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(new File(
                                        ".perfdmf_tmp" + File.separator + "postgresql-redirect.html"))));

                                String URL = "";
                                String line = r.readLine();
                                while (line != null) {
                                    if (line.startsWith("URL="))
                                        URL = line.substring(4);
                                    line = r.readLine();
                                }
                                r.close();

                                System.out.println("\nDownloading... " + URL);
                                System.out.print("Please Wait...");
                                Wget.wget(URL, jdbc_db_jarfile, true);
                                System.out.println(" Done");
                            }
                            if (jdbc_db_type.equalsIgnoreCase("mysql")) {
                                Wget.wget("http://www.cs.uoregon.edu/research/paracomp/tau/mysql-redirect.html", ".perfdmf_tmp"
                                        + File.separator + "mysql-redirect.html", false);

                                BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(new File(
                                        ".perfdmf_tmp" + File.separator + "mysql-redirect.html"))));

                                String URL = "";
                                String FILE = "";
                                String JAR = "";
                                String line = r.readLine();
                                while (line != null) {
                                    if (line.startsWith("URL="))
                                        URL = line.substring(4);
                                    if (line.startsWith("FILE="))
                                        FILE = line.substring(5);
                                    if (line.startsWith("JAR="))
                                        JAR = line.substring(4);
                                    line = r.readLine();
                                }
                                r.close();

                                System.out.println("\nDownloading... " + URL);
                                System.out.print("Please Wait...");
                                Wget.wget(URL, ".perfdmf_tmp/" + FILE, true);
                                System.out.println(" Done");
                                System.out.println("\nUncompressing...");
                                Tar.guntar(".perfdmf_tmp/" + FILE);
                                Common.copy(".perfdmf_tmp/" + JAR, jdbc_db_jarfile);
                            }

                            if (!new File(jdbc_db_jarfile).exists()) {
                                System.out.println("Unable to retrieve jarfile, please retrieve it manually");
                            }

                            Common.deltree(new File(".perfdmf_tmp"));

                        } catch (Exception e) {
                            System.out.println("Unable to retrieve jarfile:");
                            e.printStackTrace();
                        }
                    }
                }
            }

            if (jdbc_db_type.equals("db2")) {
                jdbc_db_jarfile = jdbc_db_jarfile + File.separator + "java" + File.separator + "db2java.zip:" + jdbc_db_jarfile
                        + File.separator + "java" + File.separator + "db2jcc.jar:" + jdbc_db_jarfile + File.separator
                        + "function:" + jdbc_db_jarfile + File.separator + "java" + File.separator + "db2jcc_license_cu.jar";
            }

            // Prompt for JDBC driver name
            jdbc_db_driver = promptString(reader, "Please enter the JDBC Driver name.", jdbc_db_driver);

            // Prompt for database hostname & port for non-file databases.
            if (!FILE_DB_NAMES.contains(jdbc_db_driver)) {
                db_hostname = promptString(reader, "Please enter the hostname for the database server.", db_hostname);
                db_portnum = promptString(reader, "Please enter the port number for the database JDBC connection.", db_portnum);
            }

            // Prompt for database name
            if (jdbc_db_type.equals("oracle")) {
                db_dbname = promptString(reader, "Please enter the oracle TCP service name.", db_dbname);

            } else if (FILE_DB_NAMES.contains(jdbc_db_type)) {
                db_dbname = promptString(reader, "Please enter the path to the database directory.", db_dbname);

            } else {
                db_dbname = promptString(reader, "Please enter the database name.", db_dbname);
            }

            // if the user used the ~ shortcut, expand it to $HOME, but only for file databases.
            if (FILE_DB_NAMES.contains(jdbc_db_type)) {
                db_dbname = db_dbname.replaceAll("~", System.getProperty("user.home"));
            }


            if (jdbc_db_type.compareTo("derby") == 0) {
                File f = new File(db_dbname);
                if (f.exists()) {
                    File f2 = new File(f + File.separator + "seg0");
                    System.out.println(f2);
                    if (!f2.exists()) {
                        System.out.println("\n\nWarning!  Directory \""
                                + db_dbname
                                + "\" exists and does not appear to be a derby database.\n"
                                + "Connection will most likely fail\n\n"
                                + "If you are trying to create a new Derby database, please specify a path that does not exist\n\n");
                    }
                }
            }

            if (jdbc_db_type.compareTo("h2") == 0) {
                File f = new File(db_dbname);
                if (f.exists()) {
                    File f2 = new File(f + File.separator + "perfdmf.h2.db");
                    System.out.println(f2);
                    if (!f2.exists()) {
                        System.out.println("\n\nWarning!  Directory \""
                                + db_dbname
                                + "\" exists and does not appear to be a h2 database.\n"
                                + "Connection will most likely fail\n\n"
                                + "If you are trying to create a new h2 database, please specify a path that does not exist\n\n");
                    }
                }
                db_dbname = db_dbname + File.separator + "perfdmf";
            }


            if (Arrays.asList("oracle", "db2").contains(jdbc_db_type)) {
                db_schemaprefix = promptString(reader, "Please enter the database schema name, " +
                                               "or your username if you are creating the tables now.",
                                               db_schemaprefix);
            }

            // Prompt for database username
            if (db_username.equals("")) {
                db_username = System.getProperty("user.name");
            }
            db_username = promptString(reader, "Please enter the database username.", db_username);

            //
            // Use SSL?
            //
            db_use_ssl = promptYN(reader, "Use SSL to connect to this database?");
            if (db_use_ssl) {
                // Set these up as sensible default.  They're only used if the files exist.
                // User can opt to use client certs by just installing a key.
                db_keystore = DBConnector.DEFAULT_KEYSTORE_PATH;
                db_keystore_password = DBConnector.DEFAULT_KEYSTORE_PASSWORD;
                db_truststore = DBConnector.DEFAULT_TRUSTSTORE_PATH;
                db_truststore_password = DBConnector.DEFAULT_TRUSTSTORE_PASSWORD;

                // Ask if the user wants to use keys for authentication.  If
                // they do, then just set the file up with some default
                // locations for the various keystores.
                db_use_ssl_keys = promptYN(reader, "Use SSL keys for authentication (requires password if not)?");
                if (db_use_ssl_keys) {
                    // Tell the user to use the key install tools.  Don't make them enter paths for everything.
                    System.out.println("    To use SSL client certificates with TauDB, use taudb_keygen to");
                    System.out.println("    create a key and have it signed by your database administrator.");
                    System.out.println("    Then use taudb_install_cert to add it to your TauDB configuration.");
                }
            }

            // If not using keys, get a password.
            if (!db_use_ssl_keys) {
                store_db_password = promptYN(reader,
                                             "Store the database password in CLEAR TEXT in your configuration file?");
                db_password = promptPassword("Please enter the database password:");
            }

            // Prompt for database schema file
            db_schemafile =
                promptString(reader, "Please enter the PerfDMF schema file.",
                             configFileFound ? db_schemafile : schemadir + File.separator + db_schemafile);
            db_schemafile = db_schemafile.replaceAll("~", System.getProperty("user.home"));

        } catch (IOException e) {
            // todo - get info from the exception
            System.out.println("I/O Error occurred.");
        }
    }

    /** testDBConnection method
     *  this method attempts to connect to the database.  If it cannot
     *  connect, it gives the user an error.  This method is intended
     *  to test the JDBC driver, servername, portnumber.
     **/

    public void testDBConnection() {
    // perfdmf.ConnectionManager.connect();
    // perfdmf.ConnectionManager.dbclose();
    }

    /** testDB method
     *  this method attempts to connect to the database.  If it cannot
     *  connect, it gives the user an error.  This method is intended
     *  to test the username, password, and database name.
     **/

    public void testDBTransaction() {}

    /** writeConfigFile method
     *  this method writes the configuration file back to
     *  perfdmf_home/bin/perfdmf.cfg.
     **/

    public String writeConfigFile() {
        try {
            // Check to see if the configuration file exists

            File configFile;

            File perfdmfpath = new File(System.getProperty("user.home") + File.separator + ".ParaProf");
            perfdmfpath.mkdirs();

            if (configuration_name.length() != 0) {
                // configuration name was specified
                configFile = new File(System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator
                        + "perfdmf.cfg." + configuration_name);
            } else {
                // I don't understand the logic below here
                if (configFileName == null || configFileName.length() == 0) {
                    if (configuration_name.length() == 0) {
                        configFile = new File(System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator
                                + "perfdmf.cfg");
                    } else {
                        configFile = new File(configFileName + "." + configuration_name);
                    }
                } else {
                    configFile = new File(configFileName);
                }
            }

            System.out.println("\nWriting configuration file: " + configFile);

            if (!configFile.exists()) {
                configFile.createNewFile();
            }
            BufferedWriter configWriter = new BufferedWriter(new FileWriter(configFile));
            configWriter.write("# This is the configuration file for the PerfDMF tools & API\n");
            configWriter.write("# Items are listed as name:value, one per line.\n");
            configWriter.write("# Comment lines begin with a '#' symbol.\n");
            configWriter.write("# DO NOT EDIT THIS FILE!  It is modified by the configure utility.\n");
            configWriter.newLine();

//            configWriter.write("# PerfDMF home directory\n");
//            configWriter.write("perfdmf_home:" + perfdmf_home + "\n");
//            configWriter.newLine();

            configWriter.write("# Database JDBC jar file (with path to location)\n");
            configWriter.write("jdbc_db_jarfile:" + jdbc_db_jarfile + "\n");
            configWriter.newLine();

            configWriter.write("# Database JDBC driver name\n");
            configWriter.write("jdbc_db_driver:" + jdbc_db_driver + "\n");
            configWriter.newLine();

            configWriter.write("# Database type\n");
            configWriter.write("jdbc_db_type:" + jdbc_db_type + "\n");
            configWriter.newLine();

            configWriter.write("# Database host name\n");
            configWriter.write("db_hostname:" + db_hostname + "\n");
            configWriter.newLine();

            configWriter.write("# Database port number\n");
            configWriter.write("db_portnum:" + db_portnum + "\n");
            configWriter.newLine();

            configWriter.write("# Database name\n");
            configWriter.write("db_dbname:" + db_dbname + "\n");
            configWriter.newLine();

            configWriter.write("# Database Schema name\n");
            configWriter.write("db_schemaprefix:" + db_schemaprefix + "\n");
            configWriter.newLine();

            configWriter.write("# Database username\n");
            configWriter.write("db_username:" + db_username + "\n");
            configWriter.newLine();

            if (db_use_ssl) {
                configWriter.write("# Use SSL to connect to database?\n");
                configWriter.write("db_use_ssl: yes\n");
                configWriter.newLine();

                configWriter.write("# Keystore for client authentication certificates.\n");
                configWriter.write("db_keystore:" + db_keystore + "\n");
                configWriter.newLine();

                configWriter.write("# Keystore password.\n");
                configWriter.write("db_keystore_password:" + db_keystore_password + "\n");
                configWriter.newLine();

                configWriter.write("# Truststore for server certs.\n");
                configWriter.write("db_truststore:" + db_truststore + "\n");
                configWriter.newLine();

                configWriter.write("# Truststore password.\n");
                configWriter.write("db_truststore_password:" + db_truststore_password + "\n");
                configWriter.newLine();
			}

            if (store_db_password) {
                configWriter.write("# Database password\n");
                configWriter.write("db_password:" + db_password + "\n");
                configWriter.newLine();
            }

            configWriter.write("# Database Schema file - note: the path is absolute\n");
            configWriter.write("db_schemafile:" + db_schemafile + "\n");
            configWriter.newLine();

            configWriter.write("# Database XML parser jar file - note: the path is absolute\n");
            configWriter.write("xml_sax_parser:" + xml_parser + "\n");
            configWriter.newLine();

            configWriter.close();
            return configFile.toString();
        } catch (IOException e) {
            e.printStackTrace();
            throw new TauRuntimeException(e);
        }
    }


    public void setJDBCJarfile(String inString) {
        jdbc_db_jarfile = inString;
    }

    public String getJDBCJarfile() {
        return jdbc_db_jarfile;
    }

    public void setJDBCDriver(String inString) {
        jdbc_db_driver = inString;
    }

    public String getJDBCDriver() {
        return jdbc_db_driver;
    }

    public void setJDBCType(Object object) {
        jdbc_db_type = (String) object;
    }

    public String getJDBCType() {
        return jdbc_db_type;
    }

    public void setDBHostname(String inString) {
        db_hostname = inString;
    }

    public String getDBHostname() {
        return db_hostname;
    }

    public void setDBPortNum(String inString) {
        db_portnum = inString;
    }

    public String getDBPortNum() {
        return db_portnum;
    }

    public void setDBName(String inString) {
        db_dbname = inString;
    }

    public String getDBName() {
        return db_dbname;
    }

    public void setDBUsername(String inString) {
        db_username = inString;
    }

    public String getDBUsername() {
        return db_username;
    }

    public void setDBPassword(String inString) {
        db_password = inString;
    }

    public String getDBPassword() {
        return db_password;
    }

    public void setDBSchemaFile(String inString) {
        db_schemafile = inString;
    }

    public String getDBSchemaFile() {
        return db_schemafile;
    }

    public void setXMLPaser(String inString) {
        xml_parser = inString;
    }

    public String getXMLPaser() {
        return xml_parser;
    }

    public void setConfigFileName(String inString) {
        configuration_name = inString;
    }

    public String getConfigFileName() {
        return configuration_name;
    }

    public void savePassword() {
        this.store_db_password = true;
    }

    /* Test that the database exists, and if it doesn't, create it! */
    public void createDB() throws DatabaseConfigurationException {
        ConnectionManager connector = null;
        DB db = null;
        try {
            connector = new ConnectionManager(new Database(configFileName), true);
            connector.connect();
            db = connector.getDB();
        } catch (SQLException e) {
            System.out.println("\nPlease make sure that your DBMS is configured correctly, and");
            System.out.println("the database " + db_dbname + " has been created.");
            //throw new DatabaseConfigurationException("Error Connection to Database" + db_dbname);
            throw new DatabaseException("Error Connection to Database" + db_dbname, e);
        }
        try {
            String query = new String("select * from trial;");
            ResultSet resultSet = db.executeQuery(query);
            resultSet.close();
            connector.dbclose();
        } catch (Exception e) {
            // build the database
            System.out.println(configFileName);
            connector.genParentSchema(db_schemafile);
            connector.dbclose();
            System.out.println("Congratulations! PerfDMF is configured and the database has been built.");
            System.out.println("You may begin loading applications.");
        }
        System.out.println("Configuration complete.");
    }

    public static void createDefault(String configFile, String tauroot, String arch, String dbName) {
        // Create a new Configure object, which will walk the user through
        // the process of creating/editing a configuration file.
        Configure config = new Configure(tauroot, arch);
        config.initialize(configFile);
        config.useDefaults(dbName);
        config.setDBName(dbName);

        // Write the configuration file to ${PerfDMF_Home}/bin/perfdmf.cfg
        String configFilename = config.writeConfigFile();

        ConfigureTest configTest = new ConfigureTest();
        configTest.initialize(configFilename);
        try {
            configTest.createDB(false);
        } catch (DatabaseConfigurationException e) {
            e.printStackTrace();
        }
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {

        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option configOpt = parser.addStringOption('c', "config");
        CmdLineParser.Option jarLocationOpt = parser.addStringOption('j', "jardir");
        CmdLineParser.Option schemaLocationOpt = parser.addStringOption('a', "schemadir");
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option defaultOpt = parser.addBooleanOption('d', "create-default");
        CmdLineParser.Option createOnlyOpt = parser.addBooleanOption('C', "connect-without-prompt");

        try {
            parser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.err.println(e.getMessage());
            System.err.println(Usage);
            System.exit(-1);
        }
        String configFile = (String) parser.getOptionValue(configfileOpt);
        String configName = (String) parser.getOptionValue(configOpt);
        String jardir = (String) parser.getOptionValue(jarLocationOpt);
        String schemadir = (String) parser.getOptionValue(schemaLocationOpt);
        Boolean help = (Boolean) parser.getOptionValue(helpOpt);
        Boolean useDefaults = (Boolean) parser.getOptionValue(defaultOpt);
        Boolean createOnly = (Boolean) parser.getOptionValue(createOnlyOpt);

        if (help != null && help.booleanValue()) {
            System.err.println(Usage);
            System.exit(-1);
        }

        if (configFile == null) {
            if (configName == null) {
                configFile = System.getProperty("user.home") + File.separator
                    + ".ParaProf" + File.separator + "perfdmf.cfg";
            } else {
                configFile = System.getProperty("user.home") + File.separator
                    + ".ParaProf" + File.separator + "perfdmf.cfg." + configName;
            }
        }

        System.out.println(createOnly);

        if (createOnly == null || !createOnly) {
            if (useDefaults == null) {
                useDefaults = Boolean.FALSE;
            }

            // Create a new Configure object, which will walk the user through
            // the process of creating/editing a configuration file.
            Configure config = new Configure(jardir, schemadir);
            config.initialize(configFile, configName);

            if (useDefaults == Boolean.TRUE) {
                config.useDefaults(configName);
            } else {
                // Give the user the ability to modify any/everything
                config.promptForData();
            }
            // Test the database connection
            //config.testDBConnection();

            // Test the database name/login/password, etc.
            //config.testDBTransaction();

            // Write the configuration file to ${PerfDMF_Home}/bin/perfdmf.cfg
            configFile = config.writeConfigFile();
        }

        ConfigureTest configTest = new ConfigureTest();
        configTest.initialize(configFile);
        try {
            configTest.createDB(false);
        } catch (DatabaseConfigurationException e) {
            e.printStackTrace();
            System.exit(0);
        }

        // check to see if the database is there...
        //config.createDB();

        /*
         String classpath = System.getProperty("java.class.path");

         //System.out.println ("executing '" + "java -cp " + classpath + ":" + config.getJDBCJarfile() + " edu.uoregon.tau.dms.loader.Configure");

         String execString = new String("java -cp " + classpath + ":" + config.getJDBCJarfile() + " edu.uoregon.tau.dms.loader.Configure -a " + arch + " -g " + configFile + " -t" + tauroot);

         try {
         System.out.println ("Executing " + execString);
         Process tester = Runtime.getRuntime().exec(execString);
         tester.waitFor();

         int exitValue = tester.exitValue();
         System.out.println ("exit value was: " + exitValue);
         }

         catch (Exception e) {
         System.out.println ("Error executing database test, tried to execute: " + execString);
         e.printStackTrace();
         }
         */

    }

}
