package edu.uoregon.tau.perfdmf.loader;

import jargs.gnu.CmdLineParser;

import java.io.*;
import java.sql.ResultSet;
import java.sql.SQLException;

import edu.uoregon.tau.common.Common;
import edu.uoregon.tau.common.TauRuntimeException;
import edu.uoregon.tau.common.Wget;
import edu.uoregon.tau.common.tar.Tar;
import edu.uoregon.tau.perfdmf.Database;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.database.*;

public class Configure {
    private static String Usage = "Usage: configure [{-h,--help}] --create-default [{-g,--configfile} filename] [{-c --config} configuration_name] [{-t,--tauroot} path]";
    private static String Greeting = "\nWelcome to the configuration program for PerfDMF.\n"
            + "This program will prompt you for some information necessary to ensure\n"
            + "the desired behavior for the PerfDMF tools.\n";

    // todo - remove these defaults
    // todo - consider using a hash table!
    private String configuration_name = "";
    private String jardir;
    private String schemadir;
    private String jdbc_db_jarfile = "derby.jar";
    private String jdbc_db_driver = "org.apache.derby.jdbc.EmbeddedDriver";
    private String jdbc_db_type = "derby";
    private String db_hostname = "";
    private String db_portnum = "";
    private String db_dbname = "perfdmf";
    private String db_username = "";
    private String db_password = "";
    private String db_schemaprefix = "";
    private boolean store_db_password = false;
    private String db_schemafile = "dbschema.derby.txt";
    private String xml_parser = "xerces.jar";
    private ParseConfig parser;
    private boolean configFileFound = false;
    private String etc = File.separator + "etc" + File.separator;

    private String configFileName;

    public Configure(String jardir, String schemadir) {
        super();
        this.jardir = jardir;
        this.schemadir = schemadir;
    }

    public void errorPrint(String msg) {
        System.err.println(msg);
    }

    /** Initialize method 
     *  This method will welcome the user to the program, and prompt them
     *  for some basic information. 
     **/

    public void initialize(String configFileNameIn) {

        try {
            // Check to see if the configuration file exists
            configFileName = configFileNameIn;
            File configFile = new File(configFileName);
            if (configFile.exists()) {
                System.out.println("Configuration file found...");
                // Parse the configuration file
                parseConfigFile();
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

    public void useDefaults() {
        String os = System.getProperty("os.name").toLowerCase();
        jdbc_db_jarfile = jardir + File.separator + "derby.jar";
        db_dbname = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + "perfdmf";
        jdbc_db_driver = "org.apache.derby.jdbc.EmbeddedDriver";
        db_schemafile = schemadir + File.separator + "dbschema.derby.txt";
        db_hostname = "";
        db_portnum = "";
        store_db_password = true;
    }

    public void promptForData() {
        // Welcome the user to the program
        System.out.println(Greeting);

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String tmpString;

        /*
         if (configFileFound) {
         // if the configuration file already exists, give the user the option to just 
         // stick with their current options (they may be using it to test connectivity
         System.out.println("TAU root directory: " + tau_root);
         System.out.println("

         }
         */

        System.out.println("\nYou will now be prompted for new values, if desired.  "
                + "The current or default\nvalues for each prompt are shown "
                + "in parenthesis.\nTo accept the current/default value, " + "just press Enter/Return.\n");
        try {
            System.out.print("Please enter the name of this configuration.\n():");
            tmpString = reader.readLine();
            if (tmpString.length() > 0) {
                configuration_name = tmpString;
            } else {
                configuration_name = "";
            }

         

            String old_jdbc_db_type = jdbc_db_type;
            boolean valid = false;

            while (!valid) {
                // Prompt for database type
                System.out.println("Please enter the database vendor (oracle, postgresql, mysql, db2 or derby).");
                System.out.print("(" + jdbc_db_type + "):");
                tmpString = reader.readLine();
                if (tmpString.compareTo("oracle") == 0 || tmpString.compareTo("postgresql") == 0
                        || tmpString.compareTo("mysql") == 0 || tmpString.compareTo("derby") == 0
                        || tmpString.compareTo("db2") == 0 || tmpString.length() == 0) {
                    if (tmpString.length() > 0) {
                        jdbc_db_type = tmpString;
                    }
                    valid = true;
                }
            }

            if (configFileFound) {
                if (jdbc_db_type.compareTo("postgresql") == 0 && old_jdbc_db_type.compareTo("postgresql") != 0) {
                    // if the user has chosen postgresql and the config file is not already set for it
                    jdbc_db_jarfile = getUserJarDir() + "postgresql.jar";
                    jdbc_db_driver = "org.postgresql.Driver";
                    db_schemafile = schemadir + File.separator + "dbschema.txt";
                    db_portnum = "5432";
                    db_hostname = "localhost";
                    db_dbname = "perfdmf";
                } else if (jdbc_db_type.compareTo("mysql") == 0 && old_jdbc_db_type.compareTo("mysql") != 0) {
                    // if the user has chosen mysql and the config file is not already set for it
                    jdbc_db_jarfile = getUserJarDir() + "mysql.jar";
                    jdbc_db_driver = "org.gjt.mm.mysql.Driver";
                    db_schemafile = schemadir + File.separator + "dbschema.mysql.txt";
                    db_portnum = "3306";
                    db_hostname = "localhost";
                    db_dbname = "perfdmf";
                } else if (jdbc_db_type.compareTo("oracle") == 0 && old_jdbc_db_type.compareTo("oracle") != 0) {
                    // if the user has chosen oracle and the config file is not already set for it
                    jdbc_db_jarfile = getUserJarDir() + "ojdbc14.jar";
                    jdbc_db_driver = "oracle.jdbc.OracleDriver";
                    db_schemafile = schemadir + File.separator + "dbschema.oracle.txt";
                    db_portnum = "1521";
                    db_hostname = "localhost";
                    db_dbname = "perfdmf";
                } else if (jdbc_db_type.compareTo("derby") == 0 && old_jdbc_db_type.compareTo("derby") != 0) {
                    // if the user has chosen derby and the config file is not already set for it
                    String os = System.getProperty("os.name").toLowerCase();
                    jdbc_db_jarfile = jardir + File.separator + "derby.jar";
                    jdbc_db_driver = "org.apache.derby.jdbc.EmbeddedDriver";
                    db_schemafile = schemadir + File.separator + "dbschema.derby.txt";
                    db_dbname = jardir + File.separator + "perfdmf";
                    db_hostname = "";
                    db_portnum = "";
                } else if (jdbc_db_type.compareTo("db2") == 0 && old_jdbc_db_type.compareTo("db2") != 0) {
                    // if the user has chosen db2 and the config file is not already set for it

                    jdbc_db_jarfile = ""; // there are 3 jar files...
                    jdbc_db_driver = "com.ibm.db2.jcc.DB2Driver";
                    db_schemafile = schemadir + File.separator + "dbschema.db2.txt";
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
                    db_schemafile = "dbschema.txt";
                    db_hostname = "localhost";
                    db_portnum = "5432";
                } else if (jdbc_db_type.compareTo("mysql") == 0) {
                    // if the user has chosen mysql and the config file is not already set for it
                    jdbc_db_jarfile = "mysql.jar";
                    jdbc_db_driver = "org.gjt.mm.mysql.Driver";
                    db_schemafile = "dbschema.mysql.txt";
                    db_hostname = "localhost";
                    db_portnum = "3306";
                } else if (jdbc_db_type.compareTo("oracle") == 0) {
                    // if the user has chosen oracle and the config file is not already set for it
                    jdbc_db_jarfile = "ojdbc14.jar";
                    jdbc_db_driver = "oracle.jdbc.OracleDriver";
                    db_schemafile = "dbschema.oracle.txt";
                    db_hostname = "localhost";
                    db_portnum = "1521";
                } else if (jdbc_db_type.compareTo("derby") == 0) {
                    // if the user has chosen derby and the config file is not already set for it
                    jdbc_db_jarfile = "derby.jar";
                    jdbc_db_driver = "org.apache.derby.jdbc.EmbeddedDriver";
                    db_schemafile = "dbschema.derby.txt";
                    db_dbname = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + "perfdmf";
                    db_hostname = "";
                    db_portnum = "";
                } else if (jdbc_db_type.compareTo("db2") == 0) {
                    jdbc_db_jarfile = ""; // there are 3 jar files...
                    jdbc_db_driver = "com.ibm.db2.jcc.DB2Driver";
                    db_schemafile = "dbschema.db2.txt";
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
                System.out.println("Please enter the path to the DB2 sqllib directory,");
                System.out.println("often something like /home/db2_srv/sqllib.");
                System.out.print("(" + jdbc_db_jarfile + "):");
            } else {
                System.out.print("Please enter the JDBC jar file.\n(" + jdbc_db_jarfile + "):");
            }

            tmpString = reader.readLine();
            if (tmpString.length() > 0) {
                jdbc_db_jarfile = tmpString.replaceAll("~", System.getProperty("user.home"));
            }

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

                    boolean responded = false;
                    boolean response = false;
                    while (!responded) {
                        System.out.print("\nWould you like to attempt to automatically download a JDBC driver? (y/n):");
                        tmpString = reader.readLine();
                        if (tmpString.compareToIgnoreCase("yes") == 0 || tmpString.compareToIgnoreCase("y") == 0) {
                            responded = true;
                            response = true;
                        }
                        if (tmpString.compareToIgnoreCase("no") == 0 || tmpString.compareToIgnoreCase("n") == 0) {
                            responded = true;
                            response = false;
                        }
                    }

                    if (response) {
                        try {

                            (new File(".perfdmf_tmp")).mkdirs();
                            System.setProperty("tar.location", ".perfdmf_tmp");

                            if (jdbc_db_type.compareToIgnoreCase("postgresql") == 0) {
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
                            if (jdbc_db_type.compareToIgnoreCase("mysql") == 0) {
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

            if (jdbc_db_type.compareTo("db2") == 0) {
                tmpString = jdbc_db_jarfile + File.separator + "java" + File.separator + "db2java.zip:" + jdbc_db_jarfile
                        + File.separator + "java" + File.separator + "db2jcc.jar:" + jdbc_db_jarfile + File.separator
                        + "function:" + jdbc_db_jarfile + File.separator + "java" + File.separator + "db2jcc_license_cu.jar";
                jdbc_db_jarfile = tmpString;
            }

            // Prompt for JDBC driver name
            System.out.print("Please enter the JDBC Driver name.\n(" + jdbc_db_driver + "):");
            tmpString = reader.readLine();
            if (tmpString.length() > 0)
                jdbc_db_driver = tmpString;

            if (jdbc_db_type.compareTo("derby") != 0) {
                // Prompt for database hostname
                System.out.print("Please enter the hostname for the database server.\n(" + db_hostname + "):");
                tmpString = reader.readLine();
                if (tmpString.length() > 0)
                    db_hostname = tmpString;

                // Prompt for database portnumber
                System.out.print("Please enter the port number for the database JDBC connection.\n(" + db_portnum + "):");
                tmpString = reader.readLine();
                if (tmpString.length() > 0)
                    db_portnum = tmpString;
            }

            // Prompt for database name

            if (jdbc_db_type.compareTo("oracle") == 0) {
                System.out.print("Please enter the oracle TCP service name.\n(" + db_dbname + "):");
            } else if (jdbc_db_type.compareTo("derby") == 0) {
                System.out.print("Please enter the path to the database directory.\n(" + db_dbname + "):");
            } else {
                System.out.print("Please enter the database name.\n(" + db_dbname + "):");
            }
            tmpString = reader.readLine();
            if (tmpString.length() > 0) {
                // if the user used the ~ shortcut, expand it to $HOME.
                if (jdbc_db_type.compareTo("derby") == 0) {
                    db_dbname = tmpString.replaceAll("~", System.getProperty("user.home"));
                } else {
                    db_dbname = tmpString;
                }
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

            if ((jdbc_db_type.compareTo("oracle") == 0) || (jdbc_db_type.compareTo("db2") == 0)) {
                System.out.print("Please enter the database schema name, or your username if you are creating the tables now.\n("
                        + db_schemaprefix + "):");
                tmpString = reader.readLine();
                if (tmpString.length() > 0)
                    db_schemaprefix = tmpString;
            }

            // Prompt for database username
            System.out.print("Please enter the database username.\n(" + db_username + "):");
            tmpString = reader.readLine();
            if (tmpString.length() > 0)
                db_username = tmpString;

            boolean responded = false;
            boolean response = false;
            while (!responded) {
                System.out.print("Store the database password in CLEAR TEXT in your configuration file? (y/n):");
                tmpString = reader.readLine();
                if (tmpString.compareToIgnoreCase("yes") == 0 || tmpString.compareToIgnoreCase("y") == 0) {
                    responded = true;
                    response = true;
                }
                if (tmpString.compareToIgnoreCase("no") == 0 || tmpString.compareToIgnoreCase("n") == 0) {
                    responded = true;
                    response = false;
                }
            }

            if (response == true) {
                PasswordField passwordField = new PasswordField();
                db_password = passwordField.getPassword("Please enter the database password:");
                store_db_password = true;
            }

            /*
             boolean passwordMatch = false;
             while (!passwordMatch) {
             // Prompt for database password
             System.out.println("NOTE: Passwords will be stored in an encrypted format.");
             PasswordField passwordField = new PasswordField();
             tmpString = passwordField.getPassword("Please enter the database password (default not shown):");
             if (tmpString.length() > 0) db_password = tmpString;
             String tmpString2 = passwordField.getPassword("Please enter the database password again to confirm:");
             if (tmpString.compareTo(tmpString2) == 0) {
             db_password = tmpString;
             passwordMatch = true;
             }
             else System.out.println ("Password confirmation failed.  Please try again.");
             }
             */

            // Prompt for database schema file
            if (configFileFound) {
                System.out.print("Please enter the PerfDMF schema file.\n(" + db_schemafile + "):");
            } else {
                System.out.print("Please enter the PerfDMF schema file.\n(" + schemadir + File.separator + db_schemafile + "):");
            }
            tmpString = reader.readLine();
            if (tmpString.length() > 0) {
                db_schemafile = tmpString.replaceAll("~", System.getProperty("user.home"));
            } else if (!configFileFound) {
                db_schemafile = schemadir + File.separator + db_schemafile;
            }

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
            String query = new String("select * from application;");
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
        config.useDefaults();
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

        if (help != null && help.booleanValue()) {
            System.err.println(Usage);
            System.exit(-1);
        }

        if (configFile == null) {
            if (configName == null) {
                configFile = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + "perfdmf.cfg";
            } else {
                configFile = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + "perfdmf.cfg."
                        + configName;
            }

        }

        if (useDefaults == null) {
            useDefaults = Boolean.FALSE;
        }

        // Create a new Configure object, which will walk the user through
        // the process of creating/editing a configuration file.
        Configure config = new Configure(jardir, schemadir);
        config.initialize(configFile);

        if (useDefaults == Boolean.TRUE) {
            config.useDefaults();
        } else {
            // Give the user the ability to modify any/everything
            config.promptForData();
        }
        // Test the database connection
        //config.testDBConnection();

        // Test the database name/login/password, etc.
        //config.testDBTransaction();

        // Write the configuration file to ${PerfDMF_Home}/bin/perfdmf.cfg
        String configFilename = config.writeConfigFile();

        ConfigureTest configTest = new ConfigureTest();
        configTest.initialize(configFilename);
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
