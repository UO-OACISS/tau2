package edu.uoregon.tau.dms.loader;

import edu.uoregon.tau.dms.database.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.sql.*;
import jargs.gnu.CmdLineParser;



public class Configure {
    private DB db = null;
    private static String Usage = "Usage: configure [{-h,--help}] [{-g,--configfile} filename] [{-t,--tauroot} path]";
    private static String Greeting = "\nWelcome to the configuration program for PerfDMF.\n" +
	"This program will prompt you for some information necessary to ensure\n" +
	"the desired behavior for the PerfDMF tools.\n";
    private static String PDBHomePrompt = "Please enter the PerfDMF home directory:";
		
    // todo - remove these defaults
    // todo - consider using a hash table!
    private String perfdmf_home = "";
    private String tau_root = "";
    private String arch = "";
    private String jdbc_db_jarfile = "postgresql.jar";
    private String jdbc_db_driver = "org.postgresql.Driver";
    private String jdbc_db_type = "postgresql";
    private String db_hostname = "localhost";
    private String db_portnum = "5432";
    private String db_dbname = "perfdmf";
    private String db_username = "";
    private String db_password = "";
    private String db_schemaprefix = "";
    private boolean store_db_password = false;
    private String db_schemafile = "dbschema.txt";
    private String xml_parser = "xerces.jar";
    private ParseConfig parser;
    private boolean configFileFound = false;
		
    private String configFileName;

    public Configure(String tauroot, String arch) {
	super();
	this.tau_root = tauroot;
	this.arch = arch;
	this.perfdmf_home = tauroot + "/tools/src/dms";
    }

    public void errorPrint(String msg) {
	System.err.println(msg);
    }

    /** Initialize method 
     *  This method will welcome the user to the program, and prompt them
     *  for some basic information. 
     **/
		
    public void initialize(String configFileNameIn) {
	// Create a reader to parse the input
	BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
				
	// Welcome the user to the program
	System.out.println(Greeting);
				
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
	}
	catch (IOException e) {
	    // todo - get info from the exception
	    System.out.println ("I/O Error occurred.");
	}
    }
	
    /** parseConfigFile method 
     *  This method opens the configuration file for parsing, and passes
     *  each data line to the ParseConfigField() method.
     **/

    public void parseConfigFile() throws IOException, FileNotFoundException {
	System.out.println("Parsing config file...");
	parser = new ParseConfig(configFileName);
	perfdmf_home = parser.getPerfDMFHome();
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
    public void promptForData() {
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
	
	System.out.println("\nYou will now be prompted for new values, if desired.  " +
			   "The current or default\nvalues for each prompt are shown " +
			   "in parenthesis.\nTo accept the current/default value, " +
			   "just press Enter/Return.\n");
	try {
	    System.out.print("Please enter the TAU root directory.\n(" + tau_root + "):");
	    tmpString = reader.readLine();
	    if (tmpString.length() > 0) { 
		tau_root = tmpString;
		perfdmf_home = tau_root + "/tools/src/dms";
	    }


	    String old_jdbc_db_type = jdbc_db_type;
	    boolean valid = false;

	    while (!valid) {
		// Prompt for database type
		System.out.print("Please enter the database vendor (oracle, postgresql or mysql).\n(" 
				 + jdbc_db_type + "):");
		tmpString = reader.readLine();
		if (tmpString.compareTo("oracle") == 0 || tmpString.compareTo("postgresql") == 0 
		    || tmpString.compareTo("mysql") == 0 || tmpString.length() == 0) {
		
		    if (tmpString.length() > 0) 
			jdbc_db_type = tmpString;
		    valid = true;
		}
	    }

	    if (configFileFound) {
		if (jdbc_db_type.compareTo("postgresql") == 0 && old_jdbc_db_type.compareTo("postgresql") != 0) {
		    // if the user has chosen postgresql and the config file is not already set for it
		    jdbc_db_jarfile = tau_root + "/" + arch + "/lib/" + "postgresql.jar";
		    jdbc_db_driver = "org.postgresql.Driver";
		    db_schemafile = perfdmf_home + "/data/" +"dbschema.txt";
		    db_portnum = "5432";
		} else if (jdbc_db_type.compareTo("mysql") == 0 && old_jdbc_db_type.compareTo("mysql") != 0) {
		    // if the user has chosen mysql and the config file is not already set for it
		    jdbc_db_jarfile = tau_root + "/" + arch + "/lib/" + "mysql.jar";
		    jdbc_db_driver = "org.gjt.mm.mysql.Driver";
		    db_schemafile = perfdmf_home + "/data/" + "dbschema.mysql.txt";
		    db_portnum = "3306";
		} else if (jdbc_db_type.compareTo("oracle") == 0 && old_jdbc_db_type.compareTo("oracle") != 0) {
		    // if the user has chosen oracle and the config file is not already set for it
		    jdbc_db_jarfile = tau_root + "/" + arch + "/lib/" + "ojdbc14.jar";
		    jdbc_db_driver = "oracle.jdbc.OracleDriver";
		    db_schemafile = perfdmf_home + "/data/" + "dbschema.oracle.txt";
		    db_portnum = "1521";
		}

	    } else {

		if (jdbc_db_type.compareTo("postgresql") == 0) {
		    // if the user has chosen postgresql and the config file is not already set for it
		    jdbc_db_jarfile = "postgresql.jar";
		    jdbc_db_driver = "org.postgresql.Driver";
		    db_schemafile = "dbschema.txt";
		    db_portnum = "5432";
		} else if (jdbc_db_type.compareTo("mysql") == 0) {
		    // if the user has chosen mysql and the config file is not already set for it
		    jdbc_db_jarfile = "mysql.jar";
		    jdbc_db_driver = "org.gjt.mm.mysql.Driver";
		    db_schemafile = "dbschema.mysql.txt";
		    db_portnum = "3306";
		} else if (jdbc_db_type.compareTo("oracle") == 0) {
		    // if the user has chosen mysql and the config file is not already set for it
		    jdbc_db_jarfile = "ojdbc14.jar";
		    jdbc_db_driver = "oracle.jdbc.OracleDriver";
		    db_schemafile = "dbschema.oracle.txt";
		    db_portnum = "1521";
		}
	    }
		
						
	    // Prompt for JDBC jar file
	    if (configFileFound)
		System.out.print("Please enter the JDBC jar file.\n(" + jdbc_db_jarfile + "):");
	    else
		System.out.print("Please enter the JDBC jar file.\n(" + tau_root + "/" + arch + "/lib/" + jdbc_db_jarfile + "):");
						
	    tmpString = reader.readLine();
	    if (tmpString.length() > 0) jdbc_db_jarfile = tmpString;
	    else if (!configFileFound) 
		jdbc_db_jarfile = tau_root + "/" + arch + "/lib/" + jdbc_db_jarfile;
						
	    // Prompt for JDBC driver name
	    System.out.print("Please enter the JDBC Driver name.\n(" + jdbc_db_driver + "):");
	    tmpString = reader.readLine();
	    if (tmpString.length() > 0) jdbc_db_driver = tmpString;
						
						
	    // Prompt for database hostname
	    System.out.print("Please enter the hostname for the database server.\n(" + db_hostname + "):");
	    tmpString = reader.readLine();
	    if (tmpString.length() > 0) db_hostname = tmpString;
						
	    // Prompt for database portnumber
	    System.out.print("Please enter the port number for the database JDBC connection.\n(" + db_portnum + "):");
	    tmpString = reader.readLine();
	    if (tmpString.length() > 0) db_portnum = tmpString;
						
	    // Prompt for database name

	    if (jdbc_db_type.compareTo("oracle") == 0) {
		System.out.print("Please enter the oracle TCP service name.\n(" + db_dbname + "):");
	    } else {
		System.out.print("Please enter the database name.\n(" + db_dbname + "):");
	    }
	    tmpString = reader.readLine();
	    if (tmpString.length() > 0) db_dbname = tmpString;


	    if (jdbc_db_type.compareTo("oracle") == 0) {
		System.out.print("Please enter the database schema name, or your username if you are creating the tables now.\n(" + db_schemaprefix + "):");
		tmpString = reader.readLine();
		if (tmpString.length() > 0) db_schemaprefix = tmpString;
	    }

						
	    // Prompt for database username
	    System.out.print("Please enter the database username.\n(" + db_username + "):");
	    tmpString = reader.readLine();
	    if (tmpString.length() > 0) db_username = tmpString;


	    System.out.print("Would you like to store the database password in CLEAR TEXT in your configuration file?\n(no):");
	    tmpString = reader.readLine();

	    if (tmpString.compareTo("yes")==0 || tmpString.compareTo("y")==0 || tmpString.compareTo("Y")==0) {
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
	    if (configFileFound)
		System.out.print("Please enter the PerfDMF schema file.\n(" + db_schemafile + "):");
	    else
		System.out.print("Please enter the PerfDMF schema file.\n(" + perfdmf_home + "/data/" + db_schemafile + "):");
	    tmpString = reader.readLine();
	    if (tmpString.length() > 0) db_schemafile = tmpString;
	    else if (!configFileFound)
		db_schemafile = perfdmf_home + "/data/" + db_schemafile;

						
	    // Prompt for XML Parser jar file
	    if (configFileFound)
		System.out.print("Please enter the XML Parser jar file.\n(" + xml_parser + "):");
	    else
		System.out.print("Please enter the XML Parser jar file.\n(" + tau_root + "/" + arch + "/lib/" + xml_parser + "):");
	    tmpString = reader.readLine();
	    if (tmpString.length() > 0) xml_parser = tmpString;
	    else if (!configFileFound)
		xml_parser = perfdmf_home + "/" + arch + "/lib/" + xml_parser;


	}
	catch (IOException e) {
	    // todo - get info from the exception
	    System.out.println ("I/O Error occurred.");
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

    public void testDBTransaction() {
    }

    /** writeConfigFile method
     *  this method writes the configuration file back to 
     *  perfdmf_home/bin/perfdmf.cfg.
     **/

    public void writeConfigFile() {
	System.out.println ("\nWriting configuration file: " + configFileName + "...");
	try {
	    // Check to see if the configuration file exists
	    File configFile = new File(configFileName);
	    if (!configFile.exists()) {
		configFile.createNewFile();
	    }
	    BufferedWriter configWriter = new BufferedWriter(new FileWriter(configFile));
	    configWriter.write("# This is the configuration file for the PerfDMF tools & API\n");
	    configWriter.write("# Items are listed as name:value, one per line.\n");
	    configWriter.write("# Comment lines begin with a '#' symbol.\n");
	    configWriter.write("# DO NOT EDIT THIS FILE!  It is modified by the configure utility.\n");
	    configWriter.newLine();
						
	    configWriter.write("# PerfDMF home directory\n");
	    configWriter.write("perfdmf_home:" + perfdmf_home + "\n");
	    configWriter.newLine();
						
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
	}
	catch (IOException e) {
	}
    }
    
    //Standard access methods for some of the fields.
    public void setPerfDMFHome(String inString) {
	perfdmf_home = inString;
    }
    
    public String getPerfDMFHome() {
	return perfdmf_home;
    }
        
    public void setJDBCJarfile(String inString){
	jdbc_db_jarfile = inString;}
    public String getJDBCJarfile(){
	return jdbc_db_jarfile;}
        
    public void setJDBCDriver(String inString){
	jdbc_db_driver = inString;}
    public String getJDBCDriver(){
	return jdbc_db_driver;}

    public void setJDBCType(String inString){
	jdbc_db_type = inString;}
    public String getJDBCType(){
	return jdbc_db_type;}

    public void setDBHostname(String inString){
	db_hostname = inString;}
    public String getDBHostname(){
	return db_hostname;}
        
    public void setDBPortNum(String inString){
	db_portnum = inString;}
    public String getDBPortNum(){
	return db_portnum;}
        
    public void setDBName(String inString){
	db_dbname = inString;}
    public String getDBName(){
	return db_dbname;}
        
    public void setDBUsername(String inString){
	db_username = inString;}
    public String getDBUsername(){
	return db_username;}
        
    public void setDBPassword(String inString){
	db_password = inString;}
    public String getDBPassword(){
	return db_password;}
        
    public void setDBSchemaFile(String inString){
	db_schemafile = inString;}
    public String getDBSchemaFile(){
	return db_schemafile;}
        
    public void setXMLPaser(String inString){
	xml_parser = inString;}
    public String getXMLPaser(){
	return xml_parser;}
        
    public void setConfigFileName(String inString){
	configFileName = inString;}
    public String getConfigFileName(){
	return configFileName;}

    /* Test that the database exists, and if it doesn't, create it! */
    public void createDB() {
	ConnectionManager connector = null;
	DB db = null;
	try {
            connector = new ConnectionManager(configFileName);
            connector.connect();
            db = connector.getDB();
        } catch ( Exception e ) {
	    System.out.println("\nPlease make sure that your DBMS is configured correctly, and");
	    System.out.println("the database " + db_dbname + " has been created.");
	    System.exit(0);
        }
	try {
	    String query = new String ("select * from application;");
	    ResultSet resultSet = db.executeQuery(query);
	    resultSet.close();
	    connector.dbclose();
        } catch ( Exception e ) {
	    // build the database
	    System.out.println(configFileName);
	    connector.genParentSchema(db_schemafile);
	    connector.dbclose();
	    System.out.println("Congratulations! PerfDMF is configured and the database has been built.");
	    System.out.println("You may begin loading applications.");
        }
	System.out.println("Configuration complete.");
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {

	CmdLineParser parser = new CmdLineParser();
	CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
	CmdLineParser.Option homeOpt = parser.addStringOption('t', "tauroot");
	CmdLineParser.Option archOpt = parser.addStringOption('a', "arch");
	CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
	try {
	    parser.parse(args);
	}
	catch ( CmdLineParser.OptionException e ) {
	    System.err.println(e.getMessage());
	    System.err.println(Usage);
	    System.exit(-1);
	}

	String configFile = (String)parser.getOptionValue(configfileOpt);
	String tauroot = (String)parser.getOptionValue(homeOpt);
	String arch = (String)parser.getOptionValue(archOpt);
	Boolean help = (Boolean)parser.getOptionValue(helpOpt);

	if (help != null && help.booleanValue()) {
	    System.err.println(Usage);
	    System.exit(-1);
	}

	if (configFile == null) configFile = new String("");
	if (tauroot == null) tauroot = new String("");
	if (arch == null) arch = new String("");

	// Create a new Configure object, which will walk the user through
	// the process of creating/editing a configuration file.
	Configure config = new Configure(tauroot, arch);
	config.initialize(configFile);
				
	// Give the user the ability to modify any/everything
	config.promptForData();
				
	// Test the database connection
	//config.testDBConnection();
				
	// Test the database name/login/password, etc.
	//config.testDBTransaction();
	
	// Write the configuration file to ${PerfDMF_Home}/bin/perfdmf.cfg
	config.writeConfigFile();

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

