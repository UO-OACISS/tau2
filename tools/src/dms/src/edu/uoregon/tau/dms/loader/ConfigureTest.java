
/**
 *
 * This file is basically the same thing as Configure.java
 * It was created so that the regular Configure can get the JDBC jar file
 * and write it into the script, this one will use that jar file (from the classpath).
 * An alternative is to use our own class loader.
 *
*/


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
import java.io.FileReader;

public class ConfigureTest {
    private DB db = null;
    private static String Usage = "Usage: configure [{-h,--help}] [{-g,--configfile} filename] [{-t,--tauroot} path]";
    private static String Greeting = "\nNow testing your database connection.\n";
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
    private String db_schemafile = "dbschema.txt";
    private String xml_parser = "xerces.jar";
    private ParseConfig parser;
    private boolean configFileFound = false;
		
    private String configFileName;

    public ConfigureTest(String tauroot, String arch) {
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
		//System.out.println("Configuration file found...");
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
	//System.out.println("Parsing config file...");
	parser = new ParseConfig(configFileName);
	perfdmf_home = parser.getPerfDMFHome();
	jdbc_db_jarfile = parser.getJDBCJarFile();
	jdbc_db_driver = parser.getJDBCDriver();
	jdbc_db_type = parser.getDBType();
	db_hostname = parser.getDBHost();
	db_portnum = parser.getDBPort();
	db_dbname = parser.getDBName();
	db_username = parser.getDBUserName();
	db_password = parser.getDBPasswd();
	db_schemafile = parser.getDBSchema();
	xml_parser = parser.getXMLSAXParser();
    }

    
    //Standard access methods for some of the fields.
    public void setPerfDMFHome(String inString){
	perfdmf_home = inString;}
    public String getPerfDMFHome(){
	return perfdmf_home;}
        
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
	    if (db_password != null) {
		connector = new ConnectionManager(configFileName, db_password);
	    } else {
		connector = new ConnectionManager(configFileName);
	    }
            connector.connect();
	    System.out.println();
            db = connector.getDB();
        } catch ( Exception e ) {
	    System.out.println("\nPlease make sure that your DBMS is configured correctly, and");
	    System.out.println("the database " + db_dbname + " has been created.");
	    System.exit(0);
        }

	try {
	    String query = new String ("SELECT version FROM version;");
	    ResultSet resultSet = db.executeQuery(query);
	    
	    String version = "none";

	    while (resultSet.next() != false) {
		version = resultSet.getString(1);
	    }
	    
	    
	    if (!version.equals("2.13.7")) {
		// they're using a newer version
		System.out.println("Warning: Expected database schema version 2.13.7, but found " + version);
		System.out.println("Things may not work correctly!");
	    }

 	    resultSet.close();
	    connector.dbclose();

	} catch (Exception e) {
	    
	    e.printStackTrace();
	    

	    // The schema does not have the version table, it must be older than 2.13.7 (or whatever comes after 2.13.6)

	    try {
		String query = new String ("SELECT * FROM application;");
		ResultSet resultSet = db.executeQuery(query);
		
		// if we got here (i.e. no exception) then the schema in the database must be the old one

		if (jdbc_db_type.equals("postgresql")) {
		

		    BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
    
		    System.out.print("Warning: Old database schema found.  It is recommended that you upgrade\nthe schema, would you like to do this now? [y/n]: ");
		    
		    String input = reader.readLine();
		    if (input.equals("y") || input.equals("Y")) {

			try {
			    String upgradeSchemaFile = tau_root + "/tools/src/dms/data/conversion-2.13.7.sql";
			    
			    File readSchema = new File(upgradeSchemaFile);
			    String inputString;
			    StringBuffer buf = new StringBuffer();
			    
			    if (!readSchema.exists()){
				System.out.println("Could not find " +  upgradeSchemaFile);
				return;
			    } else {
				System.out.println("Found " + upgradeSchemaFile + "\nUpgrading database schema ... ");
			    
				try{	
				    BufferedReader preader = new BufferedReader(new FileReader(readSchema));	
				    
				    while ((inputString = preader.readLine())!= null){
					inputString = inputString.replaceAll("@DATABASE_NAME@", parser.getDBName());
					buf.append(inputString);
				
					if (inputString.trim().endsWith(";")) {
					    try {
						connector.getDB().execute(buf.toString());
						buf = buf.delete(0,buf.length());
					    } catch (SQLException ex) {
						ex.printStackTrace();
					    }				
					}		
				    }
				    
				    System.out.println("Successfully upgraded schema");
				    
				} catch (Exception h) {
				    h.printStackTrace();
				}
			    }
			    
			    //connector.genParentSchema(upgradeSchemaFile);
			} catch (Exception g) {
			    g.printStackTrace();
			    return;
			}
		    }
		    
		} else {
		    
		    // what else can we do?!
		    System.out.println("Warning: Old database schema found, things may not work correctly!");
		    
		}

		resultSet.close();
		connector.dbclose();
		
		
	    } catch ( Exception f ) {
		// no schema in the database (or at least no version/application tables)
		// build the database
		System.out.println("Uploading Schema: " + configFileName);
		connector.genParentSchema(db_schemafile);
		System.out.println("Successfully uploaded schema\n");
		connector.dbclose();
	    }
	    
	    
	    
	}

	System.out.println("Database connection successful.");
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
	ConfigureTest config = new ConfigureTest(tauroot, arch);

	config.tau_root = tauroot;

	config.initialize(configFile);
	config.createDB();
    }
}

