package perfdb.configure;

import perfdb.util.dbinterface.*;
import perfdb.util.io.*;
import perfdb.dbmanager.*;
import java.io.*;
import java.net.*;
import java.sql.*;

public class Configure {
    private DB db = null;
    // protected String dbAccessString = perfdb.ConnectionManager.getPerfdbAcct();
	private static String Usage = "Usage: configure config_file";
	private static String Greeting = "\nWelcome to the configuration program for PerfDBF.\nThis program will prompt you for some information necessary to ensure\nthe desired behavior for the PerfDB tools.\n";
	private static String PDBHomePrompt = "Please enter the PerfDB home directory:";

	// todo - remove these defaults
	// todo - consider using a hash table!
	private String perfdb_home = "/home/khuck/research/PerfDB";
	private String jdbc_db_jarfile = "/home/khuck/research/PerfDB/jars/postgresql.jar";
	private String jdbc_db_driver = "org.postgresql.Driver";
	private String jdbc_db_type = "postgresql";
	private String db_hostname = "localhost";
	private String db_portnum = "5432";
	private String db_dbname = "perfdb";
	private String db_username = "khuck";
	private String db_password = "encrypted_password";
	private String db_schemafile = "/home/khuck/research/PerfDB/db/dbschema.txt";
	private String xml_parser = "/home/khuck/research/PerfDB/jars/xerces.jar";

	private String configFileName;

    public Configure() {
	super();
    }

    public void errorPrint(String msg) {
	System.err.println(msg);
    }

	/** Initialize method 
	 *  This method will welcome the user to the program, and prompt them
	 *  for some basic information. 
	 **/

    public void Initialize(String configFileNameIn) {
		// Create a reader to parse the input
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

		// Welcome the user to the program
		System.out.println(Greeting);

		try {
				/*
			// Prompt for home directory
			System.out.println(PDBHomePrompt);
			// todo - the default is pwd/..
			String tmpString = reader.readLine();
			if (tmpString.length() > 0) perfdb_home = tmpString;
			*/

			// Check to see if the configuration file exists
			configFileName = configFileNameIn;
			File configFile = new File(configFileName);
			if (configFile.exists()) {
				System.out.println("Configuration file found...");
				// Parse the configuration file
				ParseConfigFile();
			} else {
				System.out.println("Configuration file NOT found...");
				// If it doesn't exist, explain that the program looks for the 
				// configuration file in ${PerfDB_Home}/bin/perfdb.cfg
				// Since it isn't there, create a new one.
			}
		}
		catch (IOException e) {
			// todo - get info from the exception
			System.out.println ("I/O Error occurred.");
		}
	}
	
	/** ParseConfigFile method 
	 *  This method opens the configuration file for parsing, and passes
	 *  each data line to the ParseConfigField() method.
	 **/

    public void ParseConfigFile() throws IOException, FileNotFoundException {
		System.out.println("Parsing config file...");
	    BufferedReader configReader = new BufferedReader(new FileReader(configFileName));
		String inputString;
		// parse the file, line by line.
	    while ((inputString = configReader.readLine())!= null){
			// ignore blank lines
			if (inputString.length() > 0) {
				// System.out.println(inputString);
				// ignore comment lines
				// todo - find out why tokenizing won't compile!
				/*
		       	StringTokenizer checkCommentTokenizer = new StringTokenizer(inputString,"#");
				inputString = checkCommentTokenizer.nextToken();
				if (inputString[0] != '#') {
					ParseConfigField(inputString);
				}
				*/
			}
		}
	}

	/** ParseConfigField method
	 *  This method tokenizes a single line from the configuration file,
	 *  and stores the data in the appropriate member variable.
	 */

	/*
    public void ParseConfigField(String inputString) {
		StringTokenizer checkCommentTokenizer = new StringTokenizer(inputString,":");
		String fieldName = checkCommentTokenizer.nextToken();
		if (fieldName.compareTo("perfdb_home") == 0) {
			perfdb_home = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("jdbc_db_jarfile") == 0) {
			jdbc_db_jarfile = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("jdbc_db_driver") == 0) {
			jdbc_db_driver = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("jdbc_db_type") == 0) {
			jdbc_db_type = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("db_hostname") == 0) {
			db_hostname = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("db_portnum") == 0) {
			db_portnum = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("db_dbname") == 0) {
			db_dbname = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("db_username") == 0) {
			db_username = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("db_password") == 0) {
			db_password = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("db_schemafile") == 0) {
			db_schemafile = checkCommentTokenizer.nextToken();
		} else if (fieldName.compareTo("xml_parser") == 0) {
			xml_parser = checkCommentTokenizer.nextToken();
		} else {
			System.out.println("Unknown data value " + fieldName + " in configuration file.\nIt will be ignored.");
		}

	}
	*/

	/** PromptForData method
	 *  This method prompts the user for each of the data fields
	 *  in the configuration file.
	 */
    public void PromptForData() {
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		String tmpString;

		System.out.println("\nYou will now be prompted for new values, if desired.  The current or default\nvalues for each prompt are shown in parenthesis.  To accept the default, just\npress Enter/Return.\n");
		try {
			// Prompt for XML parsing jar file
			System.out.print("Please enter the new PerfDB Home directory.\n(" + perfdb_home + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) perfdb_home = tmpString;

			// Prompt for JDBC jar file
			System.out.print("Please enter the JDBC jar file.\n(" + jdbc_db_jarfile + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) jdbc_db_jarfile = tmpString;

			// Prompt for JDBC driver name
			System.out.print("Please enter the JDBC Driver name.\n(" + jdbc_db_driver + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) jdbc_db_driver = tmpString;

			// Prompt for database type
			System.out.print("Please enter the database vendor.\n(" + jdbc_db_type + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) jdbc_db_type = tmpString;

			// Prompt for database hostname
			System.out.print("Please enter the hostname for the database server.\n(" + db_hostname + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) db_hostname = tmpString;

			// Prompt for database portnumber
			System.out.print("Please enter the port number for the database JDBC connection.\n(" + db_portnum + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) db_portnum = tmpString;

			// Prompt for database name
			System.out.print("Please enter the database name.\n(" + db_dbname + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) db_dbname = tmpString;

			// Prompt for database username
			System.out.print("Please enter the database username.\n(" + db_username + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) db_username = tmpString;

			boolean passwordMatch = false;
			while (!passwordMatch) {
				// Prompt for database password
				System.out.println("NOTE: Passwords will be stored in an encrypted format.");
				System.out.print("Please enter the database password\n(default not shown):");
				tmpString = reader.readLine();
				if (tmpString.length() > 0) db_password = tmpString;
				System.out.print("Please enter the database password again to confirm:");
				String tmpString2 = reader.readLine();
				if (tmpString.compareTo(tmpString2) == 0) {
					db_password = tmpString;
					passwordMatch = true;
				}
				else System.out.println ("Password confirmation failed.  Please try again.");
			}

			// Prompt for database schema file
			System.out.print("Please enter the PerfDBF schema file.\n(" + db_schemafile + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) db_schemafile = tmpString;

			// Prompt for XML Parser jar file
			System.out.print("Please enter the XML Parser jar file.\n(" + xml_parser + "):");
			tmpString = reader.readLine();
			if (tmpString.length() > 0) xml_parser = tmpString;
		}
		catch (IOException e) {
			// todo - get info from the exception
			System.out.println ("I/O Error occurred.");
		}
	}

	/** TestDBConnection method
	 *  this method attempts to connect to the database.  If it cannot 
	 *  connect, it gives the user an error.  This method is intended
	 *  to test the JDBC driver, servername, portnumber.
	 */

	public void TestDBConnection() {
	// perfdb.ConnectionManager.connect();
	// perfdb.ConnectionManager.dbclose();
	}

	/** TestDB method
	 *  this method attempts to connect to the database.  If it cannot 
	 *  connect, it gives the user an error.  This method is intended
	 *  to test the username, password, and database name.
	 */

	public void TestDBTransaction() {
	}

	/** WriteConfigFile method
	 *  this method writes the configuration file back to 
	 *  perfdb_home/bin/perfdb.cfg.
	 */

	public void WriteConfigFile() {
		System.out.println ("\nWriting configuration file: " + configFileName + "...");
		try {
			// Check to see if the configuration file exists
			File configFile = new File(configFileName);
			if (!configFile.exists()) {
				configFile.createNewFile();
			}
			BufferedWriter configWriter = new BufferedWriter(new FileWriter(configFile + ".output"));
			configWriter.write("# This is the configuration file for the PerfDBF tools & API\n");
			configWriter.write("# Items are listed as name:value, one per line.\n");
			configWriter.write("# Comment lines begin with a '#' symbol.\n");
			configWriter.write("# DO NOT EDIT THIS FILE!  It is modified by the configure utility.\n");
			configWriter.newLine();

			configWriter.write("# PerfDB home directory\n");
			configWriter.write("perfdb_home:" + perfdb_home + "\n");
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

			configWriter.write("# Database username\n");
			configWriter.write("db_username:" + db_username + "\n");
			configWriter.newLine();

			configWriter.write("# Database password\n");
			configWriter.write("db_password:" + db_password + "\n");
			configWriter.newLine();

			configWriter.write("# Database Schema file - note: the path is absolulte\n");
			configWriter.write("db_schemafile:" + db_schemafile + "\n");
			configWriter.newLine();

			configWriter.write("# Database XML parser jar file - note: the path is absolulte\n");
			configWriter.write("xml_parser:" + xml_parser + "\n");
			configWriter.newLine();

			configWriter.close();
		}
		catch (IOException e) {
		}
	}

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
	
	if (args.length == 0) {
	    System.err.println(Usage);
	    System.exit(-1);
        }

	// Create a new Configure object, which will walk the user through
	// the process of creating/editing a configuration file.
	Configure config = new Configure();
	config.Initialize(args[0]);

	// Give the user the ability to modify any/everything
	config.PromptForData();

	// Test the database connection
	config.TestDBConnection();

	// Test the database name/login/password, etc.
	config.TestDBTransaction();

	// Write the configuration file to ${PerfDB_Home}/bin/perfdb.cfg
	config.WriteConfigFile();
    }
}

