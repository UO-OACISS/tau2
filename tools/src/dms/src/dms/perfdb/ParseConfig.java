package dms.perfdb;

import java.io.BufferedReader;
import java.io.FileReader;

/* This class is intended to read in config.txt file and parse the parameters. */

public class ParseConfig {
    private String perfdbHome;
    private String jdbcJarFile;
    private String jdbcDriver;
    private String dbType;
    private String dbHost;
    private String dbPort;
    private String dbName;
    private String dbUserName;
    private String dbPasswd;
    private String dbSchema;
    private String xmlSAXParser;

    public ParseConfig(String configLoc){
		
	String inputString;
	String name;
	String value;

	try{
	    BufferedReader reader = new BufferedReader(new FileReader(configLoc));

	    while ((inputString = reader.readLine())!= null){
		
		if (inputString.startsWith("#") || inputString.trim().equals(""))
		    continue;
		else{
		    name = getNameToken(inputString).toLowerCase();
		    value = getValueToken(inputString);
		    if (name.equals("perfdb_home")) perfdbHome = value; 
		    else if (name.equals("jdbc_db_jarfile")) jdbcJarFile = value; 
		    else if (name.equals("jdbc_db_driver")) jdbcDriver = value;  
		    else if (name.equals("jdbc_db_type")) dbType = value; 
		    else if (name.equals("db_hostname")) dbHost = value; 
		    else if (name.equals("db_portnum")) dbPort = value; 
		    else if (name.equals("db_dbname")) dbName = value; 
		    else if (name.equals("db_username")) dbUserName = value; 
		    else if (name.equals("db_password")) dbPasswd = value; 
		    else if (name.equals("db_schemafile")) dbSchema = value; 
		    else if (name.equals("xml_sax_parser")) xmlSAXParser = value; 
		    else {
			System.out.println(name+" is not a valid configuration item."); System.exit(-1);
		    }
		}
	    }
	} catch(Exception e){
		e.printStackTrace();
		System.exit(0);
	}
    }

    public String getNameToken(String aLine){
	int i = aLine.indexOf(":");
	if (i>0)
	    return aLine.substring(0,i).trim();
	else{     
	    System.out.println(aLine);
	    System.out.println("This is an abnormal term. The correct form is name:value.");
	    return null;
	}
	
    }

    public String getValueToken(String aLine){
	int i = aLine.indexOf(":");
	if (i>0)
	    return aLine.substring(i+1).trim();
	else{     
	    System.out.println("This is an abnormal term. The correct form is name:value.");
	    return null;
	}
    }

    public String getPerfDBHome(){ return perfdbHome; }
    
    public String getJDBCJarFile(){ return jdbcJarFile; }

    public String getJDBCDriver(){ return jdbcDriver; }

    public String getDBType(){ return dbType; }

    public String getDBHost(){ return dbHost; }

    public String getDBPort(){ return dbPort; }

    public String getDBName(){ return dbName; }

    public String getDBUserName(){ return dbUserName; }

    public String getDBPasswd(){ return dbPasswd; }

    public String getDBSchema(){ return dbSchema; }

    public String getXMLSAXParser(){ return xmlSAXParser; }
    
}
