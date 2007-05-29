package edu.uoregon.tau.perfdmf.database;

import java.io.BufferedReader;
import java.io.FileReader;

import edu.uoregon.tau.common.TauRuntimeException;

/* This class is intended to read in config.txt file and parse the parameters. */

public class ParseConfig {
    private String perfdmfHome;
    private String jdbcJarFile;
    private String jdbcDriver;
    private String dbType;
    private String dbHost;
    private String dbPort;
    private String dbName;
    private String dbSchemaPrefix = "";
    private String dbUserName;
    private String dbPasswd;
    private String dbSchema;
    private String xmlSAXParser;
    private String path;
    private String name;

    public ParseConfig(String configLoc) {

        path = configLoc;
        String[] split = configLoc.split("\\.");
        name = split[split.length-1];
        if (name.compareTo("cfg") == 0)
        	name = "Default";
    	String inputString;
        String name;
        String value;

        try {
            BufferedReader reader = new BufferedReader(new FileReader(configLoc));

            // The following is for reading perfdmf.cfg out of the jar file for Java Web Start
            //        URL url = ParseConfig.class.getResource("/perfdmf.cfg");
            //        
            //        if (url == null) {
            //        }
            //            throw new IOException("Couldn't get perfdmf.cfg from the jar");
            //        BufferedReader reader = new BufferedReader(new InputStreamReader(url.openStream()));
            //        

            while ((inputString = reader.readLine()) != null) {
                inputString = inputString.trim();
                if (inputString.startsWith("#") || inputString.equals("")) {
                    continue;
                } else {
                    name = getNameToken(inputString).toLowerCase();
                    value = getValueToken(inputString);
                    if (name.equals("perfdmf_home"))
                        perfdmfHome = value;
                    else if (name.equals("jdbc_db_jarfile"))
                        jdbcJarFile = value;
                    else if (name.equals("jdbc_db_driver"))
                        jdbcDriver = value;
                    else if (name.equals("jdbc_db_type"))
                        dbType = value;
                    else if (name.equals("db_hostname"))
                        dbHost = value;
                    else if (name.equals("db_portnum"))
                        dbPort = value;
                    else if (name.equals("db_dbname"))
                        dbName = value;
                    else if (name.equals("db_schemaprefix"))
                        dbSchemaPrefix = value;
                    else if (name.equals("db_username"))
                        dbUserName = value;
                    else if (name.equals("db_password"))
                        dbPasswd = value;
                    else if (name.equals("db_schemafile"))
                        dbSchema = value;
                    else if (name.equals("xml_sax_parser"))
                        xmlSAXParser = value;
                    else {
                        System.out.println(name + " is not a valid configuration item.");
                    }
                }
            }

        } catch (Exception e) {
            // wrap it up in a runtime exception
            throw new TauRuntimeException("Unable to parse \"" + configLoc + "\"", e);
        }
    }

    public String getNameToken(String aLine) {
        int i = aLine.indexOf(":");
        if (i > 0) {
            return aLine.substring(0, i).trim();
        } else {
            return null;
        }

    }

    public String getConnectionString() {
        String connectionString;
        if (getDBType().equals("derby")) {
            connectionString = "jdbc:" + getDBType() + ":" + getDBName();
        } else {
            if (getDBType().equals("oracle")) {
                connectionString = "jdbc:oracle:thin:@//" + getDBHost() + ":" + getDBPort() + "/" + getDBName();
            } else {
                connectionString = "jdbc:" + getDBType() + "://" + getDBHost() + ":" + getDBPort() + "/"
                        + getDBName();
            }
        }
        return connectionString;
    }

    public String getValueToken(String aLine) {
        int i = aLine.indexOf(":");
        if (i > 0) {
            return aLine.substring(i + 1).trim();
        } else {
            return null;
        }
    }

    public String getPerfDMFHome() {
        return perfdmfHome;
    }

    public String getJDBCJarFile() {
        return jdbcJarFile;
    }

    public String getJDBCDriver() {
        return jdbcDriver;
    }

    public String getDBType() {
        return dbType;
    }

    public String getDBHost() {
        return dbHost;
    }

    public String getDBPort() {
        return dbPort;
    }

    public String getDBName() {
        return dbName;
    }

    public String getDBSchemaPrefix() {
        return dbSchemaPrefix;
    }

    public String getDBUserName() {
        return dbUserName;
    }

    public String getDBPasswd() {
        return dbPasswd;
    }

    public String getDBSchema() {
        return dbSchema;
    }

    public String getXMLSAXParser() {
        return xmlSAXParser;
    }
    public String getName()
    {
    	return name;
    }
    public String getPath()
    {
    	return path;
    }

}
