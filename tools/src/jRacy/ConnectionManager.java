package jRacy;

import java.io.*;
import java.sql.*;

/*** Some default setups for system ***/

public class ConnectionManager {
	
    //perfdbAcct is the DBMS account 
    public static String perfdbAcct = null;//"user=lili;password=lkll0624";

    public static String parserClass = "org.apache.xerces.parsers.SAXParser";
	
    // default database schema file name
    public static String dbschema = "/scratch1/userSpace/lili/PerfDBDeliverable/db/dbschema.txt";
    public static DB db = null;
    public static int callCounter = 0;
	
    public ConnectionManager() {
	super();
    }

    public static boolean connect(String inServerAddress, String inUsername, String inPassword) {
        try {
        //Set perfdbAcct.
        perfdbAcct = "user=" + inUsername + ";password=" + inPassword;
	    ConnectionManager.setDB(new DBConnector(new JDBCAcct(perfdbAcct), inServerAddress));
	    return true;
        } catch (java.sql.SQLException ex) {
	    ex.printStackTrace();
	    return false;
        }
    }

    public static void connect(String value, String inServerAddress, String inUsername, String inPassword) {
        ConnectionManager.setPerfdbAcct(value);
        ConnectionManager.connect(inServerAddress, inUsername, inPassword);
    }


    public static String getParserClass() {
	return parserClass;
    }

    public static String getPerfdbAcct() {
	return perfdbAcct;
    }

    public String getSchemafile() {
	return dbschema;
    }

    public static void dbclose(){
	try{
	    if (db != null) {
		db.close();
	    }
	}catch (Throwable e) {
	    e.printStackTrace();
	}
    }

    public static void setParserClass(String newValue) {
	ConnectionManager.parserClass = newValue;
    }

    public static void setPerfdbAcct(String newValue) {
	ConnectionManager.perfdbAcct = newValue;
    }

    public static void setDB(DB newValue) {
        ConnectionManager.db = newValue;
    }

    public static DB getDB(){
	return db; 
    }

    public static void setSchemafile(String filename) {
        ConnectionManager.dbschema = filename;
    }

    /*** This method loads database schema. Be sure to load it ONLY once. ***/

    public static void genSchema(String filename){
	File readSchema = new File(filename);
	String inputString;
	StringBuffer buf = new StringBuffer();

	if (readSchema.exists()){
            System.out.println("Found " + filename + "  ... Loading");
        }
        else System.out.println("Did not find " +  filename);

	try{	
	    BufferedReader preader = new BufferedReader(new FileReader(readSchema));	

	    while ((inputString = preader.readLine())!= null){
		buf.append(inputString);
		if (isEnd(inputString)) {
		    try {
			getDB().executeUpdate(buf.toString());
			buf = buf.delete(0,buf.length());
		    } catch (SQLException ex) {
			ex.printStackTrace();
		    }				
		}		
	    }
	}catch (Exception e){
	    e.printStackTrace();
	}

    }

    public static boolean isEnd(String st){
	return st.trim().endsWith(";");
    }

}
