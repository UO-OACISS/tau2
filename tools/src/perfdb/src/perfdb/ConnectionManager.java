package perfdb;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.sql.SQLException;
import java.io.IOException;

import perfdb.util.dbinterface.DB;
import perfdb.util.dbinterface.DBConnector;
import perfdb.util.dbinterface.JDBCAcct;
import perfdb.util.dbinterface.ParseConfig;

/*** Some default setups for system ***/

public class ConnectionManager {
	
    //perfdbAcct is the DBMS account 
    private String perfdbAcct;
    //public static String perfdbAcct = "user=lili;password=********";

    private String parserClass = "org.apache.xerces.parsers.SAXParser";
	
    // database schema file name. default one should be "~/PerfDB/db/dbschema.txt".
    private String dbschema;

	private ParseConfig parser = null;

    private DB db = null;
	
    public ConnectionManager(String configFileName) {
		super();
		parser = new ParseConfig(configFileName);
		if (parser.getDBType().compareTo("mysql") == 0)
			perfdbAcct = "user=" + parser.getDBUserName() + "&password=" + getPassword();	
		else
			perfdbAcct = "user=" + parser.getDBUserName() + ";password=" + getPassword();	
		dbschema = parser.getDBSchema();
    }

	public ParseConfig getParseConfig () {
			return parser;
	}

    public void connect() {
        try {
	    setDB(new DBConnector(new JDBCAcct(perfdbAcct), parser));
        } catch (java.sql.SQLException ex) {
	    ex.printStackTrace();
        }
    }

    public void connectTest() throws SQLException {
	    setDB(new DBConnector(new JDBCAcct(perfdbAcct), parser, true));
    }

    public void connect(String value) {
        setPerfdbAcct(value);
        connect();
    }


    public String getParserClass() {
	return parserClass;
    }

    public String getPerfdbAcct() {
	return perfdbAcct;
    }

    public String getSchemafile() {
	return dbschema;
    }

    public void dbclose(){
	try{
	    if (db != null) {
		db.close();
	    }
	}catch (Throwable e) {
	    e.printStackTrace();
	}
    }

    public void setParserClass(String newValue) {
	parserClass = newValue;
    }

    public void setPerfdbAcct(String newValue) {
	perfdbAcct = newValue;
    }

    public void setDB(DB newValue) {
        db = newValue;
    }

    public DB getDB(){
	return db; 
    }

    public void setSchemafile(String filename) {
        dbschema = filename;
    }

    /*** This method loads database schema. Be sure to load it ONLY once. ***/

    public void genParentSchema(String filename){
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

	public String getPassword () {
		String tmpString = new String ("");
		try {
			PasswordField passwordField = new PasswordField();
			tmpString = passwordField.getPassword("Please enter database password:");
		} catch (IOException ex) {
	    	ex.printStackTrace();
			System.exit(0);
		}
		return tmpString;
	}

}
