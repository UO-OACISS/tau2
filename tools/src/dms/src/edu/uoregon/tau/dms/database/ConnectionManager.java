package edu.uoregon.tau.dms.database;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.sql.SQLException;
import java.io.IOException;

/*** Some default setups for system ***/

public class ConnectionManager {
	
    //perfdmfAcct is the DBMS account 
    private String perfdmfUser;
    private String perfdmfPass;

    private String parserClass = "org.apache.xerces.parsers.SAXParser";
	
    // database schema file name. default one should be "~/PerfDMF/db/dbschema.txt".
    private String dbschema;

	private ParseConfig parser = null;

    private DB db = null;

    public ConnectionManager(String configFileName){
	super();
	parser = new ParseConfig(configFileName);
	String password = getPassword();
	initialize(password);
	//System.out.println("\r\n");
    }
	
    public ConnectionManager(String configFileName, String password) {
	super();
	parser = new ParseConfig(configFileName);
	initialize(password);
    }
    
    public ConnectionManager(String configFileName, boolean prompt){
	super();
	parser = new ParseConfig(configFileName);
	String password = getPassword(prompt);
	initialize(password);
	//System.out.println("\r\n");
    }

    public void initialize(String password) {
		perfdmfUser = parser.getDBUserName();
		perfdmfPass = password;
		dbschema = parser.getDBSchema();
    }
    
    public ParseConfig getParseConfig () {
	return parser;
    }

    public void connect() throws SQLException {
	    setDB(new DBConnector(perfdmfUser, perfdmfPass, parser));
    }

    public String getParserClass() {
	return parserClass;
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
				inputString = inputString.replaceAll("@DATABASE_NAME@", parser.getDBName());
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

	public void genParentSchema() {
		genParentSchema(parser.getDBSchema());
	}

    public static boolean isEnd(String st){
	return st.trim().endsWith(";");
    }

    //For backward compatibility, simulate default parameter.
    public String getPassword (){
	return this.getPassword(true);}

    public String getPassword (boolean prompt){
	String tmpString = parser.getDBPasswd();	
	if(prompt){
	    if (tmpString == null) {
		try {
		    PasswordField passwordField = new PasswordField();
		    tmpString = passwordField.getPassword(parser.getDBUserName() + "'s database password:");
		} catch (IOException ex) {
		    ex.printStackTrace();
		    System.exit(0);
		}
	    }
	}
	return tmpString;
    }

}
