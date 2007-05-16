/* 
 Name:       ConnectionManager.java
 Author:     Robert Bell, Kevin Huck
 
 
 Description:

 Things to do:
 1) Clean up password handling.
 */

package edu.uoregon.tau.perfdmf.database;

import java.io.*;
import java.sql.SQLException;

public class ConnectionManager {
    private String perfdmfUser;
    private String perfdmfPass;
    private String parserClass = "org.apache.xerces.parsers.SAXParser";

    //Database schema file name. Default one should be: "~/PerfDMF/db/dbschema.txt".
    private String dbschema;

    private ParseConfig parser = null;
    private DB db = null;

    public ConnectionManager(ParseConfig prs, boolean prompt)
    {
    	parser = prs;
    	initialize(this.getPassword(prompt));
    }
    public ConnectionManager(ParseConfig prs, String password)
    {
    	parser = prs;
    	initialize(password);
    }
    public ConnectionManager(String configFileName) {
        parser = new ParseConfig(configFileName);
        initialize(this.getPassword());
    }

    public ConnectionManager(String configFileName, String password) {
        parser = new ParseConfig(configFileName);
        initialize(password);
    }

    public ConnectionManager(String configFileName, boolean prompt) {
        parser = new ParseConfig(configFileName);
        initialize(this.getPassword(prompt));
    }
    
    public ConnectionManager(ParseConfig config) {
        parser = config;
        initialize(this.getPassword(false));
    }

    public void initialize(String password) {
        perfdmfUser = parser.getDBUserName();
        perfdmfPass = password;
        dbschema = parser.getDBSchema();
    }

    public ParseConfig getParseConfig() {
        return parser;
    }

    public void connect() throws SQLException {
        setDB(new DBConnector(perfdmfUser, perfdmfPass, parser));
    }

    public void connectAndCreate() throws SQLException {
        setDB(new DBConnector(perfdmfUser, perfdmfPass, parser, true));
    }

    public String getParserClass() {
        return parserClass;
    }

    public String getSchemafile() {
        return dbschema;
    }

    public void dbclose() {
        try {
            if (db != null) {
                db.close();
            }
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public void setParserClass(String parserClass) {
        this.parserClass = parserClass;
    }

    public void setDB(DB db) {
        this.db = db;
    }

    public DB getDB() {
        return db;
    }

    public void setSchemafile(String dbschema) {
        this.dbschema = dbschema;
    }

    /*** This method loads database schema. Be sure to load it ONLY once. ***/

    public int genParentSchema(String filename) {
        File readSchema = new File(filename);
        String inputString;
        StringBuffer buf = new StringBuffer();

        if (readSchema.exists()) {
            System.out.println("Found " + filename + "  ... Loading");
        } else
            System.out.println("Did not find " + filename);

        try {
            BufferedReader preader = new BufferedReader(new FileReader(readSchema));

            while ((inputString = preader.readLine()) != null) {
                inputString = inputString.replaceAll("@DATABASE_NAME@", parser.getDBName());
                inputString = inputString.replaceAll("@DATABASE_PREFIX@", parser.getDBSchemaPrefix() + ".");
                buf.append(inputString);
                if (isEnd(db, inputString)) {
                    try {
                        if ((db.getDBType().compareTo("oracle") == 0) ||
                        	(db.getDBType().compareTo("derby") == 0) ||
                        	(db.getDBType().compareTo("db2") == 0)) {
                            buf.delete(buf.length() - 1, buf.length());
                        }
                        //System.out.println ("line: " + buf.toString());
                        getDB().executeUpdate(buf.toString());
                        buf = buf.delete(0, buf.length());
                    } catch (SQLException ex) {
						System.out.println(buf.toString());
                        ex.printStackTrace();
                        return -1;
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            return -1;
        }
        return 0;
    }

    public int genParentSchema() {
        return genParentSchema(parser.getDBSchema());
    }

    public static boolean isEnd(DB db, String st) {

        if (db.getDBType().compareTo("oracle") == 0) {
            return st.trim().endsWith("/");
        } else {
            return st.trim().endsWith(";");
        }
    }

    //For backward compatibility, simulate default parameter.
    public String getPassword() {
        return this.getPassword(true);
    }

    public String getPassword(boolean prompt) {
        String tmpString = parser.getDBPasswd();
        if (prompt) {
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
