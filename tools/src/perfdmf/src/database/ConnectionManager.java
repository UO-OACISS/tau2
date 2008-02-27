/* 
 Name:       ConnectionManager.java
 Author:     Robert Bell, Kevin Huck
 
 
 Description:

 Things to do:
 1) Clean up password handling.
 */

package edu.uoregon.tau.perfdmf.database;

import java.io.*;
import java.net.URL;
import java.sql.SQLException;

import edu.uoregon.tau.perfdmf.Database;

public class ConnectionManager {
    private String perfdmfUser;
    private String perfdmfPass;
    private String parserClass = "org.apache.xerces.parsers.SAXParser";

    //Database schema file name. Default one should be: "~/PerfDMF/db/dbschema.txt".
    private String dbschema;

    private ParseConfig config = null;
    private DB db = null;

    private Database database;

    public ConnectionManager(Database database, String password) {
        this.database = database;
        config = database.getConfig();
        initialize(password);
    }

    public ConnectionManager(Database database, boolean prompt) {
        this.database = database;
        config = database.getConfig();
        initialize(this.getPassword(prompt));
    }

    public ConnectionManager(Database database) {
        this.database = database;
        config = database.getConfig();
        initialize(this.getPassword(false));
    }

    public void initialize(String password) {
        perfdmfUser = config.getDBUserName();
        perfdmfPass = password;
        dbschema = config.getDBSchema();
    }

    public ConnectionManager(String configFile) {
        this(new Database(configFile));
    }

    public ParseConfig getParseConfig() {
        return config;
    }

    public void connect() throws SQLException {
        setDB(new DBConnector(perfdmfUser, perfdmfPass, database));
    }

    public void connectAndCreate() throws SQLException {
        setDB(new DBConnector(perfdmfUser, perfdmfPass, database, true));
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
            BufferedReader preader;
            if (filename.toLowerCase().startsWith("http:")) {
                // When it gets converted from a String to a File http:// turns into http:/
                String url_string = "";
              if (filename.toLowerCase().startsWith("http://")) {
                url_string = "http://" + filename.toString().substring(7).replace('\\', '/');
                }
              else if (filename.toLowerCase().startsWith("http:/")) {
                url_string = "http://" + filename.toString().substring(6).replace('\\', '/');
              }
                URL url = new URL(url_string);
                InputStream iostream = url.openStream();
                InputStreamReader ireader = new InputStreamReader(iostream);
                preader = new BufferedReader(ireader);
            }  else {
                preader = new BufferedReader(new FileReader(new File(filename)));
            }

            while ((inputString = preader.readLine()) != null) {
                inputString = inputString.replaceAll("@DATABASE_NAME@", config.getDBName());
                inputString = inputString.replaceAll("@DATABASE_PREFIX@", config.getDBSchemaPrefix() + ".");
                buf.append(inputString);
                if (isEnd(db, inputString)) {
                    try {
                        if ((db.getDBType().compareTo("oracle") == 0) || (db.getDBType().compareTo("derby") == 0)
                                || (db.getDBType().compareTo("db2") == 0)) {
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
        return genParentSchema(config.getDBSchema());
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
        String tmpString = config.getDBPasswd();
        if (prompt) {
            if (tmpString == null) {
                try {
                    PasswordField passwordField = new PasswordField();
                    tmpString = passwordField.getPassword(config.getDBUserName() + "'s database password:");
                } catch (IOException ex) {
                    ex.printStackTrace();
                    System.exit(0);
                }
            }
        }
        return tmpString;
    }

}
