package edu.uoregon.tau.perfdmf.database;

import java.io.File;
import java.net.URL;
import java.net.URLClassLoader;
import java.sql.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import edu.uoregon.tau.perfdmf.Database;

/*******************************************************
 * Implements access to a database
 * Be sure to specify JDBC Driver in the class path.
 *
 * Drivers are registered with the driverName.
 * connection string = dbaddress
 *******************************************************/

public class DBConnector implements DB {

    private Statement statement;
    private Connection conn;
    private ParseConfig config;

    private String dbaddress;
    // it looks like "jdbc:postgresql://zeta:5432/perfdmf;" in PostgreSQL.
    private String dbUser;
    private String dbPassword;

    private String driverName;
    private String JDBCjarFileName;

    private Database database;

    
    private static Map passwordMap = new HashMap();
    
    private static PasswordCallback passwordCallback;
    
    
    
    /*
     * This class is here because the DriverManager refuses to use a driver that is not loaded
     * by the system ClassLoader.  So we wrap it with this.
     * From: http://www.kfu.com/~nsayer/Java/dyn-jdbc.html
     */
    public static class DriverShim implements Driver {
        private Driver driver;

        DriverShim(Driver d) {
            this.driver = d;
        }

        public boolean acceptsURL(String u) throws SQLException {
            return this.driver.acceptsURL(u);
        }

        public Connection connect(String u, Properties p) throws SQLException {
            return this.driver.connect(u, p);
        }

        public int getMajorVersion() {
            return this.driver.getMajorVersion();
        }

        public int getMinorVersion() {
            return this.driver.getMinorVersion();
        }

        public DriverPropertyInfo[] getPropertyInfo(String u, Properties p) throws SQLException {
            return this.driver.getPropertyInfo(u, p);
        }

        public boolean jdbcCompliant() {
            return this.driver.jdbcCompliant();
        }
    }

    // it should be "org.postgresql.Driver" in PostgreSQL.

    //    public DBConnector(ParseConfig parser) throws SQLException {
    //        super();
    //        parseConfig = parser;
    //        setJDBC(parser);
    //        register();
    //    }

    public DBConnector(String user, String password, Database database) throws SQLException {
        this.database = database;
        config = database.getConfig();
        setJDBC(config);
        register();
        connect(user, password);
    }

    public DBConnector(String user, String password, Database database, boolean createDatabase) throws SQLException {
        this.database = database;
        config = database.getConfig();
        setJDBC(config);
        register();
        if (createDatabase) {
            connectAndCreate(user, password);
        }
    }

    public void setJDBC(ParseConfig parser) {
        driverName = parser.getJDBCDriver();
        dbaddress = parser.getConnectionString();
        JDBCjarFileName = parser.getJDBCJarFile();
    }

    public void close() {
        try {
            if (conn.isClosed()) {
                return;
            } else {
                conn.close();
            }
        } catch (SQLException ex) {
            // ugh
            ex.printStackTrace();
        }
    }

    public void setAutoCommit(boolean auto) throws SQLException {
        //System.out.println ("setting AutoCommit to " + auto);
        conn.setAutoCommit(auto);
    }

    public void commit() throws SQLException {
        conn.commit();
    }

    public void rollback() throws SQLException {
        conn.rollback();
    }
    
    private static String findPassword(ParseConfig config) {

        //System.out.println("finding password, path: " + config.getPath());
    	String password = (String) passwordMap.get(config.getPath());
        if (password == null && passwordCallback != null) {
            password = passwordCallback.getPassword(config);
            passwordMap.put(config.getPath(), password);
        }
        return password;
    }

    public boolean connect(String user, String password) throws SQLException {
        String cs = "";
        try {
            if (conn != null) {
                return true;
            }
            cs = getConnectString();

            if (password == null) {
                password = findPassword(config);
            }
            conn = DriverManager.getConnection(cs, user, password);
            return true;
        } catch (SQLException ex) {
            System.err.println("Cannot connect to server.");
            System.err.println("Connection String: " + cs);
            System.err.println("Exception Message: " + ex.getMessage());
            throw ex;
        }
    }

    public void connectAndCreate(String user, String password) throws SQLException {
        StringBuffer cs = new StringBuffer();
        try {
            cs.append(getConnectString());
            cs.append(";create=true");
            
            if (password == null) {
                password = findPassword(config);
            }
            conn = DriverManager.getConnection(cs.toString(), user, password);
            conn.close();
            System.out.println("Database created, command: " + cs.toString());
        } catch (SQLException ex) {
            System.err.println("Cannot create database.");
            System.err.println("Connection String: " + cs);
            System.err.println("Exception Message: " + ex.getMessage());
            throw ex;
        }
        return;
    }

    /*** Execute a SQL statement that returns a single ResultSet object. ***/

    public ResultSet executeQuery(String query) throws SQLException {
        if (statement == null) {
            if (conn == null) {
                System.err.println("Database is closed for " + query);
                return null;
            }
            statement = conn.createStatement();
        }
        //        conn.setAutoCommit(false);
        //        statement.setFetchSize(100);
        //	System.out.println ("executing query: " + query.trim());
        return statement.executeQuery(query.trim());
    }

    /*** Execute a SQL statement that may return multiple results. ***/

    public boolean execute(String query) throws SQLException {
        if (statement == null) {
            if (conn == null) {
                System.err.println("Database is closed for " + query);
                return false;
            }
            statement = conn.createStatement();
        }
        return statement.execute(query.trim());
    }

    /*** Execute a SQL statement that does not return
     the number of rows modified, 0 if no result returned.***/

    public int executeUpdate(String sql) throws SQLException {
        //	try {
        if (statement == null) {
            if (conn == null) {
                System.err.println("Database is closed for " + sql);
                return 0;
            }
            statement = conn.createStatement();
        }
        //	    System.out.println ("sql: " + sql);
        return statement.executeUpdate(sql.trim());
        //	} catch (SQLException ex) {
        //	    ex.printStackTrace();
        //	    return 0;
        //	}
    }

    public java.sql.Connection getConnection() {
        return conn;
    }

    public String getConnectString() {
        return dbaddress;
    }

    /*** Get the first returned value of a query. ***/

    public String getDataItem(String query) throws SQLException {
        //returns the value of the first column of the first row

        ResultSet resultSet = executeQuery(query);
        if (resultSet.next() == false) {
            resultSet.close();
            return null;
        } else {
            String result = resultSet.getString(1);
            resultSet.close();
            return result;
        }

    }

    /*** Check if the connection to database is closed. ***/

    public boolean isClosed() {
        if (conn == null) {
            return true;
        } else {
            try {
                return conn.isClosed();
            } catch (SQLException ex) {
                ex.printStackTrace();
                return true;
            }
        }
    }

    //registers the driver
    public void register() {
        try {
            // We now load the jar file dynamically based on the filename
            // in the perfdmf configuration

            URL[] urls = new URL[1];

            if (JDBCjarFileName.toLowerCase().startsWith("http:")) {
                // When it gets converted from a String to a File http:// turns into http:/
                String url_string = "";
              if (JDBCjarFileName.toLowerCase().startsWith("http://")) {
                url_string = "http://" + JDBCjarFileName.toString().substring(7).replace('\\', '/');
                }
              else if (JDBCjarFileName.toLowerCase().startsWith("http:/")) {
                url_string = "http://" + JDBCjarFileName.toString().substring(6).replace('\\', '/');
              }
              urls[0] = new URL(JDBCjarFileName);
            }  else {
                File file = new File(JDBCjarFileName);

                if (!file.exists()) {
                    System.err.println("Warning: file '" + JDBCjarFileName + "' does not exist!");
                }

	            if (System.getProperty("os.name").toLowerCase().trim().startsWith("windows")) {
	                urls[0] = new URL("file:\\" + JDBCjarFileName.replace('\\', '/'));
	            } else {
	                urls[0] = new URL("file://" + JDBCjarFileName);
	            }
            }

            URLClassLoader cl = new URLClassLoader(urls);
            try {
                Class drvCls = Class.forName(driverName, true, cl);
                Driver driver = (Driver) drvCls.newInstance();
                DriverManager.registerDriver(new DriverShim(driver));
            } catch (ClassNotFoundException cnfe) {
                try {
                    Class.forName(driverName).newInstance();
                } catch (ClassNotFoundException cnfe2) {
                    System.err.println("Unable to load driver '" + driverName + "' from " + JDBCjarFileName);
                }
            }

            // old method
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public DatabaseMetaData getMetaData() throws SQLException {
        return conn.getMetaData();
    }

    public void setDBAddress(String newValue) {
        this.dbaddress = newValue;
    }

    public String getDBType() {
        return new String(this.config.getDBType());
    }

    public String getSchemaPrefix() {
        if (this.getDBType().compareTo("oracle") == 0) {

            if (this.config.getDBSchemaPrefix() != null && this.config.getDBSchemaPrefix().compareTo("") != 0)
                return new String(this.config.getDBSchemaPrefix() + ".");
            else
                return "";
        } else if (this.getDBType().compareTo("db2") == 0) {

            if (this.config.getDBSchemaPrefix() != null && this.config.getDBSchemaPrefix().compareTo("") != 0)
                return new String(this.config.getDBSchemaPrefix() + ".");
            else
                return "";
        } else {
            return "";
        }
    }

    public PreparedStatement prepareStatement(String statement) throws SQLException {
        //System.out.println("statement = " + statement);
        return getConnection().prepareStatement(statement);
    }

    // JDBC types in Java 1.4.1_01:
    // BIT          : -7
    // TINYINT      : -6
    // SMALLINT     : 5
    // INTEGER      : 4
    // BIGINT       : -5
    // FLOAT        : 6
    // REAL         : 7
    // DOUBLE       : 8
    // NUMERIC      : 2
    // DECIMAL      : 3
    // CHAR         : 1
    // VARCHAR      : 12
    // LONGVARCHAR  : -1
    // DATE         : 91
    // TIME         : 92
    // TIMESTAMP    : 93
    // BINARY       : -2
    // VARBINARY    : -3
    // LONGVARBINARY: -4
    // NULL         : 0
    // OTHER        : 1111
    // JAVA_OBJECT  : 2000
    // DISTINCT     : 2001
    // STRUCT       : 2002
    // ARRAY        : 2003
    // BLOB         : 2004
    // CLOB         : 2005
    // REF          : 2006

    //     public static boolean isReadWriteType(int type) {
    // 	if (type == java.sql.Types.VARCHAR 
    // 	    || type == java.sql.Types.CLOB
    // 	    || type == java.sql.Types.INTEGER
    // 	    || type == java.sql.Types.DECIMAL
    // 	    || type == java.sql.Types.LONGVARCHAR)
    // 	    return true;
    // 	return false;
    //     }

    public static boolean isReadAbleType(int type) {
        if (type == java.sql.Types.VARCHAR || type == java.sql.Types.CLOB || type == java.sql.Types.INTEGER
                || type == java.sql.Types.DECIMAL || type == java.sql.Types.DOUBLE || type == java.sql.Types.FLOAT
                || type == java.sql.Types.LONGVARCHAR || type == java.sql.Types.TIME || type == java.sql.Types.TIMESTAMP
                // added binary types for XML_METADATA_GZ processing
                || type == java.sql.Types.BINARY || type == java.sql.Types.VARBINARY || type == java.sql.Types.LONGVARBINARY
                || type == java.sql.Types.BLOB)
            return true;
        return false;
    }

    public static boolean isWritableType(int type) {
        if (type == java.sql.Types.VARCHAR || type == java.sql.Types.CLOB || type == java.sql.Types.INTEGER
                || type == java.sql.Types.DECIMAL || type == java.sql.Types.DOUBLE || type == java.sql.Types.FLOAT
                || type == java.sql.Types.LONGVARCHAR
                // added binary types for XML_METADATA_GZ processing
                || type == java.sql.Types.BINARY || type == java.sql.Types.VARBINARY || type == java.sql.Types.LONGVARBINARY
                || type == java.sql.Types.BLOB)
            return true;
        return false;
    }

    public static boolean isIntegerType(int type) {
        if (type == java.sql.Types.INTEGER)
            return true;
        return false;
    }

    public static boolean isFloatingPointType(int type) {
        if (type == java.sql.Types.DECIMAL)
            return true;
        if (type == java.sql.Types.DOUBLE)
            return true;
        if (type == java.sql.Types.FLOAT)
            return true;
        return false;
    }

    //     public static boolean isReadOnlyType(int type) {
    // 	if (type == java.sql.Types.TIME 
    // 	    || type == java.sql.Types.TIMESTAMP)
    // 	    return true;
    // 	return false;
    //     }

    public int checkTable(DatabaseMetaData dbMeta, String tableName, String columns[]) throws SQLException {
        boolean checks[] = new boolean[columns.length];

        ResultSet resultSet = null;
        if ((this.getDBType().compareTo("oracle") == 0) || (this.getDBType().compareTo("derby") == 0)
                || (this.getDBType().compareTo("db2") == 0)) {
            resultSet = dbMeta.getColumns(null, null, tableName.toUpperCase(), "%");
        } else {
            resultSet = dbMeta.getColumns(null, null, tableName, "%");
        }

        while (resultSet.next() != false) {

            int ctype = resultSet.getInt("DATA_TYPE");
            String cname = resultSet.getString("COLUMN_NAME");
            String typename = resultSet.getString("TYPE_NAME");

            //System.out.println ("table: " + tableName + ", found: " + cname + ", type: " + ctype + ", typename = " + typename);

            if (DBConnector.isReadAbleType(ctype)) {

                for (int i = 0; i < columns.length; i++) {
                    if (columns[i].toUpperCase().compareTo(cname.toUpperCase()) == 0) {
                        checks[i] = true;
                    }
                }

            }
        }

        for (int i = 0; i < columns.length; i++) {
            if (!checks[i]) {
                System.out.println("Couldn't find column \"" + columns[i] + "\" in table \"" + tableName + "\"");
                return -1;
            }
        }
        return 0;

    }

    public int checkSchema() throws SQLException {

        ResultSet resultSet = null;
        DatabaseMetaData dbMeta = this.getMetaData();

        String appColumns[] = { "ID", "NAME" };
        if (checkTable(dbMeta, "application", appColumns) != 0)
            return -1;

        String expColumns[] = { "ID", "NAME", "application" };
        if (checkTable(dbMeta, "experiment", expColumns) != 0)
            return -1;

        String trialColumns[] = { "ID", "NAME", "experiment" };
        if (checkTable(dbMeta, "trial", trialColumns) != 0)
            return -1;

        String metricColumns[] = { "id", "name", "trial" };
        if (checkTable(dbMeta, "metric", metricColumns) != 0)
            return -1;

        String ieColumns[] = { "id", "name", "trial", "group_name" };
        if (checkTable(dbMeta, "interval_event", ieColumns) != 0)
            return -1;

        String aeColumns[] = { "id", "name", "trial", "group_name" };
        if (checkTable(dbMeta, "atomic_event", aeColumns) != 0)
            return -1;

        String ilpColumns[] = { "interval_event", "node", "context", "thread", "metric", "inclusive_percentage", "inclusive",
                "exclusive_percentage", "exclusive", "call", "subroutines", "inclusive_per_call" };

        if (this.getDBType().compareTo("oracle") == 0) {
            ilpColumns[8] = "excl";
        } else if (this.getDBType().compareTo("derby") == 0) {
            ilpColumns[9] = "num_calls";
        } else if (this.getDBType().compareTo("mysql") == 0) {
            ilpColumns[9] = "call";
        }

        if (checkTable(dbMeta, "interval_location_profile", ilpColumns) != 0)
            return -1;

        String alpColumns[] = { "atomic_event", "node", "context", "thread", "sample_count", "maximum_value", "minimum_value",
                "mean_value", "standard_deviation" };
        if (checkTable(dbMeta, "atomic_location_profile", alpColumns) != 0)
            return -1;

        String itsColumns[] = { "interval_event", "metric", "inclusive_percentage", "inclusive", "exclusive_percentage",
                "exclusive", "call", "subroutines", "inclusive_per_call" };

        if (this.getDBType().compareTo("oracle") == 0) {
            itsColumns[5] = "excl";
        } else if (this.getDBType().compareTo("derby") == 0) {
            itsColumns[6] = "num_calls";
        } else if (this.getDBType().compareTo("mysql") == 0) {
            itsColumns[6] = "call";
        }

        if (checkTable(dbMeta, "interval_total_summary", itsColumns) != 0)
            return -1;

        if (checkTable(dbMeta, "interval_mean_summary", itsColumns) != 0)
            return -1;

        return 0;
    }

    public Database getDatabase() {
        return database;
    }
    
    public static void setPasswordCallback(PasswordCallback callback) {
        passwordCallback = callback;
    }

}
