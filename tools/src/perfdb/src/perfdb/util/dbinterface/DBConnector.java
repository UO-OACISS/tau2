package perfdb.util.dbinterface;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

/*******************************************************
 * Implements access to a database
 * Be sure to specify JDBC Driver in the class path.
 *
 * Drivers are registered with the driverName.
 * connection string = dbaddress + acct 
 *******************************************************/

public class DBConnector implements DB {

    private Statement statement;
    private DBAcct acct = null;
    private Connection conn = null;
	private ParseConfig parseConfig = null;

    private String dbaddress;
    // it looks like "jdbc:postgresql://zeta:5432/perfdb;" in PostgreSQL.

    private String driverName;
    // it should be "org.postgresql.Driver" in PostgreSQL.

    public DBConnector(ParseConfig parser) throws SQLException {
		super();
		parseConfig = parser;
		setJDBC(parser);
		register();
    }

    public DBConnector(DBAcct acct, ParseConfig parser) throws SQLException {
		super();
		parseConfig = parser;
		setJDBC(parser);
		setAcct(acct);
		register();
		connect();
    }

	public DBConnector(DBAcct acct, ParseConfig parser, boolean test) throws SQLException {
		super();
		parseConfig = parser;
		setJDBC(parser);
		setAcct(acct);
		register();
		testConnect();
	}

    public void setJDBC(ParseConfig parser){

	if (parser.getDBType().compareTo("mysql") == 0)
		dbaddress = "jdbc:" + parser.getDBType() + "://" + parser.getDBHost()
	    + ":" + parser.getDBPort() + "/" + parser.getDBName() + "?";
	else
		dbaddress = "jdbc:" + parser.getDBType() + "://" + parser.getDBHost()
	    + ":" + parser.getDBPort() + "/" + parser.getDBName() + ";";
	//System.out.println(dbaddress);

	driverName = parser.getJDBCDriver();
	//System.out.println(driverName);
	
    }

    public void close() {
	try {
	    if (conn.isClosed()) {
		return;
	    } else {
		conn.close();
	    }
	} catch (SQLException ex) {
	    ex.printStackTrace();
	}
    }

    public boolean connect() {
	try {
	    if (conn != null) {
		return true;
	    }
	    conn = DriverManager.getConnection(getConnectString());
	    return true;
	} catch (SQLException ex) {
		System.err.println("Cannot connect to server.");
	    ex.printStackTrace();
	    return false;
	}
    }

	public boolean testConnect() throws SQLException {
	try {
		if (conn != null) {
		return true;
		}
		conn = DriverManager.getConnection(getConnectString());
		return true;
	} catch (SQLException ex) {
		System.err.println("Cannot connect to server.");
		throw ex;
	}
	}

    public boolean connect(DBAcct acct) {
	setAcct(acct);
	return connect();
    }

    public boolean connect(Connection newconn) {
	conn = newconn;
	return true;
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
	try {
	    if (statement == null) {
		if (conn == null) {
		    System.err.println("Database is closed for " + sql);
		    return 0;
		}
		statement = conn.createStatement();
	    }
	    return statement.executeUpdate(sql.trim());
	} catch (SQLException ex) {
	    ex.printStackTrace();
	    return 0;
	}
    }

    public DBAcct getAcct() {
	return acct;
    }

    public java.sql.Connection getConnection() {
	return conn;
    }

    public String getConnectString() {
	return getDBAddress() + getAcct() ;
    }

    public String getDBAddress() {
	return dbaddress;
    }

    /*** Get the first returned value of a query. ***/

    public String getDataItem(String query) {
	//returns the value of the first column of the first row
	try {
	    ResultSet resultSet = executeQuery(query);
	    if (resultSet.next() == false){
		resultSet.close();
		return null;
	    }
	    else {
		String result = resultSet.getString(1);
		resultSet.close();
		return result;
	    }
	} catch (SQLException ex) {
	    ex.printStackTrace();
	    return null;
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
	    // Class.forName(driverName);
	    Class.forName(driverName).newInstance();
	} catch (Exception ex) {
	    ex.printStackTrace();
	}
    }

    public void setAcct(JDBCAcct newValue) {
	this.acct = newValue;
    }

    public void setAcct(DBAcct newValue) {
	this.acct = newValue;
    }

    public void setDBAddress(String newValue) {
	this.dbaddress = newValue;
    }

    public String getDBType() {
	return new String(this.parseConfig.getDBType());
    }
}
