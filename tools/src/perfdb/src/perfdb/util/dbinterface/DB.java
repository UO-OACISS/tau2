package perfdb.util.dbinterface;

import java.sql.*;

/*** Interface to access a DBMS ***/

public interface DB {
    void close();
    public boolean connect(DBAcct acct);
    public boolean connect(Connection conn);
    ResultSet executeQuery(String statement) throws SQLException;
    int executeUpdate(String statement) throws SQLException;
    String getDataItem(String query);
    public boolean isClosed();
}
