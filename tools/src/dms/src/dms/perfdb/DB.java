package dms.perfdb;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

/*** Interface to access a DBMS ***/

public interface DB {
    void close();
    ResultSet executeQuery(String statement) throws SQLException;
    int executeUpdate(String statement) throws SQLException;
    String getDataItem(String query);
    public boolean isClosed();
    public String getDBType();
	public Connection getConnection();
	public PreparedStatement prepareStatement(String statement) throws SQLException;
}
