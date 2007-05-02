package edu.uoregon.tau.perfdmf.database;

import java.sql.*;

/*** Interface to access a DBMS ***/

public interface DB {
    void close();
    ResultSet executeQuery(String statement) throws SQLException;
    int executeUpdate(String statement) throws SQLException;
    boolean execute(String statement) throws SQLException;
    String getDataItem(String query) throws SQLException;
    public boolean isClosed();
    public String getDBType();
    public Connection getConnection();
    public PreparedStatement prepareStatement(String statement) throws SQLException;
    public String getSchemaPrefix();
    public DatabaseMetaData getMetaData() throws SQLException;

    public int checkSchema() throws SQLException;


    public void setAutoCommit(boolean auto) throws SQLException;
    public void commit() throws SQLException;
    public void rollback() throws SQLException;
    public String getConnectString();

}
