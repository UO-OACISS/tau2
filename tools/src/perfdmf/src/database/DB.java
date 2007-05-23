/*
 * DB.java
 *
 * Copyright 2005-2007                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.perfdmf.database;

import java.sql.*;

import edu.uoregon.tau.perfdmf.Database;

/**
 * Low level API wrapper around a JDBC connection
 *
 * <P>CVS $Id: DB.java,v 1.5 2007/05/23 01:40:18 amorris Exp $</P>
 * @version $Revision: 1.5 $
 */
public interface DB {
    public void close();

    public ResultSet executeQuery(String statement) throws SQLException;

    public int executeUpdate(String statement) throws SQLException;

    public boolean execute(String statement) throws SQLException;

    public String getDataItem(String query) throws SQLException;

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
    
    
    public Database getDatabase();
}
