package com.perfdb.util.dbinterface;

/*** Contain access information for connection to DBMS through JDBC. ***/

public class JDBCAcct implements DBAcct {

    protected String jdbcaccess = null;

    public JDBCAcct() {
	super();
    }
    public JDBCAcct(String acct) {
	super();
	setAcct(acct);
    }
    public String getAcct() {
	return jdbcaccess;
    }
    public void setAcct(String acct) {
	this.jdbcaccess = acct;
    }
    public String toString() {
	return getAcct();
    }    
}
