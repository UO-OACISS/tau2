package com.perfdb.dbmanager;

import java.util.*;
import java.sql.*;

/****************************************************************
 *
 * Implements basic operations with respect to experiment table 
 * partitioning.
 *
 ****************************************************************/

public class ExpPartitionManager extends PartitionManager{

    public ExpPartitionManager(String traceUpTableName){
	super();	
	tableLevel = "EXPERIMENTS";	
	primKey = "EXPID";
	traceUp = traceUpTableName.trim()+"(APPID)";
    }

    public String tableCreation(String tableName){

	String createStr = new String("create table ");
	createStr += tableName.trim() + " ( ";		
	createStr += "CONSTRAINT " + "prim_"+tableName + " PRIMARY KEY("+ primKey +"), ";
	createStr += "CONSTRAINT " + "ref_"+tableName + " FOREIGN KEY(APPID) REFERENCES "
                     + traceUp + " ON DELETE CASCADE ON UPDATE CASCADE";
	createStr += " ) inherits (" + tableLevel + "); ";
	return createStr;

	//return super.tableCreation(tableName, "prim_"+tableName, "ref_"+tableName, "APPID");		
    }
}
