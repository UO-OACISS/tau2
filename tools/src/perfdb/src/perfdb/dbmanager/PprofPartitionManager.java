package com.perfdb.dbmanager;

import java.util.*;
import java.sql.*;

/****************************************************************
 *
 * Implements basic operations with respect to pprof table 
 * partitioning.
 *
 ****************************************************************/

public class PprofPartitionManager extends PartitionManager{

    public PprofPartitionManager(String traceUpTableName){
	super();
	
	tableLevel = "PPROF";	
	primKey = "LOCID";
	traceUp = traceUpTableName.trim()+"(LOCID)";
    }

    public String tableCreation(String tableName){
	return super.tableCreation(tableName, "prim_"+tableName, "ref_"+tableName, "LOCID");		
    }
}
