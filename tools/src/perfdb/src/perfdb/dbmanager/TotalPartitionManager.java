package com.perfdb.dbmanager;

import java.util.*;
import java.sql.*;

/****************************************************************
 *
 * Implements basic operations with respect to total summary table 
 * partitioning.
 *
 ****************************************************************/

public class TotalPartitionManager extends PartitionManager{

    public TotalPartitionManager(String traceUpTableName){
	super();
	
	tableLevel = "TOTALSUMMARY";	
	primKey = "FUNINDEXID"; 
	traceUp = traceUpTableName.trim()+"(FUNINDEXID)";
    }

    public String tableCreation(String tableName){
	return super.tableCreation(tableName, "prim_"+tableName, "ref_"+tableName, "FUNINDEXID");		
    }
}
