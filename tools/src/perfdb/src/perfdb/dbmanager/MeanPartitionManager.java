package com.perfdb.dbmanager;

import java.util.*;
import java.sql.*;

/****************************************************************
 *
 * Implements basic operations with respect to mean summary table 
 * partitioning.
 *
 ****************************************************************/

public class MeanPartitionManager extends PartitionManager{

    public MeanPartitionManager(String traceUpTableName){
	super();
	
	tableLevel = "MEANSUMMARY";	
	primKey = "FUNINDEXID";
	traceUp = traceUpTableName.trim()+"(FUNINDEXID)";
    }

    public String tableCreation(String tableName){
	return super.tableCreation(tableName, "prim_"+tableName, "ref_"+tableName, "FUNINDEXID");		
    }
}
