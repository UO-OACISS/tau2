package com.perfdb.dbmanager;

import java.util.*;
import java.sql.*;

/****************************************************************
 *
 * Implements basic operations with respect to userevent table 
 * partitioning.
 *
 ****************************************************************/

public class UEPartitionManager extends PartitionManager{

    public UEPartitionManager(String traceUpTableName){
	super();
	
	tableLevel = "USEREVENT";	
	primKey = "LOCID";
	traceUp = traceUpTableName.trim()+"(LOCID)";
    }
    
    public String tableCreation(String tableName){
	return super.tableCreation(tableName, "prim_"+tableName, "ref_"+tableName, "LOCID");		
    }
}
