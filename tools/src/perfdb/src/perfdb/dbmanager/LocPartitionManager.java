package perfdb.dbmanager;

import java.util.*;
import java.sql.*;

/****************************************************************
 *
 * Implements basic operations with respect to Locationindex table 
 * partitioning.
 *
 ****************************************************************/

public class LocPartitionManager extends PartitionManager{
    
    public LocPartitionManager(String traceUpTableName){
	super();
	
	tableLevel = "LOCATIONINDEX";
	primKey = "LOCID";	
	traceUp = traceUpTableName.trim()+"(FUNINDEXID)";
    }

    public String tableCreation(String tableName){
	return super.tableCreation(tableName, "prim_"+tableName, "ref_"+tableName, "FUNINDEXID");		
    }
}
