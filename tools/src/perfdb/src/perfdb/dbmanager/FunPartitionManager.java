package perfdb.dbmanager;

import java.util.*;
import java.sql.*;

/****************************************************************
 *
 * Implements basic operations with respect to funindex table 
 * partitioning.
 *
 ****************************************************************/

public class FunPartitionManager extends PartitionManager{

    public FunPartitionManager(String traceUpTableName){
	super();
	
	tableLevel = "FUNINDEX";
	primKey = "FUNINDEXID";	
	traceUp = traceUpTableName.trim()+"(TRIALID)";
    }

    public String tableCreation(String tableName){
	return super.tableCreation(tableName, "prim_"+tableName, "ref_"+tableName, "TRIALID");		
    }
}
