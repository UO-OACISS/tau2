package perfdb.dbmanager;

/****************************************************************
 *
 * Implements basic operations with respect to trial table 
 * partitioning.
 *
 ****************************************************************/

public class TrialPartitionManager extends PartitionManager{

    public TrialPartitionManager(String traceUpTableName){
	super();
	
	tableLevel = "TRIALS";
	primKey = "TRIALID";	
	traceUp = traceUpTableName.trim()+"(EXPID)";
    }

    public String tableCreation(String tableName){
	String createStr = new String("create table ");
	createStr += tableName.trim() + " ( ";			
	createStr += "CONSTRAINT prim_" + tableName + " PRIMARY KEY("+ primKey +"), ";
	createStr += "CONSTRAINT xmlref_" + tableName + " FOREIGN KEY(XMLFILEID) REFERENCES XMLFILES(XMLFILEID) ON DELETE NO ACTION ON UPDATE CASCADE, ";
	createStr += "CONSTRAINT expref_" + tableName + " FOREIGN KEY(EXPID) REFERENCES "
                     + traceUp + " ON DELETE CASCADE ON UPDATE CASCADE";
	createStr += " ) inherits (" + tableLevel + "); ";
	return createStr;   	
    }
}


