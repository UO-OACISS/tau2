package perfdb.dbmanager;

/****************************************************************
 *
 * Implements basic operations with respect to table partitioning.
 *
 ****************************************************************/

public class PartitionManager{

    protected String partitionKey;

    // tableLevel records the level this table belongs to.
    protected String tableLevel;

    // primkey indicates the primary key of the table.
    protected String primKey;

    // traceUp indicates foreign reference to upper level table.
    protected String traceUp;

    public PartitionManager(){
	super();
    }

    public void setInheritTable(String tableLevel){ this.tableLevel = tableLevel; }

    public String tableCreation(String tableName, String primConsName, String refConsName, String foreignKeyName){
	String createStr = new String("create table ");
	createStr += tableName.trim() + " ( ";	
	
	createStr += "CONSTRAINT " + primConsName + " PRIMARY KEY("+ primKey +"), ";
	createStr += "CONSTRAINT " + refConsName.trim() + " FOREIGN KEY(" + foreignKeyName + ") REFERENCES "
                     + traceUp + " ON DELETE CASCADE ON UPDATE CASCADE";
	createStr += " ) inherits (" + tableLevel + "); ";
	return createStr;
    } 

}
