package perfdb.dbmanager;

/****************************************************************
 *
 * Implements basic operations with respect to Application table 
 * partitioning.
 *
 ****************************************************************/

public class AppPartitionManager extends PartitionManager{

    public AppPartitionManager(){
	super();
	tableLevel = "APPLICATIONS";
	primKey = "APPID";
    }

}
