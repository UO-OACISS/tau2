package perfdb.loadxml;

import perfdb.util.dbinterface.*;
import perfdb.util.io.*;
import perfdb.dbmanager.*;
import java.io.*;
import java.net.*;
import java.sql.*;

public class Main {
    private Load load = null;
    private DB db = null;
    
    private static String USAGE = 
        "Main configfilename \n  (help | loadschema <schemafile> \n        | loadtrial <pprof.xml> \n        | loadapp <App_Info.xml> \n        | loadexp <application id> <Sys_info.xml> <Config_info.xml> <Compiler_info.xml> <Instru_info.xml>)";

    // These two flags determines whether to store an app. or exp. separately. 
    private int enable_AppPartition=0;
    private int enable_ExpPartition=0;

    private perfdb.ConnectionManager connector;

    public Main(String configFileName) {
	super();
	connector = new perfdb.ConnectionManager(configFileName);
    }

    public perfdb.ConnectionManager getConnector(){
	return connector;
    }

    public Load getLoad() {
	if (load == null) {
	    if (connector.getDB() == null) {
		load = new Load(connector.getParserClass());
	    } else {
		load = new Load(connector.getDB(), connector.getParserClass());
	    }
	}
	return load;
    }

    public void setAppPartition(int flag){
	enable_AppPartition = flag;
    }

    public int getAppPartition(){
	return enable_AppPartition;
    }

    public void setExpPartition(int flag){
	enable_ExpPartition = flag;
    }

    public int getExpPartition(){
	return enable_ExpPartition;
    }
   
    public void errorPrint(String msg) {
	System.err.println(msg);
    }

    /*** Parse and load an application. ***/   

    public void storeApp(String appFile) {
	BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
	String parInfo;
	String appid;

	// I decided not to prompt the user for this.  
	// Default to no.  - khuck 05/03/03
	setAppPartition(0);
	try{
	/* 
	    System.out.println("Do you want to store the data associated with the application into an individual table");
	    System.out.println("(as opposed to store data of all applications into one huge table) for query efficiency reason?");
	    System.out.println("(Yes/No)");
	    parInfo = reader.readLine();

	    if (parInfo.startsWith("Y") || parInfo.startsWith("y"))
		setAppPartition(1);
	    else setAppPartition(0);
 	*/
	} catch (Throwable ex) {
	    errorPrint("Error: " + ex.getMessage());
    }
	
	try {	
	    appid = getLoad().parseApp(appFile);
	   
	} catch (Throwable ex) {
	    errorPrint("Error: " + ex.getMessage());
	    return;
	}

	if ((appid!=null)&&(appid.trim().length()>0)){
	    if (getAppPartition()==1){

		StringBuffer buf = new StringBuffer();
		
		ExpPartitionManager expTable = new ExpPartitionManager("Applications");
		String tableName0 = "EXP_FOR_APP"+appid;
		buf.append(expTable.tableCreation(tableName0));

		buf.append("update Applications set exp_table_name = '"+tableName0+"' where appid = "+appid+";");
		buf.append("ALTER TABLE ONLY "+tableName0+" ALTER trial_table_name SET DEFAULT 'TRIAL_FOR_APP" + appid+"'; ");
		
		TrialPartitionManager trialTable = new TrialPartitionManager(tableName0);
		tableName0 = "TRIAL_FOR_APP" + appid;
		buf.append(trialTable.tableCreation(tableName0));
		
		FunPartitionManager funTable = new FunPartitionManager(tableName0);
		tableName0 = "FUN_FOR_APP" + appid;
		buf.append(funTable.tableCreation(tableName0));

		LocPartitionManager locTable = new LocPartitionManager(tableName0);
		String tableName1 = "LOC_FOR_APP" + appid;
		buf.append(locTable.tableCreation(tableName1));

		PprofPartitionManager pprofTable = new PprofPartitionManager(tableName1);
		String tableName2 = "PPROF_FOR_APP"+appid;
		buf.append(pprofTable.tableCreation(tableName2));

		TotalPartitionManager totalTable = new TotalPartitionManager(tableName0);
		tableName2 = "TOTAL_FOR_APP"+appid;
		buf.append(totalTable.tableCreation(tableName2));

		MeanPartitionManager meanTable = new MeanPartitionManager(tableName0);
		tableName2 = "MEAN_FOR_APP"+appid;
		buf.append(meanTable.tableCreation(tableName2));
	    
		//System.out.println(buf.toString());

		try {
		    getLoad().getDB().executeUpdate(buf.toString());			
		} catch (SQLException ex) {
		    ex.printStackTrace();
		}			   
	    }
	}
	else {
	    System.out.println("Loadding application failed");
	    return;
	}
    }

    /* Load environemnt information associated with an experiment*/
 
    public void storeExp(String appid, String sysinfo, String configinfo, String compilerinfo, String instruinfo ) {
	BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String appname;
	String version;
	String expid;
	String parInfo;
	File ff;
	
	try{

		// I decided not to prompt the user for this.  
		// Default to no.  - khuck 05/03/03
	    setExpPartition(0);
			/* 
	    System.out.println("Do you want to store the data associated with this experiment into an individual table");
	    System.out.println("(as opposed to store data of all experiments into one huge table) for query efficiency reason?");
	    System.out.println("(Yes/No)");
	    parInfo = reader.readLine();

	    if (parInfo.startsWith("Y") || parInfo.startsWith("y"))
		setExpPartition(1);
	    else setExpPartition(0);
		*/

		// I decided to prompt the user for the Appid, 
		// rather than the name & version
	    // System.out.println("please input the id of the application associated with the experiment:");
	    // appid = reader.readLine();
	   
		/* 
	    System.out.println("please input the name of application associated with the experiment:");
	    appname = reader.readLine();
	    System.out.println("please input the version of application associated with the experiment:");
	    version = reader.readLine();
	    
	    appid = getLoad().lookupApp(appname, version);
		*/

	    if ((appid!=null)&&(appid.trim().length()>0)){ 

		appid = appid.trim();
		
		String hostname = InetAddress.getLocalHost().getHostName();
		// System.out.println("please input system information file");
		// String sysinfo = reader.readLine();
		ff = new File(sysinfo.trim());
		if (!ff.exists()) {
		    System.out.println("Warning: the System file doesn't exist!");
		    return;
		}
		sysinfo = hostname + ":" + ff.getAbsolutePath();

		// System.out.println("please input configuration information file");
		// String configinfo = reader.readLine();
		ff = new File(configinfo.trim());
		if (!ff.exists()) { 
		    System.out.println("Warning: the configuration file doesn't exist!");
		    return;
		}       
		configinfo = hostname + ":" + ff.getAbsolutePath();

		// System.out.println("please input compiler information file");
		// String compilerinfo = reader.readLine();
		ff = new File(compilerinfo.trim());
		if (!ff.exists()) {
		    System.out.println("Warning: the compiler file doesn't exist!");
		    return;
		}
		compilerinfo = hostname + ":" + ff.getAbsolutePath();

		// System.out.println("please input instrumentation information file");
		// String instruinfo = reader.readLine();
		ff = new File(instruinfo.trim());
		if (!ff.exists()) {
		    System.out.println("Warning: the instrumentation file doesn't exist!");
		    return;
		}
		instruinfo = hostname + ":" + ff.getAbsolutePath();

		// Obtain enough experiment information, begin loading. First check whether the experiment has been loaded.

		String expTableName = "Experiments";
		expid = getLoad().lookupExp(expTableName, appid, sysinfo, configinfo, compilerinfo, instruinfo);
				
		if (expid!=null){ // i.e., the expriment has been loaded. Quit loading.
		    System.out.println("The experiment has already been loaded, whose id is: "+expid);
		    System.exit(-1);
		}    

		// Otherwise, begin loading the experiment.
		
		StringBuffer buf = new StringBuffer();
		
		String nextLevelTable = null;
		String tableName0;

		buf.append("select exp_table_name from Applications where appid = "+appid+";");
		
		expTableName = getLoad().getDB().getDataItem(buf.toString());	

		if (expTableName.equalsIgnoreCase("experiments"))
		    setAppPartition(0);
		else setAppPartition(1);

		buf.delete(0, buf.toString().length());

		if (getAppPartition()==1){ // i.e., each applcation has a set of exp,trial,pprof,total,and mean tables.
		   		    	   						    
		    expid = getLoad().insertExp(expTableName, appid, sysinfo, configinfo, compilerinfo, instruinfo, nextLevelTable);
		    			
		    if ((expid!=null)&&(expid.trim().length()>0)){
			if (getExpPartition()==1){ // i.e., each experiment has a set of trial, pprof, total,and mean tables.
				
			    TrialPartitionManager trialTable = new TrialPartitionManager(expTableName);
			    trialTable.setInheritTable("TRIAL_FOR_APP"+appid);
			    tableName0 = "TRIAL_FOR_APP"+appid+"_EXP" + expid;			    
			    buf.append(trialTable.tableCreation(tableName0));				
			    nextLevelTable = tableName0;

			    FunPartitionManager funTable = new FunPartitionManager(tableName0);
			    funTable.setInheritTable("FUN_FOR_APP"+appid);
			    tableName0 = "FUN_FOR_APP"+appid+"_EXP" + expid;			    
			    buf.append(funTable.tableCreation(tableName0));

			    LocPartitionManager locTable = new LocPartitionManager(tableName0);
			    locTable.setInheritTable("LOC_FOR_APP"+appid);
			    String tableName1 = "LOC_FOR_APP"+appid+"_EXP" + expid;			    
			    buf.append(locTable.tableCreation(tableName1));

			    PprofPartitionManager pprofTable = new PprofPartitionManager(tableName1);
			    pprofTable.setInheritTable("PPROF_FOR_APP"+appid);
			    String tableName2 = "PPROF_FOR_APP"+appid+"_EXP" + expid;
			    buf.append(pprofTable.tableCreation(tableName2));

			    TotalPartitionManager totalTable = new TotalPartitionManager(tableName0);
			    totalTable.setInheritTable("TOTAL_FOR_APP"+appid);
			    tableName2 = "TOTAL_FOR_APP"+appid+"_EXP" + expid;
			    buf.append(totalTable.tableCreation(tableName2));

			    MeanPartitionManager meanTable = new MeanPartitionManager(tableName0);
			    meanTable.setInheritTable("MEAN_FOR_APP"+appid);
			    tableName2 = "MEAN_FOR_APP"+appid+"_EXP" + expid;
			    buf.append(meanTable.tableCreation(tableName2));
   			       			    
			    buf.append("update "+expTableName+" set trial_table_name = '"+nextLevelTable+"' where expid = "+expid+";");
			    System.out.println(buf.toString());
			    try {
				getLoad().getDB().executeUpdate(buf.toString());			
			    } catch (SQLException ex) {
				ex.printStackTrace();
			    }	
			}					    
		    } 
		    else {
			System.out.println("Loading the experiment failed.");
			System.exit(-1);
		    }
		}    
		else { // no application partition.
		    		    	   		    
		    expid = getLoad().insertExp(expTableName, appid, sysinfo, configinfo, compilerinfo, instruinfo, nextLevelTable);
		   
		    if ((expid!=null)&&(expid.trim().length()>0)){
			if (getExpPartition()==1){ // i.e., each experiment has a set of trial,pprof,total,and mean tables.

			    TrialPartitionManager trialTable = new TrialPartitionManager(expTableName);
			    tableName0 = "TRIAL_FOR_EXP" + expid;			    
			    buf.append(trialTable.tableCreation(tableName0));
			    nextLevelTable = tableName0;

			    FunPartitionManager funTable = new FunPartitionManager(tableName0);
			    tableName0 = "FUN_FOR_EXP" + expid;			    
			    buf.append(funTable.tableCreation(tableName0));

			    LocPartitionManager locTable = new LocPartitionManager(tableName0);
			    String tableName1 = "LOC_FOR_EXP" + expid;			    
			    buf.append(locTable.tableCreation(tableName1));

			    PprofPartitionManager pprofTable = new PprofPartitionManager(tableName1);
			    String tableName2 = "PPROF_FOR_EXP" + expid;
			    buf.append(pprofTable.tableCreation(tableName2));

			    TotalPartitionManager totalTable = new TotalPartitionManager(tableName0);
			    tableName2 = "TOTAL_FOR_EXP" + expid;
			    buf.append(totalTable.tableCreation(tableName2));

			    MeanPartitionManager meanTable = new MeanPartitionManager(tableName0);
			    tableName2 = "MEAN_FOR_EXP" + expid;
			    buf.append(meanTable.tableCreation(tableName2));

			    buf.append("update "+expTableName+" set trial_table_name = '"+nextLevelTable+"' where expid = "+expid+";");
			    System.out.println(buf.toString());
			    try {
				getLoad().getDB().executeUpdate(buf.toString());			
			    } catch (SQLException ex) {
				ex.printStackTrace();
			    }	
			}
		    }	
		    else {
			System.out.println("Loading the experiment failed.");
			System.exit(-1);
		    }
		}
	    }
	    else 
		System.out.println("Quit loading experiment.");
		
	} catch (Throwable ex) {
	    errorPrint("Error: " + ex.getMessage());
        }
    }

    /*** Store a xml document for a trial ***/

    public String storeDocument(String xmlFile) {

	String trialId = null;
	try {
	    				    
	    trialId = getLoad().parse(xmlFile);		

	} catch (Throwable ex) {
	    errorPrint("Error: " + ex.getMessage());
	}

	if (trialId != null) {
	    System.out.println("Loaded " + xmlFile + ", the trial id is: " + trialId);
	} else {
	    errorPrint("Was unable to load document from " + xmlFile);
	}

	return trialId;
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
	
	if (args.length == 0) {
	    System.err.println(USAGE);
	    System.exit(-1);
        }

	int ctr = 0;
	String command;	
	
	// create a new Main object, pass in the configuration file name
	Main demo = new Main(args[ctr++]);
	demo.getConnector().connect();

	
	while (ctr < args.length) {
	    command = args[ctr++];
	    if (command.equalsIgnoreCase("HELP")) {
		System.err.println(USAGE);
		continue;
	    }
	    /***** Load database schema to establish PerfDB, invoke at most one time. ******/
	    if (command.equalsIgnoreCase("LOADSCHEMA")) {
		demo.getConnector().genParentSchema(args[ctr++]);
		continue;
	    }
	    /***** Load appliation into PerfDB *********/
	    if (command.equalsIgnoreCase("LOADAPP")) {
		demo.storeApp(args[ctr++]);
		continue;
	    }
	    /***** Load experiment into PerfDB ********/
	    if (command.equalsIgnoreCase("LOADEXP")) {
		demo.storeExp(args[ctr], args[ctr+1], args[ctr+2], args[ctr+3], args[ctr+4]);
		continue;
	    }
	    /***** Load a trial into PerfDB *********/
	    if (command.equalsIgnoreCase("LOADXML")) {
		demo.storeDocument(args[ctr++]);
		continue;
	    }
	}

	demo.getConnector().dbclose();
    }

}

