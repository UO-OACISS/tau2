package perfdb.loadxml;

import jargs.gnu.CmdLineParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.sql.SQLException;

import perfdb.dbmanager.ExpPartitionManager;
import perfdb.dbmanager.FunPartitionManager;
import perfdb.dbmanager.LocPartitionManager;
import perfdb.dbmanager.MeanPartitionManager;
import perfdb.dbmanager.PprofPartitionManager;
import perfdb.dbmanager.TotalPartitionManager;
import perfdb.dbmanager.TrialPartitionManager;
import perfdb.util.dbinterface.DB;

public class Main {
    private Load load = null;
    private DB db = null;
    
    private static String USAGE = 
        "USAGE: Main [{-h,--help}] [{-g,--configfile} filename] \n"
		+ "    [{-c,--command} loadschema] [{-s,--schemafile} filename] \n"
		+ "  | [{-c,--command} loadapp] [{-x,--xmlfile} filename] \n"
		+ "  | [{-c,--command} loadexp] [{-a,--applicationid} value] [{-s,--systeminfo} filename] [{-n,--configinfo} filename] [{-m,--compilerinfo} filename] [{-i,--instrumentationinfo} filename]) \n"
		+ "  | [{-c,--command} loadtrial] [{-x,--xmlfile} filename] [{-t,--trialid] trial id] \n";

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

    public String storeApp(String appFile) {
	BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
	String parInfo;
	String appid = null;

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
	    return null;
	}

	if ((appid!=null)&&(appid.trim().length()>0)){
	    if (getAppPartition()==1){

		StringBuffer buf = new StringBuffer();
		
		ExpPartitionManager expTable = new ExpPartitionManager("application");
		String tableName0 = "EXP_FOR_APP"+appid;
		buf.append(expTable.tableCreation(tableName0));

		buf.append("update application set experiment_table_name = '"+tableName0+"' where id = "+appid+";");
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
	    return null;
	}
	return appid;
    }

    /* Load environemnt information associated with an experiment*/
 
    public String storeExp(String appid, String sysinfo, String configinfo, String compilerinfo, String instruinfo ) {
	BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String appname;
	String version;
	String expid = null;
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
		    return null;
		}
		sysinfo = hostname + ":" + ff.getAbsolutePath();

		// System.out.println("please input configuration information file");
		// String configinfo = reader.readLine();
		ff = new File(configinfo.trim());
		if (!ff.exists()) { 
		    System.out.println("Warning: the configuration file doesn't exist!");
		    return null;
		}       
		configinfo = hostname + ":" + ff.getAbsolutePath();

		// System.out.println("please input compiler information file");
		// String compilerinfo = reader.readLine();
		ff = new File(compilerinfo.trim());
		if (!ff.exists()) {
		    System.out.println("Warning: the compiler file doesn't exist!");
		    return null;
		}
		compilerinfo = hostname + ":" + ff.getAbsolutePath();

		// System.out.println("please input instrumentation information file");
		// String instruinfo = reader.readLine();
		ff = new File(instruinfo.trim());
		if (!ff.exists()) {
		    System.out.println("Warning: the instrumentation file doesn't exist!");
		    return null;
		}
		instruinfo = hostname + ":" + ff.getAbsolutePath();

		// Obtain enough experiment information, begin loading. First check whether the experiment has been loaded.

		String expTableName = "experiment";
		expid = getLoad().lookupExp(expTableName, appid, sysinfo, configinfo, compilerinfo, instruinfo);
				
		if (expid!=null){ // i.e., the expriment has been loaded. Quit loading.
		    System.out.println("The experiment has already been loaded, whose id is: "+expid);
		    System.exit(-1);
		}    

		// Otherwise, begin loading the experiment.
		
		StringBuffer buf = new StringBuffer();
		
		String nextLevelTable = null;
		String tableName0;

		buf.append("select experiment_table_name from application where id = "+appid+";");
		
		expTableName = getLoad().getDB().getDataItem(buf.toString());	

		if (expTableName.equalsIgnoreCase("experiment"))
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
   			       			    
			    buf.append("update "+expTableName+" set trial_table_name = '"+nextLevelTable+"' where id = "+expid+";");
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

			    buf.append("update "+expTableName+" set trial_table_name = '"+nextLevelTable+"' where id = "+expid+";");
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
		return expid;
    }

    /*** Store a xml document for a trial ***/

    public String storeDocument(String xmlFile, String trialId) {
		if (trialId.compareTo("0") != 0) {
			String trialIdOut = getLoad().lookupTrial("trial", trialId);
			if (trialIdOut==null){
		    	System.out.println("The trial " + trialId + " was not found.");
		    	System.exit(-1);
			}    
		}

		try {
	    	trialId = getLoad().parse(xmlFile, trialId);		
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
        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option commandOpt = parser.addStringOption('c', "command");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option xmlfileOpt = parser.addStringOption('x', "xmlfile");
        CmdLineParser.Option trialidOpt = parser.addStringOption('t', "trialid");
        CmdLineParser.Option applicationidOpt = parser.addStringOption('a', "applicationid");
        CmdLineParser.Option schemafileOpt = parser.addStringOption('s', "schemafile");
        CmdLineParser.Option compilerinfoOpt = parser.addStringOption('m', "compilerinfo");
        CmdLineParser.Option systeminfoOpt = parser.addStringOption('y', "systeminfo");
        CmdLineParser.Option configinfoOpt = parser.addStringOption('f', "configinfo");
        CmdLineParser.Option instrumentationinfoOpt = parser.addStringOption('i', "instrumentationinfo");

        try {
            parser.parse(args);
        }
        catch ( CmdLineParser.OptionException e ) {
            System.err.println(e.getMessage());
	    	System.err.println(USAGE);
	    	System.exit(-1);
        }

        Boolean help = (Boolean)parser.getOptionValue(helpOpt);
        String command = (String)parser.getOptionValue(commandOpt);
        String configFile = (String)parser.getOptionValue(configfileOpt);
        String xmlFile = (String)parser.getOptionValue(xmlfileOpt);
        String trialID = (String)parser.getOptionValue(trialidOpt);
        String applicationID = (String)parser.getOptionValue(applicationidOpt);
        String schemaFile = (String)parser.getOptionValue(schemafileOpt);
        String compilerInfo = (String)parser.getOptionValue(compilerinfoOpt);
        String systemInfo = (String)parser.getOptionValue(systeminfoOpt);
        String configInfo = (String)parser.getOptionValue(configinfoOpt);
        String instrumentationInfo = (String)parser.getOptionValue(instrumentationinfoOpt);

    	if (help != null && help.booleanValue()) {
			System.err.println(USAGE);
	    	System.exit(-1);
    	}

		if (command == null) {
            System.err.println("Please enter a valid command.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
		}

		if (configFile == null) {
            System.err.println("Please enter a valid config file.");
	    	System.err.println(USAGE);
	    	System.exit(-1);
		}

	// create a new Main object, pass in the configuration file name
		Main demo = new Main(configFile);
		demo.getConnector().connect();

		int exitval = 0;
	
    	/***** Load database schema to establish PerfDB, invoke at most one time. ******/
		if (command.equalsIgnoreCase("LOADSCHEMA")) {
			demo.getConnector().genParentSchema(schemaFile);
    	}
    	/***** Load appliation into PerfDB *********/
		else if (command.equalsIgnoreCase("LOADAPP")) {
			String appid = demo.storeApp(xmlFile);
			if (appid != null)
				exitval = Integer.parseInt(appid);
    	}
    	/***** Load experiment into PerfDB ********/
		else if (command.equalsIgnoreCase("LOADEXP")) {
			String expid = demo.storeExp(applicationID, systemInfo, configInfo, compilerInfo, instrumentationInfo);
			if (expid != null)
				exitval = Integer.parseInt(expid);
    	}
    	/***** Load a trial into PerfDB *********/
		else if (command.equalsIgnoreCase("LOADXML") || command.equalsIgnoreCase("LOADTRIAL")) {
			String trialid = demo.storeDocument(xmlFile, trialID);
			if (trialid != null)
				exitval = Integer.parseInt(trialid);
    	}

		demo.getConnector().dbclose();
		System.exit(exitval);
    }

}

