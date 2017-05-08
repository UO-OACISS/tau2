package edu.uoregon.tau.paraprof;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.MetaDataMap.MetaDataKey;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.taudb.TAUdbDatabaseAPI;

public class ExternalController {

    static public void runController() {
        try {
            System.out.println("Control Mode Active!");

            BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));

            String input = stdin.readLine();

            while (input != null) {
                System.out.println("got input: " + input);

                if (input.startsWith("control ")) {
                    processCommand(input.substring(8));
                }
                else{
                	System.out.println("Valid control statements start with the string 'control'");
                }
                input = stdin.readLine();
            }

            exitController();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    static public void exitController() {
        System.out.println("Control Mode Complete!");
        System.exit(0);

    }
    
    private static List<String> getControlArgs(String baseCommand, String userCommand){
    	String args="";
    	List<String> argList=new ArrayList<String>();
    	int bcl=baseCommand.length();
    	
    	if(userCommand.length()>bcl){
    		args=userCommand.substring(bcl+1).trim();
    	}
    	
    	StringTokenizer tokenizer = new StringTokenizer(args, " ");
    	
    	while(tokenizer.hasMoreTokens()){
    		argList.add(tokenizer.nextToken());
    	}
    	
    	return argList;
    }
    
    public static final String LISTAPPLICATIONS="list applications";
    public static final String LISTEXPERIMENTS="list experiments";
    public static final String LISTTRIALS="list trials";
    public static final String LOAD = "load";
    public static final String UPLOAD = "upload";
    public static final String EXPORT = "export";

    static public void processCommand(String command) throws Exception {
        System.out.println("processing command: " + command);
        if (command.equals("open manager")) {
            ParaProf.paraProfManagerWindow.setVisible(true);
        } else if (command.equals("list databases")) {
            listDatabases();
        } else if (command.startsWith(LISTAPPLICATIONS)) {
            listApplications(getControlArgs(LISTAPPLICATIONS,command));
        } else if (command.startsWith(LISTEXPERIMENTS)) {
            listExperiments(getControlArgs(LISTEXPERIMENTS,command));
        } else if (command.startsWith(LISTTRIALS)) {
            listTrials(getControlArgs(LISTTRIALS, command));
        } else if (command.startsWith(LOAD)) {
            loadDBTrial(getControlArgs(LOAD,command));
        } else if (command.startsWith(UPLOAD)) {
            uploadTauTrial(getControlArgs(UPLOAD,command));
        } 
        else if (command.startsWith(EXPORT)){
        	exportTrialsToPPK(getControlArgs(EXPORT,command));
        }else if (command.equals("exit")) {
            exitController();
        }
        else{
        	System.out.println("Valid control statements are:\n"
        			+ "open manager\n"
        			+ "list databases\n"
        			+ "list applications <database id>\n"
        			+ "list experiments\n"
        			+ "list trials\n"
        			+ "load\n"
        			+ "upload\n"
        			+ "export\n"
        			+ "exit");
        }
    }

    static public void exportTrialsToPPK(List<String> args) throws Exception{
    	if(args.size()<=3){
    		System.out.println("Invalid input. Requires writable output directory, numeric database id and one more more numeric trial ids");
    		return;
    	}
    	
    	String ppkPath=args.get(0);
    	int dbID = Integer.parseInt(args.get(1));
    	File targetDir=new File(ppkPath);
    	
    	if(!targetDir.isDirectory()||!targetDir.canWrite()){
    		System.out.println("Invalid input. ppk output location must be a writable directory.");
    		return;
    	}
    	
    	for(int i=2;i<args.size();i++)
        {
    		int trialID = Integer.parseInt(args.get(i));
    		exportTrialToPPK(targetDir,dbID,trialID);
        }
        
       
    }
    
    private static void  exportTrialToPPK(File targetDir,int dbID,int trialID) throws Exception{
    	 DatabaseAPI databaseAPI = new DatabaseAPI();
         Database selectedDB=Database.getDatabases().get(dbID);
         databaseAPI.initialize(selectedDB);
 		if (databaseAPI.db().getSchemaVersion() > 0) {
 			// copy the DatabaseAPI object data into a new TAUdbDatabaseAPI object
 			databaseAPI = new TAUdbDatabaseAPI(databaseAPI);
 		}
 		
 		Trial trial = databaseAPI.setTrial(trialID, true);
         DBDataSource dbDataSource = new DBDataSource(databaseAPI);
         dbDataSource.load();
         
          //= databaseAPI.setTrial(trialID, true);//new Trial();
         trial.setDataSource(dbDataSource);
         trial.setID(trialID);
     	

 		
 		databaseAPI.initialize(selectedDB);
         
         DB dbOb = databaseAPI.db();
 		trial.loadXMLMetadata(dbOb);
 		dbDataSource.setMetaData(trial.getMetaData());
 		
 		String newPPKString = targetDir.getCanonicalPath() + File.separator + trial.getName() + "-"
				+ trial.getID() + ".ppk";
 		File newPPKFile = new File(newPPKString);

 		DataSourceExport.writePacked(dbDataSource, newPPKFile);
 		System.out.println("Wrote "+newPPKFile.getCanonicalPath());
    }
    
    static public void loadDBTrial(List<String> ids) throws Exception {
        //StringTokenizer tokenizer = new StringTokenizer(command, " ");
        
        if(ids.size()!=2){
    		System.out.println("Invalid input. Requires numeric database id and numeric trial id");
    		//outputCommand("return "+0+" default");
    		//outputCommand("endreturn");
    		return;
    	}

        int dbID = Integer.parseInt(ids.get(0));
        int trialID = Integer.parseInt(ids.get(1));

        DatabaseAPI databaseAPI = new DatabaseAPI();
        databaseAPI.initialize(Database.getDatabases().get(dbID));
		if (databaseAPI.db().getSchemaVersion() > 0) {
			// copy the DatabaseAPI object data into a new TAUdbDatabaseAPI object
			databaseAPI = new TAUdbDatabaseAPI(databaseAPI);
		}

        databaseAPI.setTrial(trialID, false);
        DBDataSource dbDataSource = new DBDataSource(databaseAPI);
        dbDataSource.load();
        
        Trial trial = new Trial();
        trial.setDataSource(dbDataSource);
        trial.setID(trialID);
        
        ParaProfTrial ppTrial = new ParaProfTrial(trial);
        ppTrial.finishLoad();
        ppTrial.setID(trial.getID());
        ppTrial.showMainWindow();
    }

    static public void uploadTauTrial(List<String> ids) throws Exception {
        //StringTokenizer tokenizer = new StringTokenizer(command, " ");
        
        if(ids.size()!=5){
    		System.out.println("Invalid input. Requires path to profile location, numeric database id, application name, experiment name and trial name");
    		outputCommand("return "+-1);
    		outputCommand("endreturn");
    		return;
    	}

        String location = ids.get(0);// tokenizer.nextToken();
        int dbID = Integer.parseInt(ids.get(1));
        String appName = ids.get(2);//tokenizer.nextToken();
        String expName = ids.get(3);//tokenizer.nextToken();
        String trialName = ids.get(4);///tokenizer.nextToken();

        File file = new File(location);
        File[] files = new File[1];
        files[0] = file;
        int type = DataSource.TAUPROFILE;
        if(!file.isDirectory())
        {	
        	type = UtilFncs.identifyData(file);
        }
        DataSource dataSource = UtilFncs.initializeDataSource(files, type, false);
        dataSource.load();

        Trial trial = new Trial();
        trial.setDataSource(dataSource);

        DatabaseAPI databaseAPI = new DatabaseAPI();
        databaseAPI.initialize(Database.getDatabases().get(dbID));

        trial.setName(trialName);
        if(databaseAPI.db().getSchemaVersion()==0){
        Experiment exp = databaseAPI.getExperiment(appName, expName, true);
        trial.setExperimentID(exp.getID());
        }else{
        	MetaDataMap map = trial.getMetaData();
        	MetaDataKey key = map.newKey("Application");
        	map.put(key , appName);
        	key = map.newKey("Experiment");
        	map.put(key , expName);
        }
        int trialID = databaseAPI.uploadTrial(trial, false);
        outputCommand("return " + trialID);
        outputCommand("endreturn");

    }

    static public void listApplications(List<String> databaseID) throws SQLException {
    	
    	if(databaseID.size()!=1){
    		System.out.println("Invalid input. Requires numeric database id");
    		outputCommand("return "+0+" default");
    		outputCommand("endreturn");
    		return;
    	}
    	
        int id = Integer.parseInt(databaseID.get(0));
        List<Database> databases = Database.getDatabases();
        

        DatabaseAPI databaseAPI = new DatabaseAPI();
        databaseAPI.initialize(databases.get(id));
        if(databaseAPI.db().getSchemaVersion() >0){
        	System.out.println("This is a TAUdbDatabase which no longer supports applications.");
        	outputCommand("return "+0+" default");
        }
        else{
        List<Application> apps = databaseAPI.getApplicationList();
        for (Iterator<Application> it = apps.iterator(); it.hasNext();) {
            Application app = it.next();
            outputCommand("return " + app.getID() + " " + app.getName());
        }
        }
        outputCommand("endreturn");
    }

    static public void listExperiments(List<String> ids) throws SQLException {

       // StringTokenizer tokenizer = new StringTokenizer(ids, " ");
    	
    	if(ids.size()!=2){
    		System.out.println("Invalid input. Requires numeric database id and numeric application id");
    		outputCommand("return "+0+" default");
    		outputCommand("endreturn");
    		return;
    	}

        int dbID = Integer.parseInt(ids.get(0));
        int appID = Integer.parseInt(ids.get(1));

        DatabaseAPI databaseAPI = new DatabaseAPI();
        databaseAPI.initialize(Database.getDatabases().get(dbID));
        databaseAPI.setApplication(appID);
        if(databaseAPI.db().getSchemaVersion() >0){
        	System.out.println("This is a TAUdbDatabase which no longer supports Experiments.");
        	outputCommand("return "+0+" default");
        }else{
        List<Experiment> exps = databaseAPI.getExperimentList();
        for (Iterator<Experiment> it = exps.iterator(); it.hasNext();) {
            Experiment exp = it.next();
            outputCommand("return " + exp.getID() + " " + exp.getName());
        }
        }
        outputCommand("endreturn");
    }

    static public void listTrials(List<String> ids) throws SQLException {

        //StringTokenizer tokenizer = new StringTokenizer(ids, " ");
    	
    	if(ids.size()<1){
    		System.out.println("Invalid input. Requires numeric database id and, for older databases, experiment id");
    		outputCommand("return "+-1+" default");
    		outputCommand("endreturn");
    		return;
    	}

        int dbID = Integer.parseInt(ids.get(0));
        
        DatabaseAPI databaseAPI = new DatabaseAPI();
        databaseAPI.initialize(Database.getDatabases().get(dbID));
        int schemav = databaseAPI.db().getSchemaVersion();
        int expID=-1;
        if(!(schemav>0)){
        	if(ids.size()!=2){
        
        	System.out.println("Invalid input. Requires numeric database id and, for older databases, experiment id");
        	outputCommand("return "+-1+" default");
    		outputCommand("endreturn");
    		return;
        	}
        	else{
        		expID = Integer.parseInt(ids.get(1));
        	}
        }
        
        

        //DatabaseAPI databaseAPI = new DatabaseAPI();
        //databaseAPI.initialize(Database.getDatabases().get(dbID));
		if (schemav > 0) {
			// copy the DatabaseAPI object data into a new TAUdbDatabaseAPI object
			databaseAPI = new TAUdbDatabaseAPI(databaseAPI);
		}else{
			databaseAPI.setExperiment(expID);
		}
        List<Trial> trials = databaseAPI.getTrialList(false);
        for (Iterator<Trial> it = trials.iterator(); it.hasNext();) {
            Trial trial = it.next();
            outputCommand("return " + trial.getID() + " " + trial.getName());
        }
        outputCommand("endreturn");
    }

    static public void listDatabases() {
        List<Database> databases = Database.getDatabases();
        int id = 0;
        for (Iterator<Database> it = databases.iterator(); it.hasNext();) {
            Database db = it.next();
            outputCommand("return " + id + " " + db.getName());
            id++;
        }
        outputCommand("endreturn");

    }

    static public void outputCommand(String command) {
        System.out.println("control " + command);
    }

}
