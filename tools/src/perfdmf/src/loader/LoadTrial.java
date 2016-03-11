package edu.uoregon.tau.perfdmf.loader;

import jargs.gnu.CmdLineParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DataSourceException;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.UtilFncs;
import edu.uoregon.tau.perfdmf.SnapshotDataSource;
import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.View.ViewRule;
import edu.uoregon.tau.perfdmf.View.ViewRule.NumericViewComparator;
import edu.uoregon.tau.perfdmf.View.ViewRule.StringViewComparator;
import edu.uoregon.tau.perfdmf.taudb.TAUdbDatabaseAPI;

public class LoadTrial {

    private String sourceFiles[];
    private String metadataFile;
    private String metadataString;
    private Experiment exp;
    private boolean fixNames;
    private boolean summaryOnly;
    private boolean useNulls;
    private int expID;
    public int trialID;
    private DataSource dataSource;
    public String trialName;
    public String appName;
    public String expName;
    public String problemFile;
    public String configuration;
    public boolean terminate = true;

    private DatabaseAPI databaseAPI;
    private Trial trial;
	private Double reducePercentage;

    public static void usage() {
        //System.err.println("Usage: taudb_loadtrial -a <appName> -x <expName> -n <name> [options] <file>\n\n"
        System.err.println("Usage: taudb_loadtrial -n <name> [options] <file>\n\n"
                + "try `taudb_loadtrial --help' for more information");
    }

    public static void outputHelp() {

        //System.err.println("Usage: taudb_loadtrial -a <appName> -x <expName> -n <name> [options] <file>\n\n"
        System.err.println("Usage: taudb_loadtrial -n <name> [options] <file>\n\n"
                + "Required Arguments:\n\n" + "  -n, --name <text>               Specify the name of the trial\n"
                //+ "  -a, --applicationname <string>  Specify associated application name\n"
                //+ "                                    for this trial\n"
                //+ "  -x, --experimentname <string>   Specify associated experiment name\n"
                //+ "                                    for this trial\n" + "               ...or...\n\n"
                //+ "  -n, --name <text>               Specify the name of the trial\n"
                //+ "  -e, --experimentid <number>     Specify associated experiment ID\n"
                //+ "                                    for this trial\n" 
                + "  -c, --config <name>             Specify the name of the configuration to use\n"
				+ "\n" + "Optional Arguments:\n\n"
                + "  -g, --configFile <file>         Specify the configuration file to use\n"
                + "                                    (overrides -c)\n"
                + "  -f, --filetype <filetype>       Specify type of performance data, options are:\n"
                + "                                    profiles (default), pprof, dynaprof, mpip,\n"
                + "                                    gprof, psrun, hpm, packed, cube, hpc, ompp,\n"
                + "                                    snap, perixml, gptl, paraver, ipm, google\n"
                //+ "  -t, --trialid <number>          Specify trial ID\n"
                + "  -i, --fixnames                  Use the fixnames option for gprof\n"
                + "  -z, --usenull                   Include NULL values as 0 for mean calculation\n"
                + "  -r, --reduce <percentage>       Aggregate all timers less than percentage as \"other\"\n"
                + "  -d, --metadata-file <filename>  XML metadata for the trial\n" 
                + "  -m, --metadata                  Colon seperated metadata name/value pairs \n" 
                + "                                  <foo1=bar1:foo2=bar2>\n\n" + "Notes:\n"
                
                + "  -a, --applicationname <string>  Specify associated application name\n"
                + "                                    for this trial. Creates a view.\n"
                + "  -x, --experimentname <string>   Specify associated experiment name\n"
                + "                                    for this trial. Creates a view.\n"
                
                + "  For the TAU profiles type, you can specify either a specific set of profile\n"
                + "files on the commandline, or you can specify a directory (by default the current\n"
                + "directory).  The specified directory will be searched for profile.*.*.* files,\n"
                + "or, in the case of multiple counters, directories named MULTI_* containing\n" + "profile data.\n\n"
                + "Examples:\n\n" + "  taudb_loadtrial -c mydb -n \"Batch 001\"\n"
                + "    This will load profile.* (or multiple counters directories MULTI_*) into\n"
                + "    database config mydb and give the trial the name \"Batch 001\"\n\n"
                + "  taudb_loadtrial -c mydb -n \"HPM data 01\" -f hpm perfhpm*\n"
                + "    This will load perfhpm* files of type HPMToolkit into datatabase mydb and give\n"
                + "    the trial the name \"HPM data 01\"\n\n");
    }

    public LoadTrial(String configFileName, String sourceFiles[]) {
        this.sourceFiles = sourceFiles;

        databaseAPI = new DatabaseAPI();
        try {
            databaseAPI.initialize(configFileName, true);
			if (databaseAPI.db().getSchemaVersion() > 0) {
				// copy the DatabaseAPI object data into a new TAUdbDatabaseAPI object
				
				databaseAPI = new TAUdbDatabaseAPI(databaseAPI);
			}

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

    }

    public boolean checkForExp(String expid, String appName, String expName) {
    	if(databaseAPI.db().getSchemaVersion() >0){
    		System.err.println("Ignore Exp in load trial");
//            Experiment exp = databaseAPI.getExperiment(appName, expName, true);
//            return exp != null;
    		return true;
    	}
        if (expid != null) {
            this.expID = Integer.parseInt(expid);

            try {
                exp = databaseAPI.setExperiment(this.expID);
            } catch (Exception e) {}
            if (exp == null) {
                System.err.println("Experiment id " + expid + " not found,  please enter a valid experiment ID.");
                System.exit(-1);
                return false;
            } else {
                return true;
            }
        } else {
            Experiment exp = databaseAPI.getExperiment(appName, expName, true);
            this.expID = exp.getID();
            return true;
        }
    }

    public boolean checkForTrialName(String trialName) {
        Trial tmpTrial = databaseAPI.setTrial(trialName, false);
        if (tmpTrial == null)
            return false;
        else
            return true;
    }

    public boolean checkForTrial(String trialid) {
        Trial tmpTrial = databaseAPI.setTrial(Integer.parseInt(trialid), false);
        if (tmpTrial == null)
            return false;
        else
            return true;
    }

    public void loadTrial(int fileType) {

        File[] files = new File[sourceFiles.length];
        for (int i = 0; i < sourceFiles.length; i++) {
            files[i] = new File(sourceFiles[i]);
        }

        try {
            dataSource = UtilFncs.initializeDataSource(files, fileType, fixNames);
			DataSource.setMeanIncludeNulls(useNulls);
        } catch (DataSourceException e) {
            e.printStackTrace();
            System.err.println("Error: Unable to initialize datasource!");
            return;
        }

        trial = new Trial();
        trial.setDataSource(dataSource);

        // set the metadata file name before loading the data, because
        // aggregateData() is called at the end of the dataSource.load()
        // and this file has to be set before then.
        
        if(metadataString!=null&&metadataString.length()>0){
        	MetaDataMap mdMap = new MetaDataMap();
        	String[] pairs = metadataString.split(":");
        	for(int i=0;i<pairs.length;i++){
        		String[] pair = pairs[i].split("=");
        		if(pair.length==2)
        			mdMap.put(pair[0], pair[1]);
        	}
        	dataSource.setMetaData(mdMap);
        }
        
        try {
            if (metadataFile != null) {
                dataSource.setMetadataFile(metadataFile);
            }
        } catch (Exception e) {
            System.err.println("Error Loading metadata:");
            e.printStackTrace();
            System.exit(1);
        }

        try {
		    // generate all the statistics - unless we already have them!
            if (databaseAPI.db().getSchemaVersion()>=0 && !(dataSource instanceof SnapshotDataSource)) {
            	dataSource.setGenerateTAUdbStatistics(true);
            }
            dataSource.load();
            if (reducePercentage != null && reducePercentage > 0.0) {
            	dataSource.reduceTrial(reducePercentage);
            }
        } catch (Exception e) {
            System.err.println("Error Loading Trial:");
            e.printStackTrace();
        }

        // set the meta data from the datasource
        trial.setMetaData(dataSource.getMetaData());
        trial.setUncommonMetaData(dataSource.getUncommonMetaData());

        if (trialID == 0) {
            saveTrial();
        } else {
            appendToTrial();
        }
        if (this.terminate) {
            databaseAPI.terminate();
        }
    }

    public void saveTrial() {
        trial.setName(trialName);
        int appView=-2;
        String NAME="NAME";
        String ID="ID";
        String appViewName="Application-"+this.appName;
        List<ViewRule> rules = new ArrayList<ViewRule>(2);
        boolean createAppView=true;
        if (this.appName != null)
        {
        	//First see if this view already exists
        	List<View>checkViews = View.getViews(0, this.databaseAPI.db());
        	if(checkViews!=null){
        		for(View topView:checkViews){
        			String vname = topView.getField(NAME);
        			if(vname.equals(appViewName)){
        				createAppView=false;
        				appView=Integer.parseInt(topView.getField(ID));
        				break;
        			}
        		}
        	}
        	//If it doesn't then make it, otherwise use its id when making the experiment view
        	if(createAppView){
        	rules.add(ViewRule.createStringViewRule("Application", this.appName, StringViewComparator.EXACTLY));
        	try {
				appView=View.createView(this.databaseAPI.db(), appViewName, true, -1, rules);
			} catch (SQLException e) {
				e.printStackTrace();
			}
        	}
        	trial.getMetaData().put("Application", this.appName);
        }
        if (this.expName != null)
        {
        	if(appView>=0)
        	{
        		String expViewName="Experiment-"+this.expName;
        		boolean createExpView=true;
        		List<View>checkViews = View.getViews(appView, this.databaseAPI.db());
               	if(checkViews!=null){
            		for(View expView:checkViews){
            			String vname = expView.getField(NAME);
            			if(vname.equals(expViewName)){
            				createExpView=false;
            				break;
            			}
            		}
            	}
        		if(createExpView){
        		rules.add(ViewRule.createStringViewRule("Experiment", this.expName, StringViewComparator.EXACTLY));
        		try {
					View.createView(this.databaseAPI.db(), expViewName, true, appView, rules);
				} catch (SQLException e) {
					e.printStackTrace();
				}
        		}
        	}
        	trial.getMetaData().put("Experiment", this.expName);
        }

        System.err.println("TrialName: " + trialName);
        trial.setExperimentID(expID);
        try {
            trialID = databaseAPI.uploadTrial(trial, summaryOnly);
        } catch (DatabaseException e) {
            e.printStackTrace();
            Exception e2 = e.getException();
            System.err.println("from: ");
            e2.printStackTrace();
            System.exit(-1);
        }
        System.err.println("Done saving trial!");
    }

    public void appendToTrial() {
        // set some things in the trial
        trial.setID(this.trialID);
        try {
            databaseAPI.saveTrial(trial, null);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        System.err.println("Done adding metric to trial!");
    }

    public String getProblemString() {
        // if the file wasn't passed in, this is an existing trial.
        if (problemFile == null)
            return new String("");

        // open the file
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(problemFile));
        } catch (Exception e) {
            System.err.println("Problem file not found!  Exiting...");
            System.exit(0);
        }
        // read the file, one line at a time, and do some string
        // substitution to make sure that we don't blow up our
        // SQL statement. ' characters aren't allowed...
        StringBuffer problemString = new StringBuffer();
        String line;
        while (true) {
            try {
                line = reader.readLine();
            } catch (Exception e) {
                line = null;
            }
            if (line == null)
                break;
            problemString.append(line.replaceAll("'", "\'"));
        }

        // close the problem file
        try {
            reader.close();
        } catch (Exception e) {}

        // return the string
        return problemString.toString();
    }

    static public void main(String[] args) {
        // 	for (int i=0; i<args.length; i++) {
        // 	    System.out.println ("args[" + i + "]: " + args[i]);
        // 	}

        CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
        CmdLineParser.Option experimentidOpt = parser.addStringOption('e', "experimentid");
        CmdLineParser.Option nameOpt = parser.addStringOption('n', "name");
        //CmdLineParser.Option problemOpt = parser.addStringOption('p',
        // "problemfile");
        CmdLineParser.Option configOpt = parser.addStringOption('c', "config");
        CmdLineParser.Option configFileOpt = parser.addStringOption('g', "configFile");
        CmdLineParser.Option trialOpt = parser.addStringOption('t', "trialid");
        CmdLineParser.Option typeOpt = parser.addStringOption('f', "filetype");
        CmdLineParser.Option fixOpt = parser.addBooleanOption('i', "fixnames");
        CmdLineParser.Option metadataFileOpt = parser.addStringOption('l', "metadataFile");
        CmdLineParser.Option metadataOpt = parser.addStringOption('m', "metadata");
        CmdLineParser.Option appNameOpt = parser.addStringOption('a', "applicationname");
        CmdLineParser.Option expNameOpt = parser.addStringOption('x', "experimentname");
        CmdLineParser.Option summaryOpt = parser.addBooleanOption('s', "summaryonly");
        CmdLineParser.Option percentageOpt = parser.addDoubleOption('r', "reduce");
        CmdLineParser.Option useNullOpt = parser.addBooleanOption('z', "usenull");

        try {
            parser.parse(args);
        } catch (CmdLineParser.OptionException e) {
            System.err.println(e.getMessage());
            LoadTrial.usage();
            System.exit(-1);
        }

        Boolean help = (Boolean) parser.getOptionValue(helpOpt);
        //String sourceFile = (String)parser.getOptionValue(sourcefileOpt);
        String configName = (String) parser.getOptionValue(configOpt);
        String configFile = (String) parser.getOptionValue(configFileOpt);
        String experimentID = (String) parser.getOptionValue(experimentidOpt);
        String trialName = (String) parser.getOptionValue(nameOpt);
        String appName = (String) parser.getOptionValue(appNameOpt);
        String expName = (String) parser.getOptionValue(expNameOpt);
        Double percentage = (Double) parser.getOptionValue(percentageOpt);

        //String problemFile = (String)parser.getOptionValue(problemOpt);
        String trialID = (String) parser.getOptionValue(trialOpt);
        String fileTypeString = (String) parser.getOptionValue(typeOpt);
        Boolean fixNames = (Boolean) parser.getOptionValue(fixOpt);
        String metadataFile = (String) parser.getOptionValue(metadataFileOpt);
        String metadataString = (String) parser.getOptionValue(metadataOpt);
        Boolean summaryOnly = (Boolean) parser.getOptionValue(summaryOpt);
        Boolean useNull = (Boolean) parser.getOptionValue(useNullOpt);

        if (help != null && help.booleanValue()) {
            LoadTrial.outputHelp();
            System.exit(-1);
        }

        if (configFile == null) {
            if (configName == null)
                configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg";
            else
                configFile = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg." + configName;
        }

        
        String sourceFiles[] = parser.getRemainingArgs();
        
        boolean multippk=false;
        LoadTrial tmpLT = new LoadTrial(configFile, sourceFiles);
        if (trialName == null) {
        	
        	if(sourceFiles!=null&&sourceFiles.length>0){
        		
        		String s = sourceFiles[0];
        		File f=new File(s);
        		int tid = UtilFncs.identifyData(f);
        		if((fileTypeString!=null&&fileTypeString.equals("packed"))||tid==DataSource.PPK){
        			multippk=true;
        		}
        	}
        	if(!multippk){
            System.err.println("Error: Missing trial name\n");
            LoadTrial.usage();
            System.exit(-1);
        	}
			/*
        } else if (experimentID == null && expName == null && tmpLT.getDatabaseAPI().getDb().getSchemaVersion()==0) {
            System.err.println("Error: Missing experiment id or name\n");
            LoadTrial.usage();
            System.exit(-1);
        } else if (expName != null && appName == null) {
            System.err.println("Error: Missing application name\n");
            LoadTrial.usage();
            System.exit(-1);
			*/
        }

        //String sourceFiles[] = parser.getRemainingArgs();

        int fileType = DataSource.TAUPROFILE;
        if (fileTypeString != null) {
            if (fileTypeString.equals("profiles")) {
                fileType = DataSource.TAUPROFILE;
            } else if (fileTypeString.equals("pprof")) {
                fileType = DataSource.PPROF;
            } else if (fileTypeString.equals("dynaprof")) {
                fileType = DataSource.DYNAPROF;
            } else if (fileTypeString.equals("mpip")) {
                fileType = DataSource.MPIP;
            } else if (fileTypeString.equals("hpm")) {
                fileType = DataSource.HPM;
            } else if (fileTypeString.equals("gprof")) {
                fileType = DataSource.GPROF;
            } else if (fileTypeString.equals("psrun")) {
                fileType = DataSource.PSRUN;
            } else if (fileTypeString.equals("packed")) {
                fileType = DataSource.PPK;
            } else if (fileTypeString.equals("cube")) {
                fileType = DataSource.CUBE;
            } else if (fileTypeString.equals("hpc")) {
                fileType = DataSource.HPCTOOLKIT;
            } else if (fileTypeString.equals("gyro")) {
                fileType = DataSource.GYRO;
            } else if (fileTypeString.equals("gamess")) {
                fileType = DataSource.GAMESS;
            } else if (fileTypeString.equals("gptl")) {
                fileType = DataSource.GPTL;
            } else if (fileTypeString.equals("ipm")) {
                fileType = DataSource.IPM;
            } else if (fileTypeString.equals("ompp")) {
                fileType = DataSource.OMPP;
            } else if (fileTypeString.equals("snap")) {
                fileType = DataSource.SNAP;
            } else if (fileTypeString.equals("paraver")) {
                fileType = DataSource.PARAVER;
            } else if (fileTypeString.equals("perixml")) {
                fileType = DataSource.PERIXML;
            } else if (fileTypeString.equals("google")) {
                fileType = DataSource.GOOGLE;
            } else {
                System.err.println("Please enter a valid file type.");
                LoadTrial.usage();
                System.exit(-1);
            }
        } else {
            if (sourceFiles.length >= 1) {
                fileType = UtilFncs.identifyData(new File(sourceFiles[0]));
            }
        }

        if (trialName == null) {
            trialName = "";
        }

        if (fixNames == null) {
            fixNames = new Boolean(false);
        }
        if (summaryOnly == null) {
            summaryOnly = new Boolean(false);
        }
        if (useNull == null) {
            useNull = new Boolean(false);
        }
        if (percentage == null) {
        	percentage = new Double(0.0);
        }
        
        
        if(multippk){
        	
        	for(int i=0;i<sourceFiles.length;i++)
        	{
        		
        		trialName=sourceFiles[i].substring(0,sourceFiles[i].lastIndexOf('.'));
        		
        	LoadTrial trans = new LoadTrial(configFile, new String []{sourceFiles[i]});
            trans.checkForExp(experimentID, appName, expName);
            if (trialID != null) {
                trans.checkForTrial(trialID);
                trans.trialID = Integer.parseInt(trialID);
            }

            trans.trialName = trialName;
            //trans.problemFile = problemFile;
            trans.fixNames = fixNames.booleanValue();
            trans.metadataFile = metadataFile;
            trans.metadataString = metadataString;
            trans.summaryOnly = summaryOnly.booleanValue();
            trans.reducePercentage = percentage;
            trans.loadTrial(fileType);
        	}
        }
        else
        {
        LoadTrial trans = new LoadTrial(configFile, sourceFiles);
        if (trans.databaseAPI.db().getSchemaVersion()==0) {
            trans.checkForExp(experimentID, appName, expName);
        } else {
        	//System.err.println("Warning - this is the TAUdb schema, there is no Experiment table");
        }
        if (trialID != null) {
            trans.checkForTrial(trialID);
            trans.trialID = Integer.parseInt(trialID);
        }

        trans.trialName = trialName;
        //trans.problemFile = problemFile;
        trans.fixNames = fixNames.booleanValue();
        trans.metadataFile = metadataFile;
        trans.metadataString = metadataString;
        trans.summaryOnly = summaryOnly.booleanValue();
        if (appName != null) {
		  trans.appName = appName;
		}
        if (expName != null) {
		  trans.expName = expName;
		}
        trans.useNulls = useNull.booleanValue();
        trans.reducePercentage = percentage;
        trans.loadTrial(fileType);
        // the trial will be saved when the load is finished (update is called)
        }
    }

    public void setMetadataFile(String metadataFile) {
        this.metadataFile = metadataFile;
    }
    
    public void setMetadataString(String metadataString) {
        this.metadataString = metadataString;
    }

    public void setFixNames(boolean fixNames) {
        this.fixNames = fixNames;
    }

    public void setSummaryOnly(boolean summaryOnly) {
        this.summaryOnly = summaryOnly;
    }

    public DatabaseAPI getDatabaseAPI() {
        return databaseAPI;
    }

}
