/* 
   ParaProfManagerTableModel.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  The container for the MappingDataWindowPanel.
*/

/*
  To do: 
*/

package edu.uoregon.tau.paraprof;

import javax.swing.tree.*;
import javax.swing.table.*;

public class ParaProfManagerTableModel extends AbstractTableModel{
    public ParaProfManagerTableModel(Object obj, DefaultTreeModel defaultTreeModel){
	super();
	
	if(obj instanceof ParaProfApplication){
	    this.application = (ParaProfApplication)obj;
	    type = 0;
	}
	else if(obj instanceof ParaProfExperiment){
	    this.experiment = (ParaProfExperiment)obj;
	    type = 1;
	}
	else if(obj instanceof ParaProfTrial){
	    this.trial = (ParaProfTrial)obj;
	    type = 2;
	}
	else{
	    this.metric = (Metric) obj;
	    type = 3;
	}

	this.application = application;
	this.defaultTreeModel = defaultTreeModel;
    }
  
    public int getColumnCount(){
	return 2;}
  
    public int getRowCount(){
	switch(type){
	case 0:
	    return 7;
	case 1:
	    return 27;
	case 2:
	    return 10;
	case 3:
	    return 5;
	default:
	    return 0;
	}
    }
  
    public String getColumnName(int c){
	return columnNames[c];}
  
    public Object getValueAt(int r, int c){
	switch(type){
	case 0:
	    if(c==0){
		switch(r){
		case(0):
		    return "Name";
		case(1):
		    return "Application ID";
		case(2):
		    return "Language";
		case(3):
		    return "Para_diag";
		case(4):
		    return "Usage";
		case(5):
		    return "Executable Options";
		case(6):
		    return "Description";
		default:
		    return "";
		}
	    }
	    else{
		switch(r){
		case(0):
		    return application.getName();
		case(1):
		    return new Integer(application.getID());
		case(2):
		    return application.getLanguage();
		case(3):
		    return application.getParaDiag();
		case(4):
		    return application.getUsage();
		case(5):
		    return application.getExecutableOptions();
		case(6):
		    return application.getDescription();
		default:
		    return "";
		}
	    }
	case 1:
	    if(c==0){
		switch(r){
		case(0):
		    return "Name";
		case(1):
		    return "Application ID";
		case(2):
		    return "Experiment ID";
		case(3):
		    return "User Data";
		case(4):
		    return "System Name";
		case(5):
		    return "System Machine Type";
		case(6):
		    return "System Arch.";
		case(7):
		    return "System OS";
		case(8):
		    return "System Memory Size";
		case(9):
		    return "System Processor Amount";
		case(10):
		    return "System L1 Cache Size";
		case(11):
		    return "System L2 Cache Size";
		case(12):
		    return "System User Data";
		case(13):
		    return "Configuration Prefix";
		case(14):
		    return "Configuration Architecture";
		case(15):
		    return "Configuration CPP";
		case(16):
		    return "Configuration CC";
		case(17):
		    return "Configuration JDK";
		case(18):
		    return "Configuration Profile";
		case(19):
		    return "Configuration User Data";
		case(20):
		    return "Compiler CPP Name";
		case(21):
		    return "Compiler CPP Version";
		case(22):
		    return "Compiler CC Name";
		case(23):
		    return "Compiler CC Version";
		case(24):
		    return "Compiler Java Dir. Path";
		case(25):
		    return "Compiler Java Version";
		case(26):
		    return "Compiler User Data";
		default:
		    return "";
		}
	    }
	    else{
		switch(r){
		case(0):
		    return experiment.getName();
		case(1):
		    return new Integer(experiment.getApplicationID());
		case(2):
		    return new Integer(experiment.getID());
		case(3):
		    return experiment.getUserData();
		case(4):
		    return experiment.getSystemName();
		case(5):
		    return experiment.getSystemMachineType();
		case(6):
		    return experiment.getSystemArch();
		case(7):
		    return experiment.getSystemOS();
		case(8):
		    return experiment.getSystemMemorySize();
		case(9):
		    return experiment.getSystemProcessorAmount();
		case(10):
		    return experiment.getSystemL1CacheSize();
		case(11):
		    return experiment.getSystemL2CacheSize();
		case(12):
		    return experiment.getSystemUserData();
		case(13):
		    return experiment.getConfigPrefix();
		case(14):
		    return experiment.getConfigArchitecture();
		case(15):
		    return experiment.getConfigCpp();
		case(16):
		    return experiment.getConfigCc();
		case(17):
		    return experiment.getConfigJdk();
		case(18):
		    return experiment.getConfigProfile();
		case(19):
		    return experiment.getConfigUserData();
		case(20):
		    return experiment.getCompilerCppName();
		case(21):
		    return experiment.getCompilerCppVersion();
		case(22):
		    return experiment.getCompilerCcName();
		case(23):
		    return experiment.getCompilerCcVersion();
		case(24):
		    return experiment.getCompilerJavaDirpath();
		case(25):
		    return experiment.getCompilerJavaVersion();
		case(26):
		    return experiment.getCompilerUserData();
		default:
		    return "";
		}
	    }
	case 2:
	    if(c==0){
		switch(r){
		case(0):
		    return "Name";
		case(1):
		    return "Application ID";
		case(2):
		    return "Experiment ID";
		case(3):
		    return "Trial ID";
		case(4):
		    return "Time";
		case(5):
		    return "Node Count";
		case(6):
		    return "Contexts Per Node";
		case(7):
		    return "Threads Per Context";
		case(8):
		    return "User Data";
		case(9):
		    return "Problem Definition";
		default:
		    return "";
		}
	    }
	    else{
		switch(r){
		case(0):
		    return trial.getName();
		case(1):
		    return new Integer(trial.getApplicationID());
		case(2):
		    return new Integer(trial.getExperimentID());
		case(3):
		    return new Integer(trial.getID());
		case(4):
		    return trial.getTime();
		case(5):
		    return new Integer(trial.getNodeCount());
		case(6):
		    return new Integer(trial.getNumContextsPerNode());
		case(7):
		    return new Integer(trial.getNumThreadsPerContext());
		case(8):
		    return trial.getUserData();
		case(9):
		    return trial.getProblemDefinition();
		default:
		    return "";
		}
	    }
	    case 3:
	    if(c==0){
		switch(r){
		case(0):
		    return "Name";
		case(1):
		    return "Application ID";
		case(2):
		    return "Experiment ID";
		case(3):
		    return "Trial ID";
		case(4):
		    return "Metric ID";
		default:
		    return "";
		}
	    }
	    else{
		switch(r){
		case(0):
		    return metric.getName();
		case(1):
		    return new Integer(metric.getApplicationID());
		case(2):
		    return new Integer(metric.getExperimentID());
		case(3):
		    return new Integer(metric.getTrialID());
		case(4):
		    return new Integer(metric.getID());
		default:
		    return "";
		}
	    }
	default:
	    return "";
	}
    }
  
    public boolean isCellEditable(int r, int c){
	if(c==1 && r!=1)
	    return true;
	else
	    return false;
    }
  
    public void setValueAt(Object obj, int r, int c){
	//Should be getting a string.
	if(obj instanceof String){
	    String tmpString = (String) obj;
	    if(c==1){
		switch(type){
		case 0:
		    switch(r){
		    case(0):
			application.setName(tmpString);
			break;
		    case(1):
			application.setID(Integer.parseInt(tmpString));
			break;
		    case(2):
			application.setLanguage(tmpString);
			break;
		    case(3):
			application.setParaDiag(tmpString);
			break;
		    case(4):
			application.setUsage(tmpString);
			break;
		    case(5):
			application.setExecutableOptions(tmpString);
			break;
		    case(6):
			application.setDescription(tmpString);
			break;
		    }
		    defaultTreeModel.nodeChanged(application.getDMTN());
		    break;
		case 1:
		    switch(r){
		    case(0):
			experiment.setName(tmpString);
			break;
		    case(3):
			experiment.setUserData(tmpString);
			break;
		    case(4):
			experiment.setSystemName(tmpString);
			break;
		    case(5):
			experiment.setSystemMachineType(tmpString);
			break;
		    case(6):
			experiment.setSystemArch(tmpString);
			break;
		    case(7):
			experiment.setSystemOS(tmpString);
			break;
		    case(8):
			experiment.setSystemMemorySize(tmpString);
			break;
		    case(9):
			experiment.setSystemProcessorAmount(tmpString);
			break;
		    case(10):
			experiment.setSystemL1CacheSize(tmpString);
			break;
		    case(11):
			experiment.setSystemL2CacheSize(tmpString);
			break;
		    case(12):
			experiment.setSystemUserData(tmpString);
			break;
		    case(13):
			experiment.setConfigurationPrefix(tmpString);
			break;
		    case(14):
			experiment.setConfigurationArchitecture(tmpString);
			break;
		    case(15):
			experiment.setConfigurationCpp(tmpString);
			break;
		    case(16):
			experiment.setConfigurationCc(tmpString);
			break;
		    case(17):
			experiment.setConfigurationJdk(tmpString);
			break;
		    case(18):
			experiment.setConfigurationProfile(tmpString);
			break;
		    case(19):
			experiment.setConfigurationUserData(tmpString);
			break;
		    case(20):
			experiment.setCompilerCppName(tmpString);
			break;
		    case(21):
			experiment.setCompilerCppVersion(tmpString);
			break;
		    case(22):
			experiment.setCompilerCcName(tmpString);
			break;
		    case(23):
			experiment.setCompilerCcVersion(tmpString);
			break;
		    case(24):
			experiment.setCompilerJavaDirpath(tmpString);
			break;
		    case(25):
			experiment.setCompilerJavaVersion(tmpString);
			break;
		    case(26):
			experiment.setCompilerUserData(tmpString);
			break;
		    }
		    defaultTreeModel.nodeChanged(experiment.getDMTN());
		    break;
		case 2:
		    switch(r){
		    case(0):
			trial.setName(tmpString);
			break;
		    case(8):
			trial.setUserData(tmpString);
			break;
		    case(9):
			trial.setProblemDefinition(tmpString);
			break;
		    }
		    defaultTreeModel.nodeChanged(trial.getDMTN());
		    break;
		}
	    }
	}
    }
  
    private int type = -1; //0-application table model,1-experiment table model,2-trial table model.
    private ParaProfApplication application = null;
    private ParaProfExperiment experiment = null;
    private ParaProfTrial trial = null;
    private Metric metric = null;
    private DefaultTreeModel defaultTreeModel = null;
    String[] columnNames = {
	"Field", "Value"
    };
  
}
