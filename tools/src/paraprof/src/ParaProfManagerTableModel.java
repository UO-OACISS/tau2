/* 
   ParaProfManagerTableModel.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  The container for the MappingDataWindowPanel.
*/

/*
  To do: 
*/

package paraprof;

import javax.swing.tree.*;
import javax.swing.table.*;

public class ParaProfManagerTableModel extends AbstractTableModel{
    public ParaProfManagerTableModel(ParaProfApplication application, DefaultTreeModel defaultTreeModel){
	super();
	this.application = application;
	this.defaultTreeModel = defaultTreeModel;
    }
  
    public int getColumnCount(){
	return 2;}
  
    public int getRowCount(){
	return 7;}
  
    public String getColumnName(int c){
	return columnNames[c];}
  
    public Object getValueAt(int r, int c){
	Object returnObject = null;
	if(c==0){
	    switch(r){
	    case(0):
		returnObject = "Name";
		break;
	    case(1):
		returnObject = "ID";
		break;
	    case(2):
		returnObject = "Language";
		break;
	    case(3):
		returnObject = "Para_diag";
		break;
	    case(4):
		returnObject = "Usage";
		break;
	    case(5):
		returnObject = "Executable Options";
		break;
	    case(6):
		returnObject = "Description";
		break;
	    }
	}
	else{
	    switch(r){
	    case(0):
		returnObject = application.getName();
		break;
	    case(1):
		returnObject = new Integer(application.getID());
		break;
	    case(2):
		returnObject = application.getLanguage();
		break;
	    case(3):
		returnObject = application.getParaDiag();
		break;
	    case(4):
		returnObject = application.getUsage();
		break;
	    case(5):
		returnObject = application.getExecutableOptions();
		break;
	    case(6):
		returnObject = application.getDescription();
		break;
	    }
	}
    
	return returnObject; 
          
    }
  
    public boolean isCellEditable(int r, int c){
	if(c==1 && r!=1)
	    return true;
	else
	    return false;
    }
  
    public void setValueAt(Object obj, int r, int c){
	//Should be getting a string I think.
	if(obj instanceof String){
	    String tmpString = (String) obj;
	    if(c==1){
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
	    }
	}
	defaultTreeModel.nodeChanged(application.getDMTN());
	//defaultTreeModel.reload(application.getDMTN());
    }
  
    private ParaProfApplication application = null;
    private DefaultTreeModel defaultTreeModel = null;
    String[] columnNames = {
	"Field", "Value"
    };
  
}
