package edu.uoregon.tau.paraprof.tablemodel;

import java.util.ArrayList;
import java.util.List;

import javax.swing.table.AbstractTableModel;
import javax.swing.tree.DefaultTreeModel;

import edu.uoregon.tau.paraprof.ParaProfApplication;
import edu.uoregon.tau.paraprof.ParaProfManagerWindow;
import edu.uoregon.tau.perfdmf.DatabaseAPI;

public class ApplicationTableModel extends AbstractTableModel {

    /**
	 * 
	 */
	private static final long serialVersionUID = 8379814794176096620L;
	private ParaProfApplication application;
    private String[] columnNames = { "AppField", "Value" };
    private ParaProfManagerWindow paraProfManager;
    private DefaultTreeModel defaultTreeModel;
    private List<String> fieldNames;
    
    public ApplicationTableModel(ParaProfManagerWindow paraProfManager, ParaProfApplication app, DefaultTreeModel defaultTreeModel) {
        this.application = app;

        this.paraProfManager = paraProfManager;
        this.defaultTreeModel = defaultTreeModel;
        
        fieldNames = new ArrayList<String>();
        fieldNames.add("Name");
        fieldNames.add("Application ID");
        for (int i=0; i<application.getNumFields(); i++) {
            fieldNames.add(application.getFieldName(i));
        }
    }
    public void updateDatabaseFields(ParaProfApplication app)
    {
    	if (app != null)
    	{
    		fieldNames = new ArrayList<String>();
    		
	        fieldNames.add("Name");
	        fieldNames.add("Application ID");
	        for (int i=0; i<application.getNumFields(); i++) {
	            fieldNames.add(application.getFieldName(i));
	        }
	        for (int i=0; i<app.getNumFields(); i++) {
	            fieldNames.add(app.getFieldName(i));
	        }
    	}
    }

    public int getColumnCount() {
        return 2;
    }

    public String getColumnName(int c) {
        return columnNames[c];
    }

    public int getRowCount() {
        return fieldNames.size();
    }

    public Object getValueAt(int r, int c) {
        if (c == 0) {
            return fieldNames.get(r); 
        } else {
            switch (r) {
            case (0):
                return application.getName();
            case (1):
                return new Integer(application.getID());
            default:
                return application.getField(r - 2);
            }
        }
    }

    public boolean isCellEditable(int r, int c) {
        if (c == 1 && r != 1) {
            return true;
        } else {
            return false;
        }
    }

    public void setValueAt(Object obj, int r, int c) {
        if (c == 0)
            return;
        if (!(obj instanceof String)) {
            return;
        }
        String string = (String) obj;

        if (r == 0) {
            application.setName(string);
            this.updateDB();
        } else {
            application.setField(r - 2, string);
            this.updateDB();
        }
        defaultTreeModel.nodeChanged(application.getDMTN());
    }

    private void updateDB() {
        if (application.dBApplication()) {
            DatabaseAPI databaseAPI = paraProfManager.getDatabaseAPI(application.getDatabase());
            if (databaseAPI != null) {
                databaseAPI.saveApplication(application);
                databaseAPI.terminate();
            }
        }
    }
}
