package edu.uoregon.tau.paraprof.tablemodel;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import javax.swing.table.AbstractTableModel;
import javax.swing.tree.DefaultTreeModel;

import edu.uoregon.tau.paraprof.ParaProfView;
import edu.uoregon.tau.paraprof.ParaProfManagerWindow;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.taudb.TAUdbDatabaseAPI;

public class ViewTableModel extends AbstractTableModel {

    /**
	 * 
	 */
	private static final long serialVersionUID = 7691740386199068018L;
	private ParaProfView view;
    private String[] columnNames = { "View Field", "Value" };
    private ParaProfManagerWindow paraProfManager;
    private DefaultTreeModel defaultTreeModel;
    private List<String> fieldNames;
    
    public ViewTableModel(ParaProfManagerWindow paraProfManager, ParaProfView exp, DefaultTreeModel defaultTreeModel) {
        this.view = exp;

        this.paraProfManager = paraProfManager;
        this.defaultTreeModel = defaultTreeModel;
        
        fieldNames = new ArrayList<String>();
        for (int i=0; i<view.getNumFields(); i++) {
            fieldNames.add(view.getFieldName(i));
        }
    }
    public void updateDatabaseFields(ParaProfView exp)
    {
    	if (exp != null)
    	{
    		fieldNames = new ArrayList<String>();
    		
	        for (int i=0; i<view.getNumFields(); i++) {
	            fieldNames.add(view.getFieldName(i));
	        }
	        for (int i=0; i<exp.getNumFields(); i++) {
	            fieldNames.add(exp.getFieldName(i));
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
            return view.getField(r);
        }
    }

    public boolean isCellEditable(int r, int c) {
        if (c == 1 && r != 1 && r != 2) {
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

        view.setField(r, string);
        this.updateDB();
        defaultTreeModel.nodeChanged(view.getDMTN());
    }

    private void updateDB() {
        DatabaseAPI databaseAPI = paraProfManager.getDatabaseAPI(view.getDatabase());
        if (databaseAPI != null) {
            try {
				view.saveView(databaseAPI.db());
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
            databaseAPI.terminate();
        }
    }
}
