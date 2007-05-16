package edu.uoregon.tau.paraprof.tablemodel;

import java.util.ArrayList;
import java.util.List;

import javax.swing.table.AbstractTableModel;
import javax.swing.tree.DefaultTreeModel;

import edu.uoregon.tau.paraprof.ParaProfExperiment;
import edu.uoregon.tau.paraprof.ParaProfManagerWindow;
import edu.uoregon.tau.perfdmf.DatabaseAPI;

public class ExperimentTableModel extends AbstractTableModel {

    private ParaProfExperiment experiment;
    private String[] columnNames = { "ExpField", "Value" };
    private ParaProfManagerWindow paraProfManager;
    private DefaultTreeModel defaultTreeModel;
    private List fieldNames;
    
    public ExperimentTableModel(ParaProfManagerWindow paraProfManager, ParaProfExperiment exp, DefaultTreeModel defaultTreeModel) {
        this.experiment = exp;

        this.paraProfManager = paraProfManager;
        this.defaultTreeModel = defaultTreeModel;
        
        fieldNames = new ArrayList();
        fieldNames.add("Name");
        fieldNames.add("Application ID");
        fieldNames.add("Experiment ID");
        for (int i=0; i<experiment.getNumFields(); i++) {
            fieldNames.add(experiment.getFieldName(i));
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
                return experiment.getName();
            case (1):
                return new Integer(experiment.getApplicationID());
            case (2):
                return new Integer(experiment.getID());
            default:
                return experiment.getField(r - 3);
            }
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

        if (r == 0) {
            experiment.setName(string);
            this.updateDB();
        } else {
            experiment.setField(r - 3, string);
            this.updateDB();
        }
        defaultTreeModel.nodeChanged(experiment.getDMTN());
    }

    private void updateDB() {
        if (experiment.dBExperiment()) {
            DatabaseAPI databaseAPI = paraProfManager.getDatabaseAPI(experiment.getDatabase());
            if (databaseAPI != null) {
                databaseAPI.saveExperiment(experiment);
                databaseAPI.terminate();
            }
        }
    }
}
