package edu.uoregon.tau.paraprof.tablemodel;

import java.util.ArrayList;
import java.util.List;

import javax.swing.table.AbstractTableModel;
import javax.swing.tree.DefaultTreeModel;

import edu.uoregon.tau.paraprof.ParaProfApplication;
import edu.uoregon.tau.paraprof.ParaProfManagerWindow;
import edu.uoregon.tau.paraprof.ParaProfMetric;

public class MetricTableModel extends AbstractTableModel {

    private ParaProfMetric metric;
    private String[] columnNames = { "MetricField", "Value" };
    private DefaultTreeModel defaultTreeModel;
    private List<String> fieldNames;
    private List<Comparable> fieldValues;

    public MetricTableModel(ParaProfManagerWindow paraProfManager, ParaProfMetric metric, DefaultTreeModel defaultTreeModel) {
        this.metric = metric;
        this.defaultTreeModel = defaultTreeModel;

        fieldNames = new ArrayList<String>();
        fieldNames.add("Name");
        fieldNames.add("Application ID");
        fieldNames.add("Experiment ID");
        fieldNames.add("Trial ID");
        fieldNames.add("Metric ID");

        fieldValues = new ArrayList<Comparable>();
        fieldValues.add(metric.getName());
        fieldValues.add(new Integer(metric.getApplicationID()));
        fieldValues.add(new Integer(metric.getExperimentID()));
        fieldValues.add(new Integer(metric.getTrialID()));
        fieldValues.add(new Integer(metric.getDbMetricID()));
    }

 
    public boolean isCellEditable(int r, int c) {
        if (c == 1 && r == 0) {
            return true;
        } else {
            return false;
        }
    }

    public void setValueAt(Object obj, int r, int c) {
        if (c == 0 || r != 0)
            return;
        if (!(obj instanceof String)) {
            return;
        }
        String string = (String) obj;

        fieldValues.set(0,string);
        metric.setName(string);
        defaultTreeModel.nodeChanged(metric.getDMTN());

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
	        if (r < fieldNames.size())
	        	return fieldNames.get(r);
	        else
	        	return "";
	    } else {
	        if (r < fieldValues.size())
	        	return fieldValues.get(r);
	        else
	        	return "";
	    }
        
    }
}