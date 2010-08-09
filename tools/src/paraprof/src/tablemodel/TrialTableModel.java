package edu.uoregon.tau.paraprof.tablemodel;

import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.swing.JTable;
import javax.swing.table.AbstractTableModel;
import javax.swing.tree.DefaultTreeModel;

import edu.uoregon.tau.paraprof.ParaProfManagerWindow;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DBConnector;

public class TrialTableModel extends AbstractTableModel {

    /**
	 * 
	 */
	private static final long serialVersionUID = -5610815635456731795L;
	private ParaProfTrial ppTrial;
    private Trial trial;
    private String[] columnNames = { "TrialField", "Value" };
    private ParaProfManagerWindow paraProfManager;
    private DefaultTreeModel defaultTreeModel;
    private List<String> fieldNames;

    private Map<String, String> metaData = new TreeMap<String, String>();

    public TrialTableModel(ParaProfManagerWindow paraProfManager, ParaProfTrial ppTrial, DefaultTreeModel defaultTreeModel) {
        this.ppTrial = ppTrial;
        this.trial = ppTrial.getTrial();
        this.paraProfManager = paraProfManager;
        this.defaultTreeModel = defaultTreeModel;

        fieldNames = new ArrayList<String>();
        fieldNames.add("Name");
        fieldNames.add("Application ID");
        fieldNames.add("Experiment ID");
        fieldNames.add("Trial ID");
        for (int i = 0; i < ppTrial.getTrial().getNumFields(); i++) {
            fieldNames.add(ppTrial.getTrial().getFieldName(i));
        }

        metaData.putAll(ppTrial.getTrial().getMetaData());
        metaData.putAll(ppTrial.getTrial().getUncommonMetaData());

        for (Iterator<String> it = metaData.keySet().iterator(); it.hasNext();) {
            String string = it.next();
            fieldNames.add(string);
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
            if (r == 0) {
                return ppTrial.getName();
            } else if (r == 1) {
                return new Integer(ppTrial.getApplicationID());
            } else if (r == 2) {
                return new Integer(ppTrial.getExperimentID());
            } else if (r == 3) {
                return new Integer(ppTrial.getID());
            }

            int field = r - 4;
            if (field < trial.getNumFields()) {
                return ppTrial.getTrial().getField(field);
            }
            return metaData.get(fieldNames.get(r));
        }
    }

    public boolean isCellEditable(int r, int c) {

        if (c != 1)
            return false;

        if (r == 0)
            return true;

        if (r >= 1 && r <= 3) // id, experiment, application
            return false;

        int field = r - 4;
        if (field < trial.getNumFields()) {
            return DBConnector.isWritableType(ppTrial.getTrial().getFieldType(r - 4));
        }
        return true;
    }

    public void setValueAt(Object obj, int r, int c) {
        if (c == 0)
            return;
        if (!(obj instanceof String)) {
            return;
        }
        String string = (String) obj;

        if (r == 0) {
            ppTrial.getTrial().setName(string);
        } else {

            int field = r - 4;
            if (field < trial.getNumFields()) {
                ppTrial.getTrial().setField(r - 4, string);
            } else {
                metaData.put(fieldNames.get(r), string);
                trial.getMetaData().put(fieldNames.get(r), string);
            }
        }

        this.updateDB();
        defaultTreeModel.nodeChanged(ppTrial.getDMTN());
    }

    private void updateDB() {
        if (ppTrial.dBTrial()) {
            DatabaseAPI databaseAPI = paraProfManager.getDatabaseAPI(ppTrial.getDatabase());
            if (databaseAPI != null) {
                databaseAPI.saveTrial(ppTrial.getTrial());
                databaseAPI.terminate();
            }
        }
    }

    public MouseListener getMouseListener(final JTable table) {
        return new MouseListener() {

            public void mouseClicked(MouseEvent e) {
//                if (ParaProfUtils.rightClick(e)) {
//                    int row = table.rowAtPoint(e.getPoint());
//                    int column = table.columnAtPoint(e.getPoint());
//                    //System.out.println("you clicked on (" + column + "," + row + ") = " + getValueAt(row, column));
//                }
            }

            public void mouseEntered(MouseEvent e) {

            }

            public void mouseExited(MouseEvent e) {
            }

            public void mousePressed(MouseEvent e) {
            }

            public void mouseReleased(MouseEvent e) {
            }
        };
    }

}
