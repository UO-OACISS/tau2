/**
 * ParaProfManagerTableModel
 * This is the tabel model for the right side of the manager window.
 * It allows the user to change the meta data associated with a trial.
 *  
 * 
 * <P>CVS $Id: ParaProfManagerTableModel.java,v 1.14 2009/09/10 00:13:47 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.14 $
 * @see		ParaProfManagerWindow
 */

package edu.uoregon.tau.paraprof;

import javax.swing.table.AbstractTableModel;
import javax.swing.tree.DefaultTreeModel;

import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.database.DBConnector;

public class ParaProfManagerTableModel extends AbstractTableModel {

    private int type = -1; //0-application table model,1-experiment table model,2-trial table model, 3-metric table model
    private ParaProfApplication application = null;
    private ParaProfExperiment experiment = null;
    private ParaProfTrial ppTrial = null;
    private ParaProfMetric metric = null;
    private ParaProfManagerWindow paraProfManager = null;
    private DefaultTreeModel defaultTreeModel = null;
    private String[] columnNames = { "Field", "Value" };

    public ParaProfManagerTableModel(ParaProfManagerWindow paraProfManager, Object obj, DefaultTreeModel defaultTreeModel) {
        super();

        if (obj instanceof ParaProfApplication) {
            this.application = (ParaProfApplication) obj;
            type = 0;
        } else if (obj instanceof ParaProfExperiment) {
            this.experiment = (ParaProfExperiment) obj;
            type = 1;
        } else if (obj instanceof ParaProfTrial) {
            this.ppTrial = (ParaProfTrial) obj;
            type = 2;
        } else {
            this.metric = (ParaProfMetric) obj;
            type = 3;
        }

        this.paraProfManager = paraProfManager;
        this.defaultTreeModel = defaultTreeModel;
    }

    public int getColumnCount() {
        return 2;
    }

    public int getRowCount() {
        switch (type) {
        case 0:
            return application.getNumFields() + 2; // +2 for name and id
        case 1:
            return experiment.getNumFields() + 3; // +2 for name, id, and applicationID
        case 2:
            return ppTrial.getTrial().getNumFields() + 4;
        case 3:
            return 5;
        default:
            return 0;
        }
    }

    public String getColumnName(int c) {
        return columnNames[c];
    }

    // c is which column (either 0 meaning the name of the column, or nonzero for the actual value)
    // r is which row
    public Object getValueAt(int r, int c) {
        try {
            switch (type) {
            case 0: // application metadata
                if (c == 0) {
                    switch (r) {
                    case (0):
                        return "Name";
                    case (1):
                        return "Application ID";

                    default:
                        return application.getFieldName(r - 2);

                    //return "";
                    }
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
            case 1: // expriment metadata
                if (c == 0) {
                    switch (r) {
                    case (0):
                        return "Name";
                    case (1):
                        return "Application ID";
                    case (2):
                        return "Experiment ID";
                    default:
                        return experiment.getFieldName(r - 3);
                    }
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
            case 2: // trial metadata
                if (c == 0) {
                    switch (r) {
                    case (0):
                        return "Name";
                    case (1):
                        return "Application ID";
                    case (2):
                        return "Experiment ID";
                    case (3):
                        return "Trial ID";
                    default:
                        return ppTrial.getTrial().getFieldName(r - 4);
                    }
                } else {
                    switch (r) {
                    case (0):
                        return ppTrial.getName();
                    case (1):
                        return new Integer(ppTrial.getTrial().getApplicationID());
                    case (2):
                        return new Integer(ppTrial.getTrial().getExperimentID());
                    case (3):
                        return new Integer(ppTrial.getTrial().getID());
                    default:
                        return ppTrial.getTrial().getField(r - 4);
                    }
                }
            case 3: // metric metadata
                if (c == 0) {
                    switch (r) {
                    case (0):
                        return "Name";
                    case (1):
                        return "Application ID";
                    case (2):
                        return "Experiment ID";
                    case (3):
                        return "Trial ID";
                    case (4):
                        return "Metric ID";
                    default:
                        return "";
                    }
                } else {
                    switch (r) {
                    case (0):
                        return metric.getName();
                    case (1):
                        return new Integer(metric.getApplicationID());
                    case (2):
                        return new Integer(metric.getExperimentID());
                    case (3):
                        return new Integer(metric.getTrialID());
                    case (4):
                        return new Integer(metric.getDbMetricID());
                    default:
                        return "";
                    }
                }
            default:
                return "";
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
            return "";
        }
    }

    public boolean isCellEditable(int r, int c) {
        try {
            switch (type) {
            case 0: // Application
                if (c == 1 && r != 1)
                    return true;
                else
                    return false;
            case 1: // Experiment
                if (c == 1 && r != 1 && r != 2)
                    return true;
                else
                    return false;

            case 2: // Trial

                if (c != 1)
                    return false;

                if (r == 0)
                    return true;

                if (r >= 1 && r <= 3) // id, experiment, application
                    return false;

                return DBConnector.isWritableType(ppTrial.getTrial().getFieldType(r - 4));

            case 3: // Metric
                return false;

            default:
                if (c == 1 && r != 1)
                    return true;
                else
                    return false;
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
            return false;
        }

    }

    public void setValueAt(Object obj, int r, int c) {
        try {
            //Should be getting a string.
            if (obj instanceof String) {
                String tmpString = (String) obj;
                if (c == 1) {
                    switch (type) {
                    case 0: // Application
                        switch (r) {
                        case (0):
                            application.setName(tmpString);
                            this.updateDB(application);
                            break;

                        default:
                            application.setField(r - 2, tmpString);
                            this.updateDB(application);
                            break;
                        }
                        defaultTreeModel.nodeChanged(application.getDMTN());
                        break;
                    case 1:
                        switch (r) {
                        case (0):
                            experiment.setName(tmpString);
                            this.updateDB(experiment);
                            break;
                        default:
                            experiment.setField(r - 3, tmpString);
                            this.updateDB(experiment);
                            break;
                        }
                        defaultTreeModel.nodeChanged(experiment.getDMTN());
                        break;
                    case 2:
                        switch (r) {
                        case (0):
                            ppTrial.getTrial().setName(tmpString);
                            this.updateDB(ppTrial);
                            break;
                        default:
                            ppTrial.getTrial().setField(r - 4, tmpString);
                            this.updateDB(ppTrial);
                        }
                        defaultTreeModel.nodeChanged(ppTrial.getDMTN());
                        break;
                    }
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }

    }

    private void updateDB(Object obj) {
        if (obj instanceof ParaProfApplication) {
            ParaProfApplication application = (ParaProfApplication) obj;
            if (application.dBApplication()) {

                DatabaseAPI databaseAPI = paraProfManager.getDatabaseAPI(application.getDatabase());
                if (databaseAPI != null) {
                    databaseAPI.saveApplication(application);
                    databaseAPI.terminate();
                }

            }
        } else if (obj instanceof ParaProfExperiment) {
            ParaProfExperiment experiment = (ParaProfExperiment) obj;
            if (experiment.dBExperiment()) {
                DatabaseAPI databaseAPI = paraProfManager.getDatabaseAPI(experiment.getDatabase());
                if (databaseAPI != null) {
                    databaseAPI.saveExperiment(experiment);
                    databaseAPI.terminate();
                }
            }
        } else if (obj instanceof ParaProfTrial) {
            ParaProfTrial ppTrial = (ParaProfTrial) obj;
            if (ppTrial.dBTrial()) {
                DatabaseAPI databaseAPI = paraProfManager.getDatabaseAPI(ppTrial.getDatabase());
                if (databaseAPI != null) {
                    databaseAPI.saveTrial(ppTrial.getTrial());
                    databaseAPI.terminate();
                }
            }
        }
    }

}
