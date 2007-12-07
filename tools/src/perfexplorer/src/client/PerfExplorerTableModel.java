package client;

import edu.uoregon.tau.perfdmf.*;

import javax.swing.table.*;
import javax.swing.*;

import common.RMIView;
import common.RMISortableIntervalEvent;

import java.awt.Dimension;
import java.sql.SQLException;
import java.text.DecimalFormat;
import java.text.FieldPosition;

public class PerfExplorerTableModel extends AbstractTableModel {

    private Application application = null;
    private Experiment experiment = null;
    private Trial trial = null;
    private Metric metric = null;
    private RMISortableIntervalEvent event = null;
    private IntervalLocationProfile ilp = null;
    private RMIView view = null;
    private int type = -1;
    private String[] columnNames = { "Field", "Value" };
    private int currentTrial = 0;
    private int dividerLocation = 200;

    public PerfExplorerTableModel(Object object) {
        super();
        if (object != null)
            updateObject(object);
    }

    public void updateObject(Object object) {
        // get the divider location of the split pane, so we can restore it later for trials with XML data
        JSplitPane tab = (JSplitPane) PerfExplorerJTabbedPane.getPane().getTab(0);
        if (tab.getBottomComponent() != null)
            dividerLocation = tab.getDividerLocation();
        // if this is likely not a trial, so get rid of the bottom component of the split pane
        tab.setBottomComponent(null);
        currentTrial = 0;
        if (object instanceof Application) {
            this.application = (Application) object;
            type = 0;
        } else if (object instanceof Experiment) {
            this.experiment = (Experiment) object;
            type = 1;
        } else if (object instanceof Trial) {
            this.trial = (Trial) object;
            type = 2;
        } else if (object instanceof Metric) {
            this.metric = (Metric) object;
            type = 3;
        } else if (object instanceof RMISortableIntervalEvent) {
            this.event = (RMISortableIntervalEvent) object;
            try {
                ilp = event.getMeanSummary();
            } catch (SQLException exception) {}
            type = 4;
        } else if (object instanceof RMIView) {
            this.view = (RMIView) object;
            type = 5;
        }
        fireTableDataChanged();
    }

    public int getColumnCount() {
        return 2;
    }

    public int getRowCount() {
        switch (type) {
        case 0:
            return application.getNumFields() + 2;
        case 1:
            return experiment.getNumFields() + 2;
        case 2:
            return trial.getNumFields() + 2;
        case 3:
            return 3;
        case 4:
            return 11;
        case 5:
            return RMIView.getFieldCount();
        default:
            return 0;
        }
    }

    public String getColumnName(int c) {
        return columnNames[c];
    }

    public Object getValueAt(int r, int c) {
        switch (type) {
        case 0:
            if (c == 0) {
                switch (r) {
                case (0):
                    return "Name";
                case (1):
                    return "Application ID";
                default:
                    if (application.getFieldName(r - 2) != null)
                        return application.getFieldName(r - 2);
                    else
                        return "";
                }
            } else {
                switch (r) {
                case (0):
                    return application.getName();
                case (1):
                    return new Integer(application.getID());
                default:
                    if (application.getField(r - 2) != null)
                        return application.getField(r - 2);
                    else
                        return "";
                }
            }
        case 1:
            if (c == 0) {
                switch (r) {
                case (0):
                    return "Name";
                case (1):
                    return "Experiment ID";
                default:
                    if (experiment.getFieldName(r - 2) != null)
                        return experiment.getFieldName(r - 2);
                    else
                        return "";
                }
            } else {
                switch (r) {
                case (0):
                    return experiment.getName();
                case (1):
                    return new Integer(experiment.getID());
                default:
                    if (experiment.getField(r - 2) != null)
                        return experiment.getField(r - 2);
                    else
                        return "";
                }
            }
        case 2:
            if (c == 0) {
                switch (r) {
                case (0):
                    return "Name";
                case (1):
                    return "Trial ID";
                default:
                    if (trial.getFieldName(r - 2) != null)
                        return trial.getFieldName(r - 2);
                    else
                        return "";
                }
            } else {
                switch (r) {
                case (0):
                    return trial.getName();
                case (1):
                    return new Integer(trial.getID());
                default:
                    System.out.println("field " + (r-2) + " is " + trial.getField(r-2));
                    if (trial.getField(r - 2) != null) {
                        if (trial.getFieldName(r - 2).equalsIgnoreCase("XML_METADATA") && trial.getID() != currentTrial) {
                            try {
                                // This is a trial with XML data, so in the bottom half of the split
                                // pane, put the XML data in a tree viewer.
                                SAXTreeViewer viewer = new SAXTreeViewer();
                                JSplitPane tab = (JSplitPane) PerfExplorerJTabbedPane.getPane().getTab(0);
                                JScrollPane treeView = new JScrollPane(viewer.getTreeTable(trial.getField(r - 2)));
                                JScrollBar jScrollBar = treeView.getVerticalScrollBar();
                                jScrollBar.setUnitIncrement(35);
                                tab.setBottomComponent(treeView);
                                // restore the divider location from the last time we displayed
                                // XML data to the user
                                tab.setDividerLocation(dividerLocation);
                                currentTrial = trial.getID();
                                return trial.getFieldName(r - 2);
                            } catch (Exception e) {
                                System.err.println(e.getMessage());
                                e.printStackTrace();
                            }
                        } else {
                            return trial.getField(r - 2);
                        }
                    } else {
                        return "ted";
                    }
                }
            }
        case 3:
            if (c == 0) {
                switch (r) {
                case (0):
                    return "Name";
                case (1):
                    return "Metric ID";
                case (2):
                    return "Trial ID";
                default:
                    return "";
                }
            } else {
                switch (r) {
                case (0):
                    return metric.getName();
                case (1):
                    return new Integer(metric.getID());
                case (2):
                    return new Integer(metric.getTrialID());
                default:
                    return "";
                }
            }
        case 4:
            if (c == 0) {
                switch (r) {
                case (0):
                    return "Name";
                case (1):
                    return "Interval Event ID";
                case (2):
                    return "Group Name";
                case (3):
                    return "Trial ID";
                case (4):
                    return "Number of Calls";
                case (5):
                    return "Number of Subroutines";
                case (6):
                    return "Exclusive";
                case (7):
                    return "Exclusive Percentage";
                case (8):
                    return "Inclusive";
                case (9):
                    return "Inclusive Percentage";
                case (10):
                    return "Inclusive Per Call";
                default:
                    return "";
                }
            } else {
                DecimalFormat intFormat = new DecimalFormat("#,##0");
                DecimalFormat doubleFormat = new DecimalFormat("#,##0.00");
                FieldPosition f = new FieldPosition(0);
                StringBuffer s = new StringBuffer();
                switch (r) {
                case (0):
                    return event.getName();
                case (1):
                    return new Integer(event.getID());
                case (2):
                    return event.getGroup();
                case (3):
                    return new Integer(event.getTrialID());
                case (4):
                    intFormat.format(ilp.getNumCalls(), s, f);
                    return s.toString();
                case (5):
                    intFormat.format(ilp.getNumSubroutines(), s, f);
                    return s.toString();
                case (6):
                    doubleFormat.format(ilp.getExclusive(event.metricIndex), s, f);
                    return s.toString();
                case (7):
                    doubleFormat.format(ilp.getExclusivePercentage(event.metricIndex), s, f);
                    s.append("%");
                    return s.toString();
                case (8):
                    doubleFormat.format(ilp.getInclusive(event.metricIndex), s, f);
                    return s.toString();
                case (9):
                    doubleFormat.format(ilp.getInclusivePercentage(event.metricIndex), s, f);
                    s.append("%");
                    return s.toString();
                case (10):
                    doubleFormat.format(ilp.getInclusivePerCall(event.metricIndex), s, f);
                    return s.toString();
                default:
                    return "";
                }
            }
        case 5:
            if (c == 0) {
                if (RMIView.getFieldName(r) != null)
                    return RMIView.getFieldName(r);
                else
                    return "";
            } else {
                if (view.getField(r) != null)
                    return view.getField(r);
                else
                    return "";
            }
        default:
            return "";
        }
    }

    public boolean isCellEditable(int r, int c) {
        return false;
    }
}
