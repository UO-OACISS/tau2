package edu.uoregon.tau.perfexplorer.client;

import java.sql.SQLException;
import java.text.DecimalFormat;
import java.text.FieldPosition;
import java.util.Iterator;
import java.util.Map.Entry;

import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTable;
import javax.swing.table.AbstractTableModel;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.MetaDataMap.MetaDataKey;
import edu.uoregon.tau.common.MetaDataMap.MetaDataValue;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.taudb.TAUdbDataSource;
import edu.uoregon.tau.perfdmf.taudb.TAUdbTrial;
import edu.uoregon.tau.perfexplorer.common.RMISortableIntervalEvent;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;

public class PerfExplorerTableModel extends AbstractTableModel {

    /**
	 * 
	 */
	private static final long serialVersionUID = -8929532025416520825L;
	private Application application = null;
    private Experiment experiment = null;
    private Trial trial = null;
    private Metric metric = null;
    private RMISortableIntervalEvent event = null;
    private FunctionProfile ilp = null;
    private View view = null;
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
        } else if (object instanceof TAUdbTrial) {
            this.trial = (Trial) object;
            TAUdbTrial localTrial = (TAUdbTrial)this.trial;

			DatabaseAPI session = PerfExplorerServer.getServer().getSession();
			try {
				session.setTrial(trial.getID(), true);
			} catch (DatabaseException e) {
			}
			DataSource dbDataSource = new TAUdbDataSource(session);

			localTrial.setDataSource(dbDataSource);
            //localTrial.setDataSource(PerfExplorerServer.getServer().getSession());
            if(!localTrial.hasMetadata()){
				localTrial.loadMetadata(PerfExplorerServer.getServer().getDB());
            }
            type = 2;
        } else if (object instanceof Trial) {
            this.trial = (Trial) object;
            if(!trial.isXmlMetaDataLoaded()){
				try {
					trial.loadXMLMetadata(PerfExplorerServer.getServer().getDB());
				} catch (SQLException e) {
					e.printStackTrace();
				}
            }
            type = 2;
        } else if (object instanceof Metric) {
            this.metric = (Metric) object;
            type = 3;
        } else if (object instanceof RMISortableIntervalEvent) {
            this.event = (RMISortableIntervalEvent) object;
//            try {
                ilp = event.getMeanSummary();
//            } catch (SQLException exception) {}
            type = 4;
        } else if (object instanceof View) {
            this.view = (View) object;
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
            return View.getFieldCount();
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
			if (trial instanceof TAUdbTrial && trial.getID() != currentTrial) {
				TAUdbTrial dbTrial = (TAUdbTrial) trial;
				int numFields = 0;
				MetaDataMap mdm = dbTrial.getMetaData();
				if (mdm != null) {
					numFields += mdm.size();
				}
				// Map<String, String> pmd = dbTrial.getPrimaryMetadata();
				// if (pmd != null) {
				// numFields += pmd.size();
				// }
				MetaDataMap umd = dbTrial.getUncommonMetaData();
				if (umd != null) {
					numFields += umd.size();
				}
				String[][] data = new String[numFields][2];
				String[] names = { "Name", "Value" };
				int dex = 0;
				if (mdm != null) {
					Iterator<Entry<MetaDataKey, MetaDataValue>> it = mdm
							.entrySet().iterator();
					while (it.hasNext()) {
						Entry<MetaDataKey, MetaDataValue> en = it.next();
						data[dex][0] = en.getKey().toString();
						data[dex][1] = en.getValue().toString();
						dex++;
					}
				}

				// if (pmd != null) {
				// Iterator<Entry<String, String>> it = pmd.entrySet()
				// .iterator();
				// while (it.hasNext()) {
				// Entry<String, String> en = it.next();
				// data[dex][0] = en.getKey();
				// data[dex][1] = en.getValue();
				// dex++;
				// }
				// }

				if (umd != null) {
					Iterator<Entry<MetaDataKey, MetaDataValue>> it = umd
							.entrySet().iterator();
					while (it.hasNext()) {
						Entry<MetaDataKey, MetaDataValue> en = it.next();
						data[dex][0] = en.getKey().toString();
						data[dex][1] = en.getValue().toString();
						dex++;
					}
				}

				JTable mdTable = new JTable(data, names);
				JScrollPane treeView = new JScrollPane(mdTable);
				setupMetadataTable(treeView);
			}
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
                    //System.out.println("field " + (r-2) + " is " + trial.getField(r-2));
                    if (trial.getField(r - 2) != null) {
                        if (trial.getFieldName(r - 2).equalsIgnoreCase("XML_METADATA") && trial.getID() != currentTrial) {
                            try {

								SAXTreeViewer viewer = new SAXTreeViewer();
                                JScrollPane treeView = new JScrollPane(viewer.getTreeTable(trial.getField(r - 2)));
								setupMetadataTable(treeView);

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
                    return "Inclusive";
                case (8):
                    return "Inclusive Per Call";
//                case (9):
//              return "Exclusive Percentage";
//                case (10):
//                    return "Inclusive Percentage";
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
                    return event.getFunction().getName();
                case (1):
                    return new Integer(event.getFunction().getID());
                case (2):
                    return event.getFunction().getGroupString();
                case (3):
                    return new Integer(event.getTrialID());
                case (4):
                    intFormat.format(ilp.getNumCalls(), s, f);
                    return s.toString();
                case (5):
                    intFormat.format(ilp.getNumSubr(), s, f);
                    return s.toString();
                case (6):
                    doubleFormat.format(ilp.getExclusive(event.metricIndex), s, f);
                    return s.toString();
                case (7):
                    doubleFormat.format(ilp.getInclusive(event.metricIndex), s, f);
                    return s.toString();
                case (8):
                    double inc = ilp.getInclusive(event.metricIndex);
                	double call = ilp.getNumCalls();
                    doubleFormat.format(inc/call, s, f);
                    return s.toString();
/*                case (9):
                    //doubleFormat.format(ilp.getExclusivePercent(event.metricIndex), s, f);
                    s.append("%");
                    return s.toString();
                case (10):
                    //doubleFormat.format(ilp.getInclusivePercent(event.metricIndex), s, f);
                    s.append("%");
                    return s.toString(); */
                default:
                    return "";
                }
            }
        case 5:
            if (c == 0) {
                if (View.getFieldName(r) != null)
                    return View.getFieldName(r);
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

	private void setupMetadataTable(JScrollPane treeView) {
		// This is a trial with XML data, so in the bottom half of the split
		// pane, put the XML data in a tree viewer.
		JSplitPane tab = (JSplitPane) PerfExplorerJTabbedPane.getPane().getTab(
				0);
		JScrollBar jScrollBar = treeView.getVerticalScrollBar();
		jScrollBar.setUnitIncrement(35);
		tab.setBottomComponent(treeView);
		// restore the divider location from the last time we displayed
		// XML data to the user
		tab.setDividerLocation(dividerLocation);
		currentTrial = trial.getID();
	}

    public boolean isCellEditable(int r, int c) {
        return false;
    }
}
