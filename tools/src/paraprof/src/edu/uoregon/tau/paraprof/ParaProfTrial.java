/*
 * ParaProfTrial.java
 * 
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description: The manner in which this
 * class behaves is slightly different from its parent. It behaves more as a
 * container for its DataSession, than a setting for it. So, in a sense the role
 * is almost reversed (but not quite). This is a result of the fact that ParaProf
 * must maintain the majority of its data itself, and as such, ParaProfTrial
 * serves as the reference through which data is accessed.
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import javax.swing.tree.*;

import edu.uoregon.tau.dms.dss.*;

public class ParaProfTrial implements ParaProfTreeNodeUserObject {

    private Function highlightedFunction = null;
    private Group highlightedGroup = null;
    private UserEvent highlightedUserEvent = null;

    private DatabaseAPI dbAPI;
    private ParaProfExperiment experiment = null;
    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean dBTrial = false;
    private boolean upload = false;
    private boolean loading = false;

    private SystemEvents systemEvents = new SystemEvents();
    private GlobalDataWindow fullDataWindow = null;
    private ColorChooser clrChooser = ParaProf.colorChooser;
    private PreferencesWindow preferencesWindow = ParaProf.preferencesWindow;

    private String path = null;
    private String pathReverse = null;
    private int defaultMetricID = 0;
    private Vector observers = new Vector();

    private Trial trial;

    public ParaProfTrial() {
        trial = new Trial();
        trial.setID(-1);
        trial.setExperimentID(-1);
        trial.setApplicationID(-1);
        trial.setName("");
    }

    public ParaProfTrial(Trial trial) {
        this.trial = new Trial(trial);
    }

    public Trial getTrial() {
        return trial;
    }

    public int getApplicationID() {
        return trial.getApplicationID();
    }

    public int getExperimentID() {
        return trial.getExperimentID();
    }

    public int getID() {
        return trial.getID();
    }

    public void setApplicationID(int id) {
        trial.setApplicationID(id);
    }

    public void setExperimentID(int id) {
        trial.setExperimentID(id);
    }

    public void setID(int id) {
        trial.setID(id);
    }

    public String getName() {
        return trial.getName();
    }

    public DataSource getDataSource() {
        return trial.getDataSource();
    }

    public void setExperiment(ParaProfExperiment experiment) {
        this.experiment = experiment;
    }

    public ParaProfExperiment getExperiment() {
        return experiment;
    }

    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode) {
        this.defaultMutableTreeNode = defaultMutableTreeNode;
    }

    public DefaultMutableTreeNode getDMTN() {
        return defaultMutableTreeNode;
    }

    public void setTreePath(TreePath treePath) {
        this.treePath = treePath;
    }

    public TreePath getTreePath() {
        return treePath;
    }

    public void setDBTrial(boolean dBTrial) {
        this.dBTrial = dBTrial;
    }

    public boolean dBTrial() {
        return dBTrial;
    }

    public void setUpload(boolean upload) {
        this.upload = upload;
    }

    public boolean upload() {
        return upload;
    }

    public void setLoading(boolean loading) {
        this.loading = loading;
    }

    public boolean loading() {
        return loading;
    }

    public String getIDString() {
        if (experiment != null)
            return (experiment.getIDString()) + ":" + (trial.getID());
        else
            return ":" + (trial.getID());
    }

    public ColorChooser getColorChooser() {
        return clrChooser;
    }

    public PreferencesWindow getPreferencesWindow() {
        return preferencesWindow;
    }

    //Used in many ParaProf windows for the title of the window.
    public String getTrialIdentifier(boolean reverse) {
        if (path != null) {
            if (reverse)
                return pathReverse;
            else
                return path;
        } else
            return "Application " + trial.getApplicationID() + ", Experiment " + trial.getExperimentID()
                    + ", Trial " + trial.getID() + ".";
    }

    //Sets both the path and the path reverse.
    public void setPaths(String path) {
        this.path = path;
        this.pathReverse = FileList.getPathReverse(path);
    }

    public String getPath() {
        return path;
    }

    public String getPathReverse() {
        return pathReverse;
    }

    public String toString() {
        if (this.loading()) {
            return trial.getName() + " (Loading...)";
        } else {
            return trial.getName();
        }
    }

    public void clearDefaultMutableTreeNodes() {
        this.setDMTN(null);
    }

    //####################################
    //Functions that control the obtaining and the opening
    //and closing of the static main window for this trial.
    //####################################
    public GlobalDataWindow getFullDataWindow() {
        return fullDataWindow;
    }

    public void showMainWindow() {
        if (fullDataWindow == null) {
            fullDataWindow = new GlobalDataWindow(this, trial.getDataSource().getTopLevelPhase());
            ParaProf.incrementNumWindows();
            fullDataWindow.setVisible(true);

        } else {
            ParaProf.incrementNumWindows();
            fullDataWindow.setVisible(true);
        }
    }

    public void closeStaticMainWindow() {
        if (fullDataWindow != null) {
            this.getSystemEvents().deleteObserver(fullDataWindow);
            fullDataWindow.setVisible(false);
        }
    }

    //####################################
    //End - Functions that control the opening
    //and closing of the static main window for this trial.
    //####################################

    public SystemEvents getSystemEvents() {
        return systemEvents;
    }

    //####################################
    //Interface for ParaProfMetrics.
    //####################################
    public void setDefaultMetricID(int selectedMetricID) {
        this.defaultMetricID = selectedMetricID;
    }

    public int getDefaultMetricID() {
        return defaultMetricID;
    }

    public boolean isTimeMetric() {
        String metricName = this.getMetricName(this.getDefaultMetricID());
        metricName = metricName.toUpperCase();
        if (metricName.indexOf("TIME") == -1)
            return false;
        else
            return true;
    }

    //    public boolean isTimeMetric(int metricID) {
    //        String metricName = this.getMetricName(metricID);
    //        metricName = metricName.toUpperCase();
    //        if (metricName.indexOf("TIME") == -1)
    //            return false;
    //        else
    //            return true;
    //    }

    public boolean isDerivedMetric() {

        // We can't do this, HPMToolkit stuff has /'s and -'s all over the place
        //String metricName = this.getMetricName(this.getSelectedMetricID());
        //if (metricName.indexOf("*") != -1 || metricName.indexOf("/") != -1)
        //    return true;
        return this.getMetric(this.getDefaultMetricID()).getDerivedMetric();
    }

    //Override this function.
    public List getMetrics() {
        return trial.getDataSource().getMetrics();
    }

    public int getNumberOfMetrics() {
        return trial.getDataSource().getNumberOfMetrics();
    }

    public ParaProfMetric getMetric(int metricID) {
        return (ParaProfMetric) trial.getDataSource().getMetric(metricID);
    }

    public String getMetricName(int metricID) {
        return trial.getDataSource().getMetricName(metricID);
    }

    public ParaProfMetric addMetric() {
        ParaProfMetric metric = new ParaProfMetric();
        trial.getDataSource().addMetric(metric);
        return metric;
    }

    //####################################
    //End - Interface for ParaProfMetrics.
    //####################################

    //####################################
    //Pass-though methods to the data session for this instance.
    //####################################

    //    public edu.uoregon.tau.dms.dss.Thread getThread(int nodeID, int contextID, int threadID) {
    //        return dataSource.getThread(nodeID, contextID, threadID);
    //    }

    public boolean groupNamesPresent() {
        return trial.getDataSource().getGroupNamesPresent();
    }

    public boolean userEventsPresent() {
        return trial.getDataSource().getUserEventsPresent();
    }

    public boolean callPathDataPresent() {
        return trial.getDataSource().getCallPathDataPresent();
    }

    //Overrides the parent getMaxNCTNumbers.
    public int[] getMaxNCTNumbers() {
        return trial.getDataSource().getMaxNCTNumbers();
    }

    public void setMeanData(int metricID) {
        trial.getDataSource().generateStatistics(metricID, metricID);
    }

    // return a vector of only those functions that are currently "displayed" (i.e. group masks, etc)
    public List getDisplayedFunctions() {
        List displayedFunctions = new ArrayList();

        for (Iterator it = this.getDataSource().getFunctions(); it.hasNext();) {
            Function function = (Function) it.next();
            if (this.displayFunction(function)) {
                displayedFunctions.add(function);
            }
        }
        return displayedFunctions;
    }

    public boolean displayFunction(Function func) {
        switch (groupFilter) {
        case 0:
            //No specific group selection is required.
            return true;
        case 1:
            //Show this group only.
            if (func.isGroupMember(this.selectedGroup))
                return true;
            else
                return false;
        case 2:
            //Show all groups except this one.
            if (func.isGroupMember(this.selectedGroup))
                return false;
            else
                return true;
        default:
            //Default case behaves as case 0.
            return true;
        }
    }

    public void setSelectedGroup(Group group) {
        this.selectedGroup = group;
    }

    public Group getSelectedGroup() {
        return selectedGroup;
    }

    public void setGroupFilter(int groupFilter) {
        this.groupFilter = groupFilter;
    }

    public int getGroupFilter() {
        return groupFilter;
    }

    private Group selectedGroup;
    private int groupFilter = 0;

    //####################################
    //end - Pass-though methods to the data session for this instance.
    //####################################

    public void finishLoad() {

        // The dataSource has accumulated edu.uoregon.tau.dms.dss.Metrics.
        // Inside ParaProf, these need to be ParaProfMetrics.

        int numberOfMetrics = trial.getDataSource().getNumberOfMetrics();
        Vector ppMetrics = new Vector();
        for (int i = 0; i < numberOfMetrics; i++) {
            ParaProfMetric ppMetric = new ParaProfMetric();
            ppMetric.setName(trial.getDataSource().getMetricName(i));
            ppMetric.setID(i);
            ppMetric.setPpTrial(this);
            ppMetrics.add(ppMetric);
        }

        
        // Now set the dataSource's metrics.
        trial.getDataSource().setMetrics(ppMetrics);

        
        // Set the default metric to the first time based metric (if it exists)
        for (int i = 0; i < numberOfMetrics; i++) {
            ParaProfMetric ppMetric = (ParaProfMetric) trial.getDataSource().getMetric(i);
            if (ppMetric.isTimeMetric()) {
                setDefaultMetricID(i);
                break;
            }
        }
        

        // Set the default metric to the first metric named "Time" (if it exists)
        for (int i = 0; i < numberOfMetrics; i++) {
            ParaProfMetric ppMetric = (ParaProfMetric) trial.getDataSource().getMetric(i);
            if (ppMetric.getName().equalsIgnoreCase("Time")) {
                setDefaultMetricID(i);
                break;
            }
        }

        
        
        // set the colors
        clrChooser.setColors(this, -1);

        //Set this trial's loading flag to false.
        this.setLoading(false);
    }

    public DatabaseAPI getDatabaseAPI() {
        return dbAPI;
    }

    public void setDatabaseAPI(DatabaseAPI dbAPI) {
        this.dbAPI = dbAPI;
    }

    public void setHighlightedFunction(Function func) {
        this.highlightedFunction = func;
        this.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    public Function getHighlightedFunction() {
        return this.highlightedFunction;
    }

    public void toggleHighlightedFunction(Function function) {
        if (highlightedFunction == function)
            highlightedFunction = null;
        else
            highlightedFunction = function;
        this.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    public void setHighlightedGroup(Group group) {
        this.highlightedGroup = group;
        this.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    public Group getHighlightedGroup() {
        return highlightedGroup;
    }

    public void toggleHighlightedGroup(Group group) {
        if (highlightedGroup == group)
            highlightedGroup = null;
        else
            highlightedGroup = group;
        this.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    //    private boolean uploading;

    public void setHighlightedUserEvent(UserEvent userEvent) {
        this.highlightedUserEvent = userEvent;
        this.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    public UserEvent getHighlightedUserEvent() {
        return highlightedUserEvent;
    }

    public void toggleHighlightedUserEvent(UserEvent userEvent) {
        if (highlightedUserEvent == userEvent)
            highlightedUserEvent = null;
        else
            highlightedUserEvent = userEvent;
        this.getSystemEvents().updateRegisteredObjects("colorEvent");
    }
}