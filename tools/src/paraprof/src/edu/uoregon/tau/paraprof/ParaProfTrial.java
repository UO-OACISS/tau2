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

public class ParaProfTrial extends Trial implements ParaProfObserver, ParaProfTreeNodeUserObject {

    public ParaProfTrial(int type) {
        super(0);
        this.debug = UtilFncs.debug;

        this.setID(-1);
        this.setExperimentID(-1);
        this.setApplicationID(-1);
        this.setName("");
        // 	 this.setNodeCount(-1);
        // 	 this.setNumContextsPerNode(-1);
        // 	 this.setNumThreadsPerContext(-1);
        this.type = type;
    }

    public ParaProfTrial(Trial trial, int type) {
        super(trial);
        this.debug = UtilFncs.debug;
        this.type = type;
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

    public void setDefaultTrial(boolean defaultTrial) {
        this.defaultTrial = defaultTrial;
    }

    public boolean defaultTrial() {
        return defaultTrial;
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
            return (experiment.getIDString()) + ":" + (super.getID());
        else
            return ":" + (super.getID());
    }

    public ColorChooser getColorChooser() {
        return clrChooser;
    }

    public Preferences getPreferences() {
        return preferences;
    }

    //Used in many ParaProf windows for the title of the window.
    public String getTrialIdentifier(boolean reverse) {
        if (path != null) {
            if (reverse)
                return pathReverse;
            else
                return path;
        } else
            return "Application " + this.getApplicationID() + ", Experiment " + this.getExperimentID()
                    + ", Trial " + this.getID() + ".";
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
        return super.getName();
    }

    //    public ParaProfDataSession getParaProfDataSession() {
    //      return (ParaProfDataSession) dataSession;
    //  }

    //####################################
    //Interface code.
    //####################################

    //######
    //ParaProfTreeUserObject
    //######
    public void clearDefaultMutableTreeNodes() {
        this.setDMTN(null);
    }

    //######
    //End - ParaProfTreeUserObject
    //######

    //####################################
    //End - Interface code.
    //####################################

    //####################################
    //Functions that control the obtaining and the opening
    //and closing of the static main window for
    //this trial.
    //####################################
    public StaticMainWindow getStaticMainWindow() {
        return sMW;
    }

    public void showMainWindow() {

        if (sMW == null) {
            sMW = new StaticMainWindow(this, UtilFncs.debug);
            ParaProf.incrementNumWindows();
            sMW.setVisible(true);
            this.getSystemEvents().addObserver(sMW);
        } else {
            ParaProf.incrementNumWindows();
            this.getSystemEvents().addObserver(sMW);
            sMW.show();
        }
    }

    public void closeStaticMainWindow() {
        if (sMW != null) {
            this.getSystemEvents().deleteObserver(sMW);
            sMW.setVisible(false);
        }
    }

    //####################################
    //End - Functions that control the opening
    //and closing of the static main window for
    //this trial.
    //####################################

    public SystemEvents getSystemEvents() {
        return systemEvents;
    }

    //####################################
    //Interface for ParaProfMetrics.
    //####################################
    public void setSelectedMetricID(int selectedMetricID) {
        this.selectedMetricID = selectedMetricID;
    }

    public int getSelectedMetricID() {
        return selectedMetricID;
    }

    public boolean isTimeMetric() {
        String metricName = this.getMetricName(this.getSelectedMetricID());
        metricName = metricName.toUpperCase();
        if (metricName.indexOf("TIME") == -1)
            return false;
        else
            return true;
    }

    public boolean isDerivedMetric() {

        // We can't do this, HPMToolkit stuff has /'s and -'s all over the place
        //String metricName = this.getMetricName(this.getSelectedMetricID());
        //if (metricName.indexOf("*") != -1 || metricName.indexOf("/") != -1)
        //    return true;
        return this.getMetric(this.getSelectedMetricID()).getDerivedMetric();
    }

    //Overide this function.
    public Vector getMetrics() {
        return dataSource.getMetrics();
    }

    public DssIterator getMetricList() {
        return new DssIterator(this.getMetrics());
    }

    public int getNumberOfMetrics() {
        return dataSource.getNumberOfMetrics();
    }

    public ParaProfMetric getMetric(int metricID) {
        return (ParaProfMetric) dataSource.getMetric(metricID);
    }

    public String getMetricName(int metricID) {
        return dataSource.getMetricName(metricID);
    }

    public ParaProfMetric addMetric() {
        ParaProfMetric metric = new ParaProfMetric();
        dataSource.addMetric(metric);
        return metric;
    }

    //####################################
    //End - Interface for ParaProfMetrics.
    //####################################

    //####################################
    //Pass-though methods to the data session for this instance.
    //####################################
    public TrialData getTrialData() {
        return dataSource.getTrialData();
    }

    public NCT getNCT() {
        return dataSource.getNCT();
    }

    public boolean groupNamesPresent() {
        return dataSource.groupNamesPresent();
    }

    public boolean userEventsPresent() {
        return dataSource.userEventsPresent();
    }

    public boolean callPathDataPresent() {
        return dataSource.callPathDataPresent();
    }

    //Overides the parent getMaxNCTNumbers.
    public int[] getMaxNCTNumbers() {
        return dataSource.getMaxNCTNumbers();
    }

    public void setMeanData(int metricID) {

        //dataSource.setMeanDataOLD(metricID);

        dataSource.setMeanData(metricID, metricID);
        dataSource.getMeanData().setThreadDataAllMetrics();

    }

    //####################################
    //end - Pass-though methods to the data session for this instance.
    //####################################

    //######
    //ParaProfObserver interface.
    //######
    public void update(Object obj) throws DatabaseException {

        DataSource dataSource = (DataSource) obj;

        //Data session has finished loading. Call its terminate method to
        //ensure proper cleanup.
        //dataSession.terminate();

        // The dataSource has accumulated edu.uoregon.tau.dms.dss.Metrics.
        // Inside ParaProf, these need to be paraprof.Metrics.
        
        int numberOfMetrics = dataSource.getNumberOfMetrics();
        Vector metrics = new Vector();
        for (int i = 0; i < numberOfMetrics; i++) {
            ParaProfMetric metric = new ParaProfMetric();
            metric.setName(dataSource.getMetricName(i));
            metric.setID(i);
            metric.setTrial(this);
            metrics.add(metric);
        }

        //Now set the dataSource's metrics.
        dataSource.setMetrics(metrics);

        //Now set the trial's dataSource object to be this one.
        this.setDataSource(dataSource);

        // set the colors
        clrChooser.setColors(this, -1);

        //Set this trial's loading flag to false.
        this.setLoading(false);

        //upload to database if necessary

        //Check to see if this trial needs to be uploaded to the database.
        if (this.upload()) {
            DatabaseAPI databaseAPI = ParaProf.paraProfManager.getDBSession();
            if (databaseAPI != null) {
                this.setID(databaseAPI.saveParaProfTrial(this, -1));
                databaseAPI.terminate();
            }

            //Now safe to set this to be a dbTrial.
            this.setDBTrial(true);
        } else {

            ParaProf.paraProfManager.populateTrialMetrics(this);

        }

    }

    public void update() {
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    int type = -1;
    boolean defaultTrial = false;
    ParaProfExperiment experiment = null;
    DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean dBTrial = false;
    private boolean upload = false;
    private boolean loading = false;

    private SystemEvents systemEvents = new SystemEvents();
    private StaticMainWindow sMW = null;
    private ColorChooser clrChooser = new ColorChooser(this, null);
    private Preferences preferences = new Preferences(this, ParaProf.savedPreferences);

    private String path = null;
    private String pathReverse = null;
    private int selectedMetricID = 0;
    private Vector observers = new Vector();
    private boolean debug = false;
    //####################################
    //Instance data.
    //####################################
}