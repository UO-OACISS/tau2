/* 
 Metric.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import edu.uoregon.tau.dms.dss.*;
import javax.swing.tree.*;

public class ParaProfMetric extends Metric implements ParaProfTreeNodeUserObject {
    public ParaProfMetric() {
    }

    public void setTrial(ParaProfTrial trial) {
        this.trial = trial;
    }

    public ParaProfTrial getTrial() {
        return trial;
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

    public void setDBMetric(boolean dBMetric) {
        this.dBMetric = dBMetric;
    }

    public boolean dBMetric() {
        return dBMetric;
    }

    public void setDerivedMetric(boolean derivedMetric) {
        this.derivedMetric = derivedMetric;
    }

    public boolean getDerivedMetric() {
        return derivedMetric;
    }

    public int getApplicationID() {
        return trial.getApplicationID();
    }

    public int getExperimentID() {
        return trial.getExperimentID();
    }

    public int getTrialID() {
        return trial.getID();
    }

    public String getIDString() {
        if (trial != null)
            return trial.getIDString() + ":" + this.getID() + " - " + this.getName();
        else
            return ":" + this.getID() + " - " + this.getName();
    }

    public String toString() {
        return super.getName();
    }

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

    private ParaProfTrial trial = null;
    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean dBMetric = false;
    private boolean derivedMetric = false;
}