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

    public void setPpTrial(ParaProfTrial trial) {
        this.ppTrial = trial;
    }

    public ParaProfTrial getParaProfTrial() {
        return ppTrial;
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

    public void setDerivedMetric(boolean derivedMetric) {
        this.derivedMetric = derivedMetric;
    }

    public boolean getDerivedMetric() {
        return derivedMetric;
    }

    public int getApplicationID() {
        return ppTrial.getApplicationID();
    }

    public int getExperimentID() {
        return ppTrial.getExperimentID();
    }

    public int getTrialID() {
        return ppTrial.getID();
    }

    
    public boolean isTimeMetric() {
        String metricName = this.getName().toUpperCase();
        if (metricName.indexOf("TIME") == -1)
            return false;
        else
            return true;
    }

    
    public String getIDString() {
        if (ppTrial != null)
            return ppTrial.getIDString() + ":" + this.getID() + " - " + this.getName();
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

    private ParaProfTrial ppTrial = null;
    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean derivedMetric = false;
}