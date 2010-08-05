/* 
 ParaProfExperiment.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.util.Enumeration;
import java.util.ListIterator;
import java.util.Vector;

import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.perfdmf.Database;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.database.DB;

public class ParaProfExperiment extends Experiment implements ParaProfTreeNodeUserObject {
    private ParaProfApplication application = null;
    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean dBExperiment = false;
    private Vector<ParaProfTrial> trials = new Vector<ParaProfTrial>();

    public ParaProfExperiment() {
        super();
        setID(-1);
        setApplicationID(-1);
        setName("");
    }

    public ParaProfExperiment(DB db) throws DatabaseException {
        super();
        setID(-1);
        setApplicationID(-1);
        setName("");
    }

    public ParaProfExperiment(Experiment experiment) {
        super(experiment);
    }

    public void setApplication(ParaProfApplication application) {
        this.application = application;
    }

    public ParaProfApplication getApplication() {
        return application;
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

    public void setDBExperiment(boolean dBExperiment) {
        this.dBExperiment = dBExperiment;
    }

    public boolean dBExperiment() {
        return dBExperiment;
    }

    public Vector<ParaProfTrial> getTrials() {
        return trials;
    }

    public ListIterator<ParaProfTrial> getTrialList() {
        return trials.listIterator();
    }

    public ParaProfTrial getTrial(int trialID) {
        return trials.elementAt(trialID);
    }

    public void addTrial(ParaProfTrial ppTrial) {
        ppTrial.setExperiment(this);
        ppTrial.getTrial().setID((trials.size()));
        trials.add(ppTrial);
    }

    public void removeTrial(ParaProfTrial ppTrial) {
        trials.remove(ppTrial);
    }

    public boolean isTrialPresent(String name) {
        for (Enumeration<ParaProfTrial> e = trials.elements(); e.hasMoreElements();) {
            ParaProfTrial ppTrial = e.nextElement();
            if (name.equals(ppTrial.toString()))
                return true;
        }
        return false;
    }

    public String getIDString() {
        if (application != null)
            return (application.getIDString()) + ":" + (super.getID());
        else
            return ":" + (super.getID());
    }

    public String toString() {
        return super.getName();
    }

    public void clearDefaultMutableTreeNode() {
        this.setDMTN(null);
    }

}
