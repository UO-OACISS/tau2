/* 
 ParaProfApplication.java

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

import edu.uoregon.tau.perfdmf.Application;

public class ParaProfApplication extends Application implements ParaProfTreeNodeUserObject {

    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean dBApplication = false;
    private Vector<ParaProfExperiment> experiments = new Vector<ParaProfExperiment>();

    public ParaProfApplication() {
        super();
        this.setID(-1);
        this.setName("");
    }

    public ParaProfApplication(Application application) {
        super(application);
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

    public void setDBApplication(boolean dBApplication) {
        this.dBApplication = dBApplication;
    }

    public boolean dBApplication() {
        return dBApplication;
    }

    public ParaProfExperiment getExperiment(int experimentID) {
        return experiments.elementAt(experimentID);
    }

    public Vector<ParaProfExperiment> getExperiments() {
        return experiments;
    }

    public ListIterator<ParaProfExperiment> getExperimentList() {
        return experiments.listIterator();
    }

    public ParaProfExperiment addExperiment() {
        ParaProfExperiment experiment = new ParaProfExperiment();
        experiment.setApplication(this);
        experiment.setApplicationID(this.getID());
        experiment.setID((experiments.size()));
        experiments.add(experiment);
        return experiment;
    }

    public void removeExperiment(ParaProfExperiment experiment) {
        experiments.remove(experiment);
    }

    public String getIDString() {
        return Integer.toString(this.getID());
    }

    public String toString() {
        return super.getName();
    }

    public void clearDefaultMutableTreeNode() {
        this.setDMTN(null);
    }

}
