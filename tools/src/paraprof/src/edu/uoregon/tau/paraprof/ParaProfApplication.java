/* 
 ParaProfApplication.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import javax.swing.tree.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.database.*;

public class ParaProfApplication extends Application implements ParaProfTreeNodeUserObject {

    public ParaProfApplication(DB db) {
        super(db);
        this.setID(-1);
        this.setName("");
    }

    public ParaProfApplication() {
        super(0);
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
        return (ParaProfExperiment) experiments.elementAt(experimentID);
    }

    public Vector getExperiments() {
        return experiments;
    }

    public ListIterator getExperimentList() {
        return new DssIterator(experiments);
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

    public boolean isExperimentPresent(String name) {
        for (Enumeration e = experiments.elements(); e.hasMoreElements();) {
            ParaProfExperiment exp = (ParaProfExperiment) e.nextElement();
            if (name.equals(exp.getName()))
                return true;
        }
        //If we make it here, the experiment run name is not present.  Return false.
        return false;
    }

    public String getIDString() {
        return Integer.toString(this.getID());
    }

    public String toString() {
        return super.getName();
    }

    public void clearDefaultMutableTreeNodes() {
        this.setDMTN(null);
    }

    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean dBApplication = false;
    private Vector experiments = new Vector();
}
