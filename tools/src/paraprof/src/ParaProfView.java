/* 
 ParaProfView.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.util.List;
import java.util.ListIterator;
import java.util.Vector;

import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.perfdmf.View;

public class ParaProfView extends View implements ParaProfTreeNodeUserObject {

    /**
	 * 
	 */
	private static final long serialVersionUID = 1662692425763662408L;
	private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private Vector<ParaProfTrial> trials = new Vector<ParaProfTrial>();
    private Vector<ParaProfView> subViews = new Vector<ParaProfView>();

    public ParaProfView() {
        super();
    }

    public ParaProfView(View view) {
        super(view);
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

    public ParaProfTrial getTrial(int trialID) {
        return trials.elementAt(trialID);
    }

    public ParaProfView getView(int viewID) {
        return subViews.elementAt(viewID);
    }

    public List<ParaProfTrial> getTrials() {
        return trials;
    }

    public List<ParaProfView> getSubViews() {
        return subViews;
    }

    public ListIterator<ParaProfTrial> getTrialList() {
        return trials.listIterator();
    }

    public ListIterator<ParaProfView> getViewList() {
        return subViews.listIterator();
    }

    public ParaProfView addView() {
        ParaProfView view = new ParaProfView();
        view.setParent(this);
        subViews.add(view);
        return view;
    }

    public ParaProfTrial addTrial() {
        ParaProfTrial trial = new ParaProfTrial();
        trial.setView(this);
        trial.setID((trials.size()));
        trials.add(trial);
        return trial;
    }

    public void removeTrial(ParaProfTrial trial) {
        trials.remove(trial);
    }

    public void removeView(ParaProfView view) {
        subViews.remove(view);
    }

    public String getIDString() {
        return Integer.toString(this.getID());
    }

    public String toString() {
        return super.getField("NAME");
    }

    public void clearDefaultMutableTreeNode() {
        this.setDMTN(null);
    }

	public void setField(int r, String string) {
	}

}
