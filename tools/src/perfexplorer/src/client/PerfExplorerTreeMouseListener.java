package edu.uoregon.tau.perfexplorer.client;

import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import edu.uoregon.tau.perfdmf.Metric;

public class PerfExplorerTreeMouseListener implements MouseListener {

	//private PerfExplorerJTree tree;

	public PerfExplorerTreeMouseListener(PerfExplorerJTree tree) {
		super();
		//this.tree = tree;
	}

	public void mouseClicked(MouseEvent e) {
		//if (userObject instanceof ParaProfMetric) {
    //      ParaProfMetric ppMetric = (ParaProfMetric) userObject;
	
  //  if (e.getClickCount() == 2) {
   
	Object selected = PerfExplorerModel.getModel().getCurrentSelection();
	if(selected != null){
	if(selected instanceof Metric){
		PerfExplorerJTabbedPane pane = PerfExplorerJTabbedPane.getPane();
		int index = pane.getSelectedIndex();
		if (index == 4) {
			DeriveMetricsPane.getPane().metricClick((Metric)selected);
		}
	       
	}
	}
 

	}

	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub

	}

}
