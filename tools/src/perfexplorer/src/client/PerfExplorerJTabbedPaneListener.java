package edu.uoregon.tau.perfexplorer.client;

import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class PerfExplorerJTabbedPaneListener implements ChangeListener {

	public void stateChanged(ChangeEvent e) {
		PerfExplorerJTabbedPane pane = PerfExplorerJTabbedPane.getPane();
		int index = pane.getSelectedIndex();
		if (index == 0) {
			// update the managment view
			PerfExplorerTableModel model = (PerfExplorerTableModel)AnalysisManagementPane.getPane().getTable().getModel();
			model.updateObject(PerfExplorerModel.getModel().getCurrentSelection());
		} else if (index == 1) {
			// update the results view
			PerformanceExplorerPane.getPane().updateImagePanel();
		} else if (index == 2) {
			PerfExplorerCorrelationPane.getPane().updateImagePanel();
		} else if (index == 3){
			ChartPane.getPane().refreshDynamicControls(true, true, true);
			ChartPane.getPane().drawChart();
		} else {
			DeriveMetricsPane.getPane();//.refreshDynamicControls(true, true, true);
			//DeriveMetricsPane.getPane().drawChart();
		}
	}

}
