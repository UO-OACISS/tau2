package edu.uoregon.tau.perfexplorer.client;

import javax.swing.*;

import java.awt.event.*;

public class PerfExplorerMainJMenuBar extends JMenuBar {

	public PerfExplorerMainJMenuBar(ActionListener listener) {
		super();
		createFileMenu(listener);
		createAnalysisMenu(listener);
		createViewMenu(listener);
		createChartMenu(listener);
		createVisualizationMenu(listener);
		createHelpMenu(listener);
	}

	private void createFileMenu(ActionListener listener) {
		//File menu.
		JMenu fileMenu = new JMenu("File");
		fileMenu.setMnemonic(KeyEvent.VK_F);

		//Add a menu item.
		JMenuItem profileItem = new JMenuItem(
				PerfExplorerActionListener.LOAD_PROFILE);
		profileItem.addActionListener(listener);
		fileMenu.add(profileItem);
	
		//Add a menu item.
		JMenuItem deriveMetricItem = new JMenuItem(
				PerfExplorerActionListener.DERIVE_METRIC);
		deriveMetricItem.addActionListener(listener);
		fileMenu.add(deriveMetricItem);
		
		//Add a menu item.
		JMenuItem parseItem = new JMenuItem(
				PerfExplorerActionListener.PARSE_EXPRESSION);
		parseItem.addActionListener(listener);
		fileMenu.add(parseItem);
		
		//Add a menu item.
		JMenuItem reparseItem = new JMenuItem(
				PerfExplorerActionListener.REPARSE_EXPRESSION);
		reparseItem.addActionListener(listener);
		fileMenu.add(reparseItem);

		

		//Add a menu item.
		JMenuItem scriptItem = new JMenuItem(
				PerfExplorerActionListener.LOADSCRIPT);
		scriptItem.addActionListener(listener);
		fileMenu.add(scriptItem);

		//Add a menu item.
		JMenuItem rerunItem = new JMenuItem(
				PerfExplorerActionListener.RERUNSCRIPT);
		rerunItem.addActionListener(listener);
		fileMenu.add(rerunItem);
		
		fileMenu.add(new JSeparator());
		
		
		
		//Add a menu item.
		JMenuItem databaseItem = new JMenuItem(
				PerfExplorerActionListener.DATABASE_CONFIGURATION);
		databaseItem.addActionListener(listener);
		fileMenu.add(databaseItem);
		

		//Add a menu item.
		JMenuItem consoleItem = new JMenuItem(
				PerfExplorerActionListener.CONSOLE);
		consoleItem.addActionListener(listener);
		fileMenu.add(consoleItem);

		//Add a menu item.
		JMenuItem quitItem = new JMenuItem(
				PerfExplorerActionListener.QUIT);
		quitItem.setAccelerator(KeyStroke.getKeyStroke(
			//KeyEvent.VK_Q, ActionEvent.SHIFT_MASK & ActionEvent.ALT_MASK));
			KeyEvent.VK_Q, ActionEvent.ALT_MASK));
		quitItem.addActionListener(listener);
		fileMenu.add(quitItem);

/*
		//Add a menu item.
		JMenuItem quitServerItem = new JMenuItem(
				PerfExplorerActionListener.QUIT_SERVER);
		quitServerItem.setAccelerator(KeyStroke.getKeyStroke(
			KeyEvent.VK_Q, ActionEvent.ALT_MASK));
		quitServerItem.addActionListener(listener);
		fileMenu.add(quitServerItem);
*/

/*
		try {
			UIManager.LookAndFeelInfo[] info = UIManager.getInstalledLookAndFeels();
			JMenuItem item[] = new JMenuItem[info.length];
			for (int i = 0 ; i < info.length ; i++) {
				//System.out.println(info[i].getClassName() + ": " + info[i].getName());
				//Add a menu item.
				item[i] = new JMenuItem(PerfExplorerActionListener.LOOK_AND_FEEL + info[i].getName());
				item[i].addActionListener(listener);
				fileMenu.add(item[i]);
			}
		} catch (Exception e) { }
*/

		this.add(fileMenu);
	}

	private void createHelpMenu(ActionListener listener) {
		//Help Menu
		JMenu helpMenu = new JMenu("Help");
		helpMenu.setMnemonic(KeyEvent.VK_H);

		//Add a menu item.
		JMenuItem aboutItem = new JMenuItem(
			PerfExplorerActionListener.ABOUT, KeyEvent.VK_A);
		aboutItem.addActionListener(listener);
		helpMenu.add(aboutItem);

		//Add a menu item.
		JMenuItem showHelpWindowItem = new JMenuItem(
			PerfExplorerActionListener.SEARCH, KeyEvent.VK_H);
		showHelpWindowItem.setAccelerator(KeyStroke.getKeyStroke(
			KeyEvent.VK_H, ActionEvent.ALT_MASK));
		showHelpWindowItem.addActionListener(listener);
		helpMenu.add(showHelpWindowItem);

		this.add(helpMenu);
	}

	private void createVisualizationMenu(ActionListener listener) {
		//Help Menu
		JMenu visualizationMenu = new JMenu("Visualization");
		visualizationMenu.setMnemonic(KeyEvent.VK_Z);

		//Add a menu item.
		JMenuItem cubeItem = new JMenuItem(
				PerfExplorerActionListener.DO_CORRELATION_CUBE);
		cubeItem.addActionListener(listener);
		visualizationMenu.add(cubeItem);

		//Add a menu item.
		JMenuItem varianceItem = new JMenuItem(
				PerfExplorerActionListener.DO_VARIATION_ANALYSIS);
		varianceItem.addActionListener(listener);
		visualizationMenu.add(varianceItem);

		//Add a menu item.
		JMenuItem iqrItem = new JMenuItem(
				PerfExplorerActionListener.DO_IQR_BOXCHART);
		iqrItem.addActionListener(listener);
		visualizationMenu.add(iqrItem);
		
		//Add a menu item.
		JMenuItem histogramItem = new JMenuItem(
				PerfExplorerActionListener.DO_HISTOGRAM);
		histogramItem.addActionListener(listener);
		visualizationMenu.add(histogramItem);

		//Add a menu item.
		JMenuItem probabilityPlotItem = new JMenuItem(
				PerfExplorerActionListener.DO_PROBABILITY_PLOT);
		probabilityPlotItem.addActionListener(listener);
		visualizationMenu.add(probabilityPlotItem);
		
		//Add a menu item.
		JMenuItem commMatrixItem = new JMenuItem(
				PerfExplorerActionListener.DO_COMMUNICATION_MATRIX);
		commMatrixItem.addActionListener(listener);
		visualizationMenu.add(commMatrixItem);
		
		this.add(visualizationMenu);
	}

	private void createAnalysisMenu(ActionListener listener) {
		//Analysis menu.
		JMenu analysisMenu = new JMenu("Analysis");
		analysisMenu.setMnemonic(KeyEvent.VK_A);

		//Add a menu item.
		JMenuItem clusteringItem = new JMenuItem(
				PerfExplorerActionListener.CLUSTERING_METHOD);
		clusteringItem.addActionListener(listener);
		clusteringItem.setEnabled(false);
		analysisMenu.add(clusteringItem);

		//Add a menu item.
		JMenuItem dimensionItem = new JMenuItem(
				PerfExplorerActionListener.DIMENSION_REDUCTION);
		dimensionItem.addActionListener(listener);
		analysisMenu.add(dimensionItem);

		//Add a menu item.
		JMenuItem normalizationItem = new JMenuItem(
				PerfExplorerActionListener.NORMALIZATION);
		normalizationItem.addActionListener(listener);
		normalizationItem.setEnabled(false);
		analysisMenu.add(normalizationItem);

		//Add a menu item.
		JMenuItem numClustersItem = new JMenuItem(
				PerfExplorerActionListener.NUM_CLUSTERS);
		numClustersItem.addActionListener(listener);
		analysisMenu.add(numClustersItem);

		analysisMenu.add(new JSeparator());

		//Add a menu item.
		JMenuItem doItem = new JMenuItem(
				PerfExplorerActionListener.DO_CLUSTERING);
		doItem.addActionListener(listener);
		analysisMenu.add(doItem);
		
		//Add a menu item.
		JMenuItem doIncItem = new JMenuItem(
				PerfExplorerActionListener.DO_INC_CLUSTERING);
		doIncItem.addActionListener(listener);
		analysisMenu.add(doIncItem);

		//Add a menu item.
		JMenuItem correlateItem = new JMenuItem(
				PerfExplorerActionListener.DO_CORRELATION_ANALYSIS);
		correlateItem.addActionListener(listener);
		analysisMenu.add(correlateItem);

		//Add a menu item.
		JMenuItem correlateIncItem = new JMenuItem(
				PerfExplorerActionListener.DO_INC_CORRELATION_ANALYSIS);
		correlateIncItem.addActionListener(listener);
		analysisMenu.add(correlateIncItem);

		this.add(analysisMenu);


	}

	private void createChartMenu(ActionListener listener) {
		//Chart Menu
		JMenu chartMenu = new JMenu("Charts");
		chartMenu.setMnemonic(KeyEvent.VK_C);

		//Add a menu item.
		JMenuItem allInOneItem = new JMenuItem(
			PerfExplorerActionListener.DO_CHARTS);
		allInOneItem.addActionListener(listener);
		chartMenu.add(allInOneItem);

		//Add a menu item.
		JMenuItem groupNameItem = new JMenuItem(
			PerfExplorerActionListener.SET_GROUPNAME);
		groupNameItem.addActionListener(listener);
		chartMenu.add(groupNameItem);

		//Add a menu item.
		JMenuItem metricNameItem = new JMenuItem(
			PerfExplorerActionListener.SET_METRICNAME);
		metricNameItem.addActionListener(listener);
		chartMenu.add(metricNameItem);

		//Add a menu item.
		JMenuItem eventNameItem = new JMenuItem(
			PerfExplorerActionListener.SET_EVENTNAME);
		eventNameItem.addActionListener(listener);
		chartMenu.add(eventNameItem);

		//Add a menu item.
		JMenuItem timestepItem = new JMenuItem(
			PerfExplorerActionListener.SET_TIMESTEPS);
		timestepItem.addActionListener(listener);
		chartMenu.add(timestepItem);

		//Add a menu item.
		JMenuItem problemItem = new JMenuItem(
			PerfExplorerActionListener.SET_PROBLEM_SIZE);
		problemItem.addActionListener(listener);
		chartMenu.add(problemItem);

		//Add a menu item.
		JMenuItem stackedBar = new JMenuItem(
			PerfExplorerActionListener.STACKED_BAR_CHART);
		stackedBar.addActionListener(listener);
		chartMenu.add(stackedBar);
		
		//Add a menu item.
		JMenuItem aStackedBar = new JMenuItem(
			PerfExplorerActionListener.ALIGNED_STACKED_BAR_CHART);
		aStackedBar.addActionListener(listener);
		chartMenu.add(aStackedBar);

		//Add a menu item.
		JMenuItem totalTime = new JMenuItem(
			PerfExplorerActionListener.TOTAL_TIME_CHART);
		totalTime.addActionListener(listener);
		chartMenu.add(totalTime);

		//Add a menu item.
		JMenuItem timesteps = new JMenuItem(
			PerfExplorerActionListener.TIMESTEPS_CHART);
		timesteps.addActionListener(listener);
		chartMenu.add(timesteps);

		//Add a menu item.
		JMenuItem efficiency = new JMenuItem(
			PerfExplorerActionListener.EFFICIENCY_CHART);
		efficiency.addActionListener(listener);
		chartMenu.add(efficiency);

		//Add a menu item.
		JMenuItem efficiencyEvents = new JMenuItem(
			PerfExplorerActionListener.EFFICIENCY_EVENTS_CHART);
		efficiencyEvents.addActionListener(listener);
		chartMenu.add(efficiencyEvents);

		//Add a menu item.
		JMenuItem efficiencyOneEvent = new JMenuItem(
			PerfExplorerActionListener.EFFICIENCY_ONE_EVENT_CHART);
		efficiencyOneEvent.addActionListener(listener);
		chartMenu.add(efficiencyOneEvent);

		//Add a menu item.
		JMenuItem speedup = new JMenuItem(
			PerfExplorerActionListener.SPEEDUP_CHART);
		speedup.addActionListener(listener);
		chartMenu.add(speedup);

		//Add a menu item.
		JMenuItem speedupEvents = new JMenuItem(
			PerfExplorerActionListener.SPEEDUP_EVENTS_CHART);
		speedupEvents.addActionListener(listener);
		chartMenu.add(speedupEvents);

		//Add a menu item.
		JMenuItem speedupOneEvent = new JMenuItem(
			PerfExplorerActionListener.SPEEDUP_ONE_EVENT_CHART);
		speedupOneEvent.addActionListener(listener);
		chartMenu.add(speedupOneEvent);

		//Add a menu item.
		JMenuItem transpose = new JMenuItem(
			PerfExplorerActionListener.COMMUNICATION_CHART);
		transpose.addActionListener(listener);
		chartMenu.add(transpose);

		//Add a menu item.
		JMenuItem fraction = new JMenuItem(
			PerfExplorerActionListener.FRACTION_CHART);
		fraction.addActionListener(listener);
		chartMenu.add(fraction);

		//Add a menu item.
		JMenuItem correlation = new JMenuItem(
			PerfExplorerActionListener.CORRELATION_CHART);
		correlation.addActionListener(listener);
		chartMenu.add(correlation);

		//Add a menu item.
		JMenuItem efficiencyPhase = new JMenuItem(
			PerfExplorerActionListener.EFFICIENCY_PHASE_CHART);
		efficiencyPhase.addActionListener(listener);
		chartMenu.add(efficiencyPhase);

		//Add a menu item.
		JMenuItem speedupPhase = new JMenuItem(
			PerfExplorerActionListener.SPEEDUP_PHASE_CHART);
		speedupPhase.addActionListener(listener);
		chartMenu.add(speedupPhase);

		//Add a menu item.
		JMenuItem fractionPhases = new JMenuItem(
			PerfExplorerActionListener.FRACTION_PHASE_CHART);
		fractionPhases.addActionListener(listener);
		chartMenu.add(fractionPhases);

		this.add(chartMenu);
	}

	private void createViewMenu(ActionListener listener) {
		//File menu.
		JMenu viewMenu = new JMenu("Views");
		viewMenu.setMnemonic(KeyEvent.VK_V);

		//Add a menu item.
		JMenuItem createView = new JMenuItem(PerfExplorerActionListener.CREATE_NEW_VIEW);
		createView.addActionListener(listener);
		viewMenu.add(createView);

		//Add a menu item.
		JMenuItem createSubView = new JMenuItem(PerfExplorerActionListener.CREATE_NEW_SUB_VIEW);
		createSubView.addActionListener(listener);
		viewMenu.add(createSubView);

		//Add a menu item.
		JMenuItem deleteView = new JMenuItem(PerfExplorerActionListener.DELETE_CURRENT_VIEW);
		deleteView.addActionListener(listener);
		viewMenu.add(deleteView);

		this.add(viewMenu);
	}

}
