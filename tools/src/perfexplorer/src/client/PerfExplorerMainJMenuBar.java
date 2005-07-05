package client;

import javax.swing.*;
import java.awt.event.*;

public class PerfExplorerMainJMenuBar extends JMenuBar {

	public PerfExplorerMainJMenuBar(ActionListener listener) {
		super();
		createFileMenu(listener);
		createAnalysisMenu(listener);
		createViewMenu(listener);
		createChartMenu(listener);
		createHelpMenu(listener);
	}

	private void createFileMenu(ActionListener listener) {
		//File menu.
		JMenu fileMenu = new JMenu("File");
		fileMenu.setMnemonic(KeyEvent.VK_F);

		//Add a menu item.
		JMenuItem menuItem = new JMenuItem(
				PerfExplorerActionListener.QUIT, KeyEvent.VK_Q);
		menuItem.setAccelerator(KeyStroke.getKeyStroke(
			KeyEvent.VK_Q, ActionEvent.ALT_MASK));
		menuItem.addActionListener(listener);
		fileMenu.add(menuItem);

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

	private void createAnalysisMenu(ActionListener listener) {
		//Analysis menu.
		JMenu analysisMenu = new JMenu("Analysis");
		analysisMenu.setMnemonic(KeyEvent.VK_A);

		//Add a menu item.
		JMenuItem clusteringItem = new JMenuItem(
				PerfExplorerActionListener.CLUSTERING_METHOD,
				KeyEvent.VK_M);
		clusteringItem.setAccelerator(KeyStroke.getKeyStroke(
				KeyEvent.VK_M, ActionEvent.ALT_MASK));
		clusteringItem.addActionListener(listener);
		analysisMenu.add(clusteringItem);

		//Add a menu item.
		JMenuItem dimensionItem = new JMenuItem(
				PerfExplorerActionListener.DIMENSION_REDUCTION,
				KeyEvent.VK_D);
		dimensionItem.setAccelerator(KeyStroke.getKeyStroke(
				KeyEvent.VK_D, ActionEvent.ALT_MASK));
		dimensionItem.addActionListener(listener);
		analysisMenu.add(dimensionItem);

		//Add a menu item.
		JMenuItem normalizationItem = new JMenuItem(
				PerfExplorerActionListener.NORMALIZATION,
				KeyEvent.VK_N);
		normalizationItem.setAccelerator(KeyStroke.getKeyStroke(
				KeyEvent.VK_N, ActionEvent.ALT_MASK));
		normalizationItem.addActionListener(listener);
		analysisMenu.add(normalizationItem);

		//Add a menu item.
		JMenuItem numClustersItem = new JMenuItem(
				PerfExplorerActionListener.NUM_CLUSTERS,
				KeyEvent.VK_S);
		numClustersItem.setAccelerator(KeyStroke.getKeyStroke(
				KeyEvent.VK_S, ActionEvent.ALT_MASK));
		numClustersItem.addActionListener(listener);
		analysisMenu.add(numClustersItem);

		analysisMenu.add(new JSeparator());

		//Add a menu item.
		JMenuItem doItem = new JMenuItem(
				PerfExplorerActionListener.DO_CLUSTERING,
				KeyEvent.VK_C);
		doItem.setAccelerator(KeyStroke.getKeyStroke(
				KeyEvent.VK_C, ActionEvent.ALT_MASK));
		doItem.addActionListener(listener);
		analysisMenu.add(doItem);

		//Add a menu item.
		JMenuItem correlateItem = new JMenuItem(
				PerfExplorerActionListener.DO_CORRELATION_ANALYSIS,
				KeyEvent.VK_C);
		correlateItem.setAccelerator(KeyStroke.getKeyStroke(
				KeyEvent.VK_C, ActionEvent.ALT_MASK));
		correlateItem.addActionListener(listener);
		analysisMenu.add(correlateItem);

		//Add a menu item.
		JMenuItem cubeItem = new JMenuItem(
				PerfExplorerActionListener.DO_CORRELATION_CUBE,
				null);
		//correlateItem.setAccelerator(KeyStroke.getKeyStroke(
				//KeyEvent.VK_C, ActionEvent.ALT_MASK));
		cubeItem.addActionListener(listener);
		analysisMenu.add(cubeItem);

		//Add a menu item.
		JMenuItem varianceItem = new JMenuItem(
				PerfExplorerActionListener.DO_VARIATION_ANALYSIS,
				null);
		varianceItem.addActionListener(listener);
		analysisMenu.add(varianceItem);

		this.add(analysisMenu);


	}

	private void createChartMenu(ActionListener listener) {
		//Chart Menu
		JMenu chartMenu = new JMenu("Charts");
		chartMenu.setMnemonic(KeyEvent.VK_C);

		//Add a menu item.
		JMenuItem groupNameItem = new JMenuItem(
			PerfExplorerActionListener.SET_GROUPNAME, KeyEvent.VK_G);
		groupNameItem.addActionListener(listener);
		chartMenu.add(groupNameItem);

		//Add a menu item.
		JMenuItem metricNameItem = new JMenuItem(
			PerfExplorerActionListener.SET_METRICNAME, KeyEvent.VK_M);
		metricNameItem.addActionListener(listener);
		chartMenu.add(metricNameItem);

		//Add a menu item.
		JMenuItem eventNameItem = new JMenuItem(
			PerfExplorerActionListener.SET_EVENTNAME, KeyEvent.VK_E);
		eventNameItem.addActionListener(listener);
		chartMenu.add(eventNameItem);

		//Add a menu item.
		JMenuItem timestepItem = new JMenuItem(
			PerfExplorerActionListener.SET_TIMESTEPS, KeyEvent.VK_P);
		timestepItem.addActionListener(listener);
		chartMenu.add(timestepItem);

		//Add a menu item.
		JMenuItem timesteps = new JMenuItem(
			PerfExplorerActionListener.TIMESTEPS_CHART, KeyEvent.VK_T);
		timesteps.addActionListener(listener);
		chartMenu.add(timesteps);

		//Add a menu item.
		JMenuItem efficiency = new JMenuItem(
			PerfExplorerActionListener.EFFICIENCY_CHART, KeyEvent.VK_R);
		efficiency.addActionListener(listener);
		chartMenu.add(efficiency);

		//Add a menu item.
		JMenuItem efficiencyEvents = new JMenuItem(
			PerfExplorerActionListener.EFFICIENCY_EVENTS_CHART, KeyEvent.VK_F);
		efficiencyEvents.addActionListener(listener);
		chartMenu.add(efficiencyEvents);

		//Add a menu item.
		JMenuItem efficiencyOneEvent = new JMenuItem(
			PerfExplorerActionListener.EFFICIENCY_ONE_EVENT_CHART, KeyEvent.VK_F);
		efficiencyOneEvent.addActionListener(listener);
		chartMenu.add(efficiencyOneEvent);

		//Add a menu item.
		JMenuItem speedup = new JMenuItem(
			PerfExplorerActionListener.SPEEDUP_CHART, KeyEvent.VK_S);
		speedup.addActionListener(listener);
		chartMenu.add(speedup);

		//Add a menu item.
		JMenuItem speedupEvents = new JMenuItem(
			PerfExplorerActionListener.SPEEDUP_EVENTS_CHART, KeyEvent.VK_S);
		speedupEvents.addActionListener(listener);
		chartMenu.add(speedupEvents);

		//Add a menu item.
		JMenuItem speedupOneEvent = new JMenuItem(
			PerfExplorerActionListener.SPEEDUP_ONE_EVENT_CHART, KeyEvent.VK_S);
		speedupOneEvent.addActionListener(listener);
		chartMenu.add(speedupOneEvent);

		//Add a menu item.
		JMenuItem transpose = new JMenuItem(
			PerfExplorerActionListener.COMMUNICATION_CHART, KeyEvent.VK_C);
		transpose.addActionListener(listener);
		chartMenu.add(transpose);

		//Add a menu item.
		JMenuItem fraction = new JMenuItem(
			PerfExplorerActionListener.FRACTION_CHART, KeyEvent.VK_B);
		fraction.addActionListener(listener);
		chartMenu.add(fraction);

		//Add a menu item.
		JMenuItem efficiencyPhase = new JMenuItem(
			PerfExplorerActionListener.EFFICIENCY_PHASE_CHART, KeyEvent.VK_B);
		efficiencyPhase.addActionListener(listener);
		chartMenu.add(efficiencyPhase);

		//Add a menu item.
		JMenuItem speedupPhase = new JMenuItem(
			PerfExplorerActionListener.SPEEDUP_PHASE_CHART, KeyEvent.VK_B);
		speedupPhase.addActionListener(listener);
		chartMenu.add(speedupPhase);

		//Add a menu item.
		JMenuItem fractionPhases = new JMenuItem(
			PerfExplorerActionListener.FRACTION_PHASE_CHART, KeyEvent.VK_B);
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

		this.add(viewMenu);
	}

}
