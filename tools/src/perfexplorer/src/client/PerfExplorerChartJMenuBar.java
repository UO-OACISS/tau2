package edu.uoregon.tau.perfexplorer.client;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;

import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.KeyStroke;

public class PerfExplorerChartJMenuBar extends JMenuBar {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6359699047764668001L;

	public PerfExplorerChartJMenuBar(ActionListener listener) {
		super();
		createFileMenu(listener);
		createHelpMenu(listener);
	}

	private void createFileMenu(ActionListener listener) {
		//File menu.
		JMenu fileMenu = new JMenu("File");
		fileMenu.setMnemonic(KeyEvent.VK_F);

		//Add a menu item.
		JMenuItem saveItem = new JMenuItem(
				PerfExplorerChart.SAVE);
		saveItem.addActionListener(listener);
		fileMenu.add(saveItem);

		//Add a menu item.
		JMenuItem closeItem = new JMenuItem(
				PerfExplorerChart.CLOSE);
		closeItem.addActionListener(listener);
		fileMenu.add(closeItem);

		this.add(fileMenu);
	}

	private void createHelpMenu(ActionListener listener) {
		//Help Menu
		JMenu helpMenu = new JMenu("Help");
		helpMenu.setMnemonic(KeyEvent.VK_H);

		//Add a menu item.
		JMenuItem aboutItem = new JMenuItem(
			PerfExplorerChart.ABOUT, KeyEvent.VK_A);
		aboutItem.addActionListener(listener);
		helpMenu.add(aboutItem);

		//Add a menu item.
		JMenuItem showHelpWindowItem = new JMenuItem(
			PerfExplorerChart.SEARCH, KeyEvent.VK_H);
		showHelpWindowItem.setAccelerator(KeyStroke.getKeyStroke(
			KeyEvent.VK_H, ActionEvent.ALT_MASK));
		showHelpWindowItem.addActionListener(listener);
		helpMenu.add(showHelpWindowItem);

		this.add(helpMenu);
	}

}
