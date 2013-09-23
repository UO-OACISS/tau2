package edu.uoregon.tau.perfexplorer.glue;

import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.io.Serializable;
import java.net.URL;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import javax.swing.BoxLayout;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTabbedPane;
import javax.swing.JTable;
import javax.swing.KeyStroke;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableModel;

import org.jfree.chart.ChartPanel;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerActionListener;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerChart;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerChartWindow;

public class DrawMultiGraph extends AbstractPerformanceOperation {
	
	public static class MapTableModel extends AbstractTableModel {

		public MapTableModel(Map<String, String> tableData) {
			super();
			this.tableData = tableData.entrySet().toArray();
		}

		/**
		 * 
		 */
		private static final long serialVersionUID = 2404657420216796650L;

		Object[] tableData = null;

		public int getRowCount() {
			if (tableData != null) {
				return tableData.length;
			}
			return 0;
		}

		public int getColumnCount() {
			if (tableData != null) {
				return 2;
			}
			return 0;
		}

		private static final String colA = "Field";
		private static final String colB = "Value";

		public String getColumnName(int column) {
			if (column == 0) {
				return colA;
			} else if (column == 1) {
				return colB;
			}
			return "-";
		}

		public Object getValueAt(int rowIndex, int columnIndex) {
			if (tableData == null || tableData.length < rowIndex) {
				return null;
			}
			@SuppressWarnings("unchecked")
			Map.Entry<String, String> entry = ((Map.Entry<String, String>) tableData[rowIndex]);
			if (columnIndex == 0) {
				return entry.getKey();
			} else if (columnIndex == 1) {
				return entry.getValue();
			}
			return null;
		}

	}

	public static class MultiGraphWindow extends JFrame implements
			ActionListener {

		/**
		 * 
		 */
		private static final long serialVersionUID = 5842078287780393184L;

		// public void export(Graphics2D g2d, boolean toScreen,
		// boolean fullWindow, boolean drawHeader) {
		//
		// }

		// public Dimension getImageSize(boolean fullScreen, boolean header) {
		// return null;
		// }

		public void actionPerformed(ActionEvent event) {
			try {
				Object EventSrc = event.getSource();
				if (EventSrc instanceof JMenuItem) {
					String arg = event.getActionCommand();
					if (arg.equals(PerfExplorerChartWindow.ABOUT)) {
						createAboutWindow();
					} else if (arg.equals(PerfExplorerChartWindow.SEARCH)) {
						createHelpWindow();
					}
					// else if (arg.equals(PerfExplorerChartWindow.SAVE)) {
					// saveThyself();}
					else if (arg.equals(PerfExplorerChartWindow.CLOSE)) {
						dispose();
					} else {
						System.out.println("unknown event! " + arg);
					}
				}
			} catch (Exception e) {
				System.err.println("actionPerformed Exception: "
						+ e.getMessage());
				e.printStackTrace();
			}
			
		}

		public void createAboutWindow() {
			long memUsage = (Runtime.getRuntime().totalMemory() - Runtime
					.getRuntime().freeMemory()) / 1024;

			String message = new String("PerfExplorer 1.0\n"
					+ PerfExplorerActionListener.getVersionString()
					+ "\nJVM Heap Size: " + memUsage + "kb\n");
			JOptionPane.showMessageDialog(this, message, "About PerfExplorer",
					JOptionPane.PLAIN_MESSAGE);
		}

		public void createHelpWindow() {
			JOptionPane
					.showMessageDialog(
							this,
							"Help not implemented.\nFor the most up-to-date documentation, please see\n<html><a href='http://www.cs.uoregon.edu/research/tau/'>http://www.cs.uoregon.edu/research/tau/</a></html>",
							"PerfExplorer Help", JOptionPane.PLAIN_MESSAGE);
		}

		// public void saveThyself() {
		// // System.out.println("Daemon come out!");
		// try {
		// VectorExport.promptForVectorExport(this, "PerfExplorer");
		// } catch (Exception e) {
		// System.out.println("File Export Failed!");
		// e.printStackTrace();
		// }
		// return;
		// }
		
		public MultiGraphWindow (String name,List<GraphTab> tabData) {
			
			super("TAU/PerfExplorer: " + name);


			
			JTabbedPane tabs = new JTabbedPane();
			for(int i=0;i<tabData.size();i++){
				JComponent tab = tabData.get(i).getTab();
				tabs.addTab(tabData.get(i).getName(), tab);
			}
			this.add(tabs);
			
			ActionListener listener = this;
			this.setJMenuBar(new PerfExplorerMultiGraphJMenuBar(listener));
	        URL url = Utility.getResource("tau32x32.gif");
			if (url != null) 
				setIconImage(Toolkit.getDefaultToolkit().getImage(url));
			PerfExplorerChartWindow.centerFrame(this);
			this.pack();
			this.setVisible(true);
		}
		
	}

	public static class GraphTab implements ActionListener, Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = -933317318947126719L;

		public GraphTab(String name, List<GraphTabRegion> regions) {
			super();
			this.name = name;
			this.regions = regions;
		}

		/**
		 * @return the name
		 */
		public String getName() {
			return name;
		}

		/**
		 * @param name
		 *            the name to set
		 */
		public void setName(String name) {
			this.name = name;
		}

		/**
		 * @return the regions
		 */
		public List<GraphTabRegion> getRegions() {
			return regions;
		}

		/**
		 * @param regions
		 *            the regions to set
		 */
		public void setRegions(List<GraphTabRegion> regions) {
			this.regions = regions;
		}

		public List<String> getRegionNames() {
			List<String> names = new ArrayList<String>();
			if (regions != null) {
				for (int i = 0; i < regions.size(); i++) {
					names.add(regions.get(i).getName());
				}
			}
			return names;
		}

		private String name;
		private List<GraphTabRegion> regions;

		private JSplitPane tab = null;

		private JComboBox regionBox = null;
		private JTable table = null;
		private JPanel graphPanel = null;
		private JScrollPane graphScroll = null;

		public JComponent getTab() {
			if (tab != null) {
				return tab;
			}
			tab = createTab(this);

			return tab;
		}

		private JSplitPane createTab(GraphTab tabData){
			JComponent table = createTablePane(
tabData.getRegions().get(0)
					.getMapModel(),
					tabData.getRegionNames());
			JComponent graph = createGraphPane(tabData.getRegions().get(0).charts);
			JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT,
                    table, graph);
			return splitPane;
		}
		
		private JComponent createTablePane(TableModel tableModel,
				List<String> regionNames) {
			table = new JTable(tableModel);
			JScrollPane scrollpane = new JScrollPane(table);
			JPanel tablePanel = new JPanel();
			tablePanel
					.setLayout(new BoxLayout(tablePanel, BoxLayout.PAGE_AXIS));

			if (regionNames != null && regionNames.size() > 0) {
				regionBox = new JComboBox(regionNames.toArray());
				regionBox.addActionListener(this);
				Dimension d = new Dimension();
				d.setSize(100000, regionBox.getPreferredSize().height);
				regionBox.setMaximumSize(d);
				tablePanel.add(regionBox);
			}
			tablePanel.add(scrollpane);
			return tablePanel;
		}
		
		private JComponent createGraphPane(List<ChartPanel> charts) {
			graphPanel = new JPanel();
			if (charts != null) {

				graphPanel.setLayout(new BoxLayout(graphPanel,
						BoxLayout.PAGE_AXIS));

				for (int i = 0; i < charts.size(); i++) {

					graphPanel.add(charts.get(i));
				}

			}
			graphScroll = new JScrollPane(graphPanel);
			return graphScroll;
		}

		public void actionPerformed(ActionEvent e) {
			if (e.getSource().equals(this.regionBox)) {
				// String region = (String) regionBox.getSelectedItem();
				// System.out.println(region);

				GraphTabRegion selected = regions.get(regionBox
						.getSelectedIndex());
				table.setModel(selected.getMapModel());

				graphPanel.removeAll();
				for (int i = 0; i < selected.charts.size(); i++) {
					graphPanel.add(selected.charts.get(i));
				}

				graphPanel.validate();
				graphPanel.repaint();
				graphScroll.revalidate();
			}
		}
		
	}
	
	public static class GraphTabRegion implements Serializable{
		/**
		 * 
		 */
		private static final long serialVersionUID = -1219370845965690129L;

		/**
		 * @return the name
		 */
		public String getName() {
			return name;
		}
		/**
		 * @param name the name to set
		 */
		public void setName(String name) {
			this.name = name;
		}

		public GraphTabRegion(String name, List<DrawGraph> graphs,
				Map<String, String> tableData) {
			super();
			this.name = name;
			mapModel = new MapTableModel(tableData);
			charts = new ArrayList<ChartPanel>();
			if (graphs != null) {
			for (int i = 0; i < graphs.size(); i++) {
				graphs.get(i).setAutoDisplayWindow(false);
				graphs.get(i).processData();
				ChartPanel cp = new ChartPanel(graphs.get(i).getChart());
				cp.setDisplayToolTips(true);
					cp.setMinimumSize(cp.getPreferredSize());
				charts.add(cp);
			}
			}

		}

		/**
		 * @return the mapModel
		 */
		public TableModel getMapModel() {
			return mapModel;
		}

		private String name;
		private List<ChartPanel> charts;
		private TableModel mapModel;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -3530762753159608317L;
	
	private List<GraphTab> tabData;
	private String name;
	
	public DrawMultiGraph(String name, List<GraphTab> tabData){
		this.tabData=tabData;
		this.name=name;
	}

	public List<PerformanceResult> processData() {
		new MultiGraphWindow(name,tabData);
		return null;
	}
	
	public static void main(String[] args) {

		String rName = "A REGION!!!";
		Map<String, String> mapa = new LinkedHashMap<String, String>();
		mapa.put("Foo", "Bar");
		mapa.put("fish", "stew");
		List<DrawGraph> dg = null;// new DrawGraph((PerformanceResult) null);
		GraphTabRegion rega = new GraphTabRegion(rName, dg, mapa);
		
		
		String rbName = "B REGION!!!";
		Map<String, String> mapb = new LinkedHashMap<String, String>();
		mapb.put("Foob", "Barb");
		mapb.put("fishb", "stewb");
		// DrawGraph dgb = null;// new DrawGraph((PerformanceResult) null);
		GraphTabRegion regb = new GraphTabRegion(rbName, dg, mapb);
		

		List<GraphTabRegion> regions = new ArrayList<GraphTabRegion>();
		regions.add(rega);
		regions.add(regb);

		String name = "A TAB!!!";
		GraphTab taba = new GraphTab(name, regions);

		List<GraphTab> tabs = new ArrayList<GraphTab>();
		tabs.add(taba);
		tabs.add(new GraphTab("B TAB!!!",regions));

		new MultiGraphWindow("TESTING!!!", tabs);
	}

	public static class PerfExplorerMultiGraphJMenuBar extends JMenuBar {

		/**
		 * 
		 */
		private static final long serialVersionUID = -6359699047764668001L;

		public PerfExplorerMultiGraphJMenuBar(ActionListener listener) {
			super();
			createFileMenu(listener);
			createHelpMenu(listener);
		}

		private void createFileMenu(ActionListener listener) {
			// File menu.
			JMenu fileMenu = new JMenu("File");
			fileMenu.setMnemonic(KeyEvent.VK_F);

			// Add a menu item.
			// JMenuItem saveItem = new JMenuItem(
			// PerfExplorerChart.SAVE);
			// saveItem.addActionListener(listener);
			// fileMenu.add(saveItem);

			// Add a menu item.
			JMenuItem closeItem = new JMenuItem(PerfExplorerChart.CLOSE);
			closeItem.addActionListener(listener);
			fileMenu.add(closeItem);

			this.add(fileMenu);
		}

		private void createHelpMenu(ActionListener listener) {
			// Help Menu
			JMenu helpMenu = new JMenu("Help");
			helpMenu.setMnemonic(KeyEvent.VK_H);

			// Add a menu item.
			JMenuItem aboutItem = new JMenuItem(PerfExplorerChart.ABOUT,
					KeyEvent.VK_A);
			aboutItem.addActionListener(listener);
			helpMenu.add(aboutItem);

			// Add a menu item.
			JMenuItem showHelpWindowItem = new JMenuItem(
					PerfExplorerChart.SEARCH, KeyEvent.VK_H);
			showHelpWindowItem.setAccelerator(KeyStroke.getKeyStroke(
					KeyEvent.VK_H, ActionEvent.ALT_MASK));
			showHelpWindowItem.addActionListener(listener);
			helpMenu.add(showHelpWindowItem);

			this.add(helpMenu);
		}

	}

}