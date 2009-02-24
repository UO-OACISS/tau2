package edu.uoregon.tau.perfexplorer.client;

import java.awt.*;
import java.awt.event.*;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import java.util.List;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.DefaultTableXYDataset;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.data.xy.XYSeries;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.StandardLegend;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import java.text.DecimalFormat;
import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.VectorExport;
import java.net.URL;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.common.RMIChartData;

import java.awt.Toolkit;
import javax.swing.JPanel;

public class PerfExplorerChartWindow extends JFrame implements ActionListener, ImageExport {

	public final static String ABOUT = "About PerfExplorer";
	public final static String SEARCH = "Search For Help On...";
	public final static String SAVE = "Save As Vector Image";
	public final static String CLOSE = "Close Window";

	private ChartPanel panel = null;

	public PerfExplorerChartWindow (JFreeChart chart, String name) {
		super("TAU/PerfExplorer: " + name);
		this.panel = new ChartPanel(chart);
		this.panel.setDisplayToolTips(true);
        this.getContentPane().add(this.panel);
		ActionListener listener = this;
		this.setJMenuBar(new PerfExplorerChartJMenuBar(listener));
        URL url = Utility.getResource("tau32x32.gif");
		if (url != null) 
			setIconImage(Toolkit.getDefaultToolkit().getImage(url));
		centerFrame(this);
		this.pack();
		this.setVisible(true);
	}

	public static void centerFrame(JFrame frame) {
        //Window Stuff.
        int windowWidth = 700;
        int windowHeight = 450;
        
        //Grab paraProfManager position and size.
        Point parentPosition = PerfExplorerClient.getMainFrame().getLocationOnScreen();
        Dimension parentSize = PerfExplorerClient.getMainFrame().getSize();
        int parentWidth = parentSize.width;
        int parentHeight = parentSize.height;
        
        //Set the window to come up in the center of the screen.
        int xPosition = (parentWidth - windowWidth) / 2;
        int yPosition = (parentHeight - windowHeight) / 2;

        xPosition = (int) parentPosition.getX() + xPosition;
        yPosition = (int) parentPosition.getY() + yPosition;

        frame.setLocation(xPosition, yPosition);
        frame.setSize(new java.awt.Dimension(windowWidth, windowHeight));
 	}

	public void saveThyself() {
		//System.out.println("Daemon come out!");
		try {
			VectorExport.promptForVectorExport (this, "PerfExplorer");
		} catch (Exception e) {
			System.out.println("File Export Failed!");
		}
		return;
	}

	public void actionPerformed (ActionEvent event) {
		try {
			Object EventSrc = event.getSource();
			if(EventSrc instanceof JMenuItem) {
				String arg = event.getActionCommand();
				if (arg.equals(ABOUT)) {
					createAboutWindow();
				} else if (arg.equals(SEARCH)) {
					createHelpWindow();
				} else if (arg.equals(SAVE)) {
					saveThyself();
				} else if (arg.equals(CLOSE)) {
					dispose();
				} else {
					System.out.println("unknown event! " + arg);
				}
			}
		} catch (Exception e) {
			System.err.println("actionPerformed Exception: " + e.getMessage());
			e.printStackTrace();
		} 
	}

	public void createAboutWindow() {
		long memUsage = (Runtime.getRuntime().totalMemory() -
			Runtime.getRuntime().freeMemory()) / 1024;

		String message = new String("PerfExplorer 1.0\n" +
					PerfExplorerActionListener.getVersionString() + "\nJVM Heap Size: " + memUsage
					+ "kb\n");
		JOptionPane.showMessageDialog(this, message, 
			"About PerfExplorer", JOptionPane.PLAIN_MESSAGE);
	}

	public void createHelpWindow() {
		JOptionPane.showMessageDialog(this, 
			"Help not implemented.\nFor the most up-to-date documentation, please see\n<html><a href='http://www.cs.uoregon.edu/research/tau/'>http://www.cs.uoregon.edu/research/tau/</a></html>",
			"PerfExplorer Help", JOptionPane.PLAIN_MESSAGE);
	}

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        return panel.getSize();
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        panel.setDoubleBuffered(false);
        panel.paintAll(g2D);
        panel.setDoubleBuffered(true);
    }

}
