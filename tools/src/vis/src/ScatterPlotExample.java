/*
 * ScatterPlotExample.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

/**
 * An example of the vis package's ScatterPlot. <p>
 * 
 * <pre>
 
public class ScatterPlotExample {

    private static void createAndShowGUI() {

        // Create and set up the window.
        JFrame frame = new JFrame("ScatterPlotExample");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Create some values
        float values[][] = new float[100][4];
        for (int i = 0; i < 100; i++) {
            values[i][0] = i * (float) Math.sin(i); // value for the x axis
            values[i][1] = i * (float) Math.cos(i); // value for the y axis
            values[i][2] = i * i; // value for the z axis
            values[i][3] = i; // value for the color axis
        }


        // Create the visRenderer and register it with the canvas
        VisRenderer visRenderer = new VisRenderer();

        // Create the canvas
        VisCanvas visCanvas = new VisCanvas(visRenderer);
        visCanvas.getActualCanvas().setSize(600,600);

        
        ColorScale colorScale = new ColorScale();

        // Create the scatterPlot
        ScatterPlot scatterPlot = PlotFactory.createScatterPlot("x axis", "y axis", "z axis", 
                "color axis", values, true, colorScale);

        // Set the size
        scatterPlot.setSize(10, 10, 10);

        // point at the center of the scatterPlot
        visRenderer.setAim(new Vec(5, 5, 5));

        // Add the drawable objects to the visRenderer (the scatterPlot will draw the axes)
        visRenderer.addShape(scatterPlot);
        visRenderer.addShape(colorScale);

        // Create the control panel, if desired
        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
        tabbedPane.addTab("ScatterPlot", scatterPlot.getControlPanel(visRenderer));
        tabbedPane.addTab("Axes", scatterPlot.getAxes().getControlPanel(visRenderer));
        tabbedPane.addTab("ColorScale", colorScale.getControlPanel(visRenderer));
        tabbedPane.addTab("Render", visRenderer.getControlPanel());
        tabbedPane.setMinimumSize(new Dimension(300, 160));

        // Add everything to a JPanel and add the panel to the frame
        JPanel panel = new JPanel();
        panel.setLayout(new GridBagLayout());
        panel.add(visCanvas.getActualCanvas(), new GridBagConstraints(0, 0, 1, 1, 0.9, 1.0, GridBagConstraints.WEST, 
                GridBagConstraints.BOTH, new Insets(5, 5, 5, 5), 1, 1));
        panel.add(tabbedPane, new GridBagConstraints(1, 0, 1, 1, 0.1, 1.0, GridBagConstraints.EAST,
                GridBagConstraints.HORIZONTAL, new Insets(5, 5, 5, 5), 1, 1));

        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        // Schedule a job for the event-dispatching thread:
        // creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
    }
}
  
 * </pre>
 *    
 * <P>CVS $Id: ScatterPlotExample.java,v 1.6 2007/12/07 02:05:22 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.6 $
 */
public class ScatterPlotExample {

    private static void createAndShowGUI() {

        // Create and set up the window.
        JFrame frame = new JFrame("ScatterPlotExample");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Create some values
        float values[][] = new float[100][4];
        for (int i = 0; i < 100; i++) {
            values[i][0] = i * (float) Math.sin(i); // value for the x axis
            values[i][1] = i * (float) Math.cos(i); // value for the y axis
            values[i][2] = i * i; // value for the z axis
            values[i][3] = i; // value for the color axis
        }

        // Create the visRenderer and register it with the canvas
        VisRenderer visRenderer = new VisRenderer();

        // Create the canvas
        VisCanvas visCanvas = new VisCanvas(visRenderer);
        visCanvas.getActualCanvas().setSize(600,600);

        
        ColorScale colorScale = new ColorScale();

        // Create the scatterPlot
        ScatterPlot scatterPlot = PlotFactory.createScatterPlot("x axis", "y axis", "z axis", 
                "color axis", values, true, colorScale);

        // Set the size
        scatterPlot.setSize(10, 10, 10);

        // point at the center of the scatterPlot
        visRenderer.setAim(new Vec(5, 5, 5));

        // Add the drawable objects to the visRenderer (the scatterPlot will draw the axes)
        visRenderer.addShape(scatterPlot);
        visRenderer.addShape(colorScale);

        // Create the control panel, if desired
        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
        tabbedPane.addTab("ScatterPlot", scatterPlot.getControlPanel(visRenderer));
        tabbedPane.addTab("Axes", scatterPlot.getAxes().getControlPanel(visRenderer));
        tabbedPane.addTab("ColorScale", colorScale.getControlPanel(visRenderer));
        tabbedPane.addTab("Render", visRenderer.getControlPanel());
        tabbedPane.setMinimumSize(new Dimension(300, 260));

        // Add everything to a JPanel and add the panel to the frame
        JPanel panel = new JPanel();
        panel.setLayout(new GridBagLayout());
        panel.add(visCanvas.getActualCanvas(), new GridBagConstraints(0, 0, 1, 1, 0.9, 1.0, GridBagConstraints.WEST, 
                GridBagConstraints.BOTH, new Insets(5, 5, 5, 5), 1, 1));
        panel.add(tabbedPane, new GridBagConstraints(1, 0, 1, 1, 0.1, 1.0, GridBagConstraints.EAST,
                GridBagConstraints.HORIZONTAL, new Insets(5, 5, 5, 5), 1, 1));

        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        // Schedule a job for the event-dispatching thread:
        // creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
    }
}
