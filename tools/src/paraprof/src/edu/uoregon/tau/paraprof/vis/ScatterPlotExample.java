package edu.uoregon.tau.paraprof.vis;

import java.awt.*;
import java.util.Vector;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;

import net.java.games.jogl.GLCanvas;
import net.java.games.jogl.GLCapabilities;
import net.java.games.jogl.GLDrawableFactory;

/**
 * An example of the vis package's ScatterPlot
 */
public class ScatterPlotExample {

    private static void createAndShowGUI() {

        // Create and set up the window.
        JFrame frame = new JFrame("ScatterPlotExample");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Create some values
        float values[][] = new float[25][4];
        for (int i = 0; i < 25; i++) {
            values[i][0] = 1000+i * i;  // value for the x axis
            values[i][1] = i;      // value for the y axis
            values[i][2] = i * i;  // value for the z axis
            values[i][3] = i;      // value for the color axis
        }

        // Create the JOGL canvas
        GLCanvas canvas = GLDrawableFactory.getFactory().createGLCanvas(new GLCapabilities());
        canvas.setSize(600, 600);

        // Create the visRenderer and register it with the canvas
        VisRenderer visRenderer = new VisRenderer();
        canvas.addGLEventListener(visRenderer);

        ColorScale colorScale = new ColorScale(visRenderer);


        ScatterPlot scatterPlot = PlotFactory.createScatterPlot("x axis", "y axis", "z axis", "color axis", values, true, colorScale, visRenderer);
        
        
        scatterPlot.setSize(10, 10, 10);
        // point at the center of the scatterPlot
        visRenderer.setAim(new Vec(5,5,5));
        
        // Add the drawable objects to the visRenderer (the scatterPlot will draw the axes)
        visRenderer.addShape(scatterPlot);
        visRenderer.addShape(colorScale);

        
        // Create the control panel, if desired
        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
        tabbedPane.addTab("ScatterPlot", scatterPlot.getControlPanel());
        tabbedPane.addTab("Axes", scatterPlot.getAxes().getControlPanel());
        tabbedPane.addTab("ColorScale", colorScale.getControlPanel());
        tabbedPane.addTab("Render", visRenderer.getControlPanel());
        tabbedPane.setMinimumSize(new Dimension(300, 160));

        // Add everything to a JPanel and add the panel to the frame
        JPanel panel = new JPanel();
        panel.setLayout(new GridBagLayout());
        panel.add(canvas, new GridBagConstraints(0, 0, 1, 1, 0.9, 1.0, GridBagConstraints.WEST,
                GridBagConstraints.BOTH, new Insets(5, 5, 5, 5), 1, 1));
        panel.add(tabbedPane, new GridBagConstraints(1, 0, 1, 1, 0.1, 1.0, GridBagConstraints.EAST,
                GridBagConstraints.HORIZONTAL, new Insets(5, 5, 5, 5), 1, 1));

        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        //Schedule a job for the event-dispatching thread:
        //creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
    }
}
