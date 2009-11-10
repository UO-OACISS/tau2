package edu.uoregon.tau.perfexplorer.client;

import java.awt.*;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import java.net.URL;
import edu.uoregon.tau.common.Utility;


import edu.uoregon.tau.perfexplorer.common.RMICubeData;
import edu.uoregon.tau.vis.*;

public class PerfExplorerCube {

	public static void doCorrelationCube () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMICubeData data = server.requestCubeData(
			PerfExplorerModel.getModel()); 
		createAndShowGUI(data);
	}

    private static void createAndShowGUI(RMICubeData data) {

        if(data==null)
        {
        	return;
        }
    	
        // Create and set up the window.
        JFrame frame = new JFrame("TAU/PerfExplorer: Correlation of top 4 variant events");

		// get the data values
        float values[][] = data.getValues();

        // Create the visRenderer and register it with the canvas
        VisRenderer visRenderer = new VisRenderer();

        // Create the JOGL canvas
        VisCanvas canvas = new VisCanvas(visRenderer);
        canvas.getActualCanvas().setSize(600, 600);

        ColorScale colorScale = new ColorScale();

        // Create the scatterPlot
        String[] tmp = new String[4];
        for (int i = 0 ; i < 4 ; i++) {
        	int index = data.getNames(i).indexOf("[");
        	if (index > 0) {
        		tmp[i] = data.getNames(i).substring(0, index-1);
        	} else {
        		tmp[i] = data.getNames(i);
        	}
        }
        
        ScatterPlot scatterPlot = PlotFactory.createScatterPlot(tmp[0], tmp[1], tmp[2], tmp[3], values, true, colorScale);
        
        // Set the size
        scatterPlot.setSize(10, 10, 10);

        // point at the center of the scatterPlot
        visRenderer.setAim(new Vec(5,5,5));
        
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
        tabbedPane.setMinimumSize(new Dimension(300, 300));

        // Add everything to a JPanel and add the panel to the frame
        JPanel panel = new JPanel();
        panel.setLayout(new GridBagLayout());
        panel.add(canvas.getActualCanvas(), new GridBagConstraints(0, 0, 1, 1, 0.9, 1.0, GridBagConstraints.WEST,
                GridBagConstraints.BOTH, new Insets(5, 5, 5, 5), 1, 1));
        panel.add(tabbedPane, new GridBagConstraints(1, 0, 1, 1, 0.1, 1.0, GridBagConstraints.EAST,
                GridBagConstraints.HORIZONTAL, new Insets(5, 5, 5, 5), 1, 1));

        //Window Stuff.
        int windowWidth = 760;
        int windowHeight = 600;
        
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

        URL url = Utility.getResource("tau32x32.gif");
		if (url != null) 
			frame.setIconImage(Toolkit.getDefaultToolkit().getImage(url));

        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }

}
