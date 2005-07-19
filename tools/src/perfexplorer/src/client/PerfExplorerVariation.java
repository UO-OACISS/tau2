/*
 * Created on May 26, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package client;


import java.awt.Point;
import java.awt.Dimension;
import javax.swing.JFrame;
import javax.swing.JTable;
import javax.swing.JScrollPane;
import common.RMIVarianceData;

/**
 * @author khuck
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class PerfExplorerVariation {
	
	public static void doVariationAnalysis() {
		// for each event, get the variation across all threads.
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIVarianceData data = server.requestVariationAnalysis(
			PerfExplorerModel.getModel());
	
		displayResults(data);
	}

	private static void displayResults(RMIVarianceData data) {
        // Create and set up the window.
        JFrame frame = new JFrame("Summarization Analysis");
        //frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // arrange the data
        Object[][] mydata = data.getDataMatrix();
        // get the columns
        Object[] columns = data.getColumnNames();
        
        
       
        TableSorter sorter = new TableSorter(new MyTableModel(columns, mydata)); //ADDED THIS
        JTable table = new JTable(sorter);             //NEW
        sorter.setTableHeader(table.getTableHeader()); //ADDED THIS
        
        // Make the table vertically scrollable
        JScrollPane scrollPane = new JScrollPane(table);
        
        //Window Stuff.
        int windowWidth = 800;
        int windowHeight = 400;
        
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
        scrollPane.setPreferredSize(new java.awt.Dimension(windowWidth, windowHeight));

        frame.getContentPane().add(scrollPane);
        frame.pack();
        frame.setVisible(true);
		
	}
	
}
