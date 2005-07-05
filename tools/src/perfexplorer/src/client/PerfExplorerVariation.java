/*
 * Created on May 26, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package client;


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
		
		/*
		System.out.print("name\t");
	
		for (int j = 0 ; j < data.getValueCount() ; j++) {
			System.out.print(data.getValueName(j) + "\t");
		}
		System.out.println();
		
		for (int i = 0 ; i < data.getEventCount() ; i++) {
			double[] values = data.getValues(i);
			System.out.print(data.getEventName(i) + "\t");
			for (int j = 0 ; j < values.length ; j++) {
				System.out.print(values[j] + "\t");
			}
			System.out.println();
		}
		*/
		
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
        
        frame.getContentPane().add(scrollPane);
        frame.pack();
        frame.setVisible(true);
		
	}
	
}
