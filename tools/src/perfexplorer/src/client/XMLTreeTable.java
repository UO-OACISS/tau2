package edu.uoregon.tau.perfexplorer.client;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;

import edu.uoregon.tau.common.treetable.*;

public class XMLTreeTable {

    public static void main(String[] args) {

    	try {
	    	String inString = "<?xml version='1.0'?><note from='Santhosh' to='JRoller'><heading>Reminder</heading><body>Don't forget me this weekend</body></note>";
	        JFrame frame = new JFrame("TreeTable");
	    	SAXTreeViewer viewer = new SAXTreeViewer();
	        JTreeTable treeTable = viewer.getTreeTable(inString);
	
	        frame.addWindowListener(new WindowAdapter() {
	            public void windowClosing(WindowEvent we) {
	                System.exit(0);
	            }
	        });
	
	        frame.getContentPane().add(new JScrollPane(treeTable));
	        frame.pack();
	        frame.show();
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	}

}
