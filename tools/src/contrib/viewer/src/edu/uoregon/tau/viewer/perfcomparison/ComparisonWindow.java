package edu.uoregon.tau.viewer.perfcomparison;

import java.util.*;
import java.awt.*;
import javax.swing.*;
import java.awt.event.*;
import javax.swing.border.*;

import edu.uoregon.tau.dms.dss.*;

/**
 * @author lili
 * This class creates various components of performance comparison window.
 */

public class ComparisonWindow extends JFrame implements ActionListener{
    private PerfComparison comparisonEnv;
    private JMenuItem showMeanItem, showTotalItem, showGroupMeanItem, showGroupTotalItem;
    private JRadioButtonMenuItem nameButton, valueButton, ascendButton, descendButton;

    private ComparisonWindowPanel cwPanel;

    public ComparisonWindow(PerfComparison env){
	this.comparisonEnv = env;

	// customize window.
	setTitle("Comparison of the selected trials.");
	int windowWidth = 750;
 	int windowHeight = 600;
	setSize(new java.awt.Dimension(windowWidth, windowHeight));

	//Add window listener 
	addWindowListener(new WindowAdapter() {
		public void windowClosing(WindowEvent evt) {
		    setVisible(false);
		    dispose();
		}
	    });
      
      //get the screen size.
      Toolkit tk = Toolkit.getDefaultToolkit();
      Dimension screenSize = tk.getScreenSize();
      int screenHeight = screenSize.height;
      int screenWidth = screenSize.width;
      
      //Set the window location.
      int xPosition = (screenWidth - windowWidth) / 2;
      int yPosition = (screenHeight - windowHeight) / 2;      
      setLocation(xPosition, yPosition);
	
      // set menu bar.
      setJMenuBar(createMenuBar());
        
      // set main panel.  
      cwPanel = new ComparisonWindowPanel(comparisonEnv);
      
      JScrollPane scrollPane = new JScrollPane(cwPanel);
      Border mainloweredbev = BorderFactory.createLoweredBevelBorder();
      scrollPane.setBorder(mainloweredbev);
      scrollPane.setPreferredSize(new Dimension(600, 500));

      getContentPane().add(scrollPane);

    }

	// create menus.
    private JMenuBar createMenuBar(){
	   JMenuBar menuBar = new JMenuBar();
	   
	    // option menu.
	   JMenu opMenu = new JMenu("Options");
	   menuBar.add(opMenu);
	   	   
	   showMeanItem = new JMenuItem("show mean statistics");

	   if (comparisonEnv.isShowMeanValueEnabled())
	       showMeanItem.setEnabled(false);
	   else 
	       showMeanItem.setEnabled(true);

	   showMeanItem.addActionListener(this);
	   opMenu.add(showMeanItem);
	   opMenu.addSeparator();

	   showTotalItem = new JMenuItem("show total statistics");
	   
	   if (comparisonEnv.isShowTotalValueEnabled())
	       showTotalItem.setEnabled(false);
	   else 
	       showTotalItem.setEnabled(true);

	   showTotalItem.addActionListener(this);
	   opMenu.add(showTotalItem);
	   opMenu.addSeparator();
	    
	   showGroupMeanItem = new JMenuItem("show mean statistics of a group");
	   showGroupMeanItem.addActionListener(this);
	   opMenu.add(showGroupMeanItem); 
	   opMenu.addSeparator();

	   showGroupTotalItem = new JMenuItem("show total statistics of a group");
	   showGroupTotalItem.addActionListener(this);
	   opMenu.add(showGroupTotalItem);
	   opMenu.addSeparator();

	   if (comparisonEnv.isIntervalEventComparison()){
	       showGroupMeanItem.setEnabled(false);
	       showGroupTotalItem.setEnabled(false);
	   }
	   else {
	       showGroupMeanItem.setEnabled(true);
	       showGroupTotalItem.setEnabled(true);
	   }
	          	
	   JMenu sortMenu = new JMenu("sort by");
	   opMenu.add(sortMenu);
	   opMenu.addSeparator();

	   ButtonGroup group = new ButtonGroup();
	   nameButton = new JRadioButtonMenuItem("function name", false);
	   nameButton.addActionListener(this);
	   sortMenu.add(nameButton);
	   group.add(nameButton);
	   sortMenu.addSeparator();

	   valueButton = new JRadioButtonMenuItem("value", true);
	   valueButton.addActionListener(this);
	   sortMenu.add(valueButton);
	   group.add(valueButton);

	   JMenu orderMenu = new JMenu("sort order");
	   opMenu.add(orderMenu);
	   
	   ButtonGroup group2 = new ButtonGroup();

	   ascendButton = new JRadioButtonMenuItem("ascending", false);
	   ascendButton.addActionListener(this);
	   orderMenu.add(ascendButton);
	   group2.add(ascendButton);
	   orderMenu.addSeparator();

	   descendButton = new JRadioButtonMenuItem("descending", true);
	   descendButton.addActionListener(this);
	   orderMenu.add(descendButton);
	   group2.add(descendButton);

	   // help menu.
	   JMenu helpMenu = new JMenu("Help");
	   menuBar.add(helpMenu);

	   return menuBar;
    }

    public void actionPerformed(ActionEvent evt) {
	Object eventObj = evt.getSource();
      
	if (eventObj instanceof JMenuItem){ // if this event is fired by a menu item.

	    String commond = evt.getActionCommand();
	    
	    if (commond.equals("show total statistics")){	// if show total
		comparisonEnv.setShowTotalValue();
		comparisonEnv.resort();

		showTotalItem.setEnabled(false);
		showMeanItem.setEnabled(true);

		if (comparisonEnv.isTrialComparison()){
		    showGroupMeanItem.setEnabled(true);
		    showGroupTotalItem.setEnabled(true);
		}
		else {
		    showGroupMeanItem.setEnabled(false);
		    showGroupTotalItem.setEnabled(false);
		}

		// repaint
		cwPanel.revalidate();
		cwPanel.repaint();
	    }
	    else if (commond.equals("show mean statistics")){ // if show mean
		comparisonEnv.setShowMeanValue();
		comparisonEnv.resort();

		showTotalItem.setEnabled(true);
		showMeanItem.setEnabled(false);

		if (comparisonEnv.isTrialComparison()){
		    showGroupMeanItem.setEnabled(true);
		    showGroupTotalItem.setEnabled(true);
		}
		else {
		    showGroupMeanItem.setEnabled(false);
		    showGroupTotalItem.setEnabled(false);
		}

		cwPanel.revalidate();
		cwPanel.repaint();
	    }
	    else if (commond.equals("show mean statistics of a group")){ // if show group mean
		String selectedGroup = null;
		Hashtable groupHT = comparisonEnv.getMeanGroupValues(selectedGroup);


		// let the user select a group 
		String[] groupNames = new String[groupHT.size()];
		int counter = 0;
		for(Enumeration e1 = groupHT.keys(); e1.hasMoreElements();){
		    groupNames[counter++] = (String) e1.nextElement();
		}
				
		String s = (String)JOptionPane.showInputDialog(this,
							       "Please select a group",
							       "Select group",
							       JOptionPane.PLAIN_MESSAGE,
							       null,
							       groupNames,
							       groupNames[0]);

		if (s==null)
		    return;

		comparisonEnv.setShowGroupMeanValue(s);
		comparisonEnv.resort();

		showTotalItem.setEnabled(true);
		showMeanItem.setEnabled(true);
		showGroupMeanItem.setEnabled(true);
		showGroupTotalItem.setEnabled(true);

		cwPanel.revalidate();
		cwPanel.repaint();
	    }
	    else if (commond.equals("show total statistics of a group")){ // if show group total
		String selectedGroup = null;
		Hashtable groupHT = comparisonEnv.getTotalGroupValues(selectedGroup);

		// let the user select a group
		String[] groupNames = new String[groupHT.size()];
		int counter = 0;
		for(Enumeration e1 = groupHT.keys(); e1.hasMoreElements();){
		    groupNames[counter++] = (String) e1.nextElement();
		}
				
		String s = (String)JOptionPane.showInputDialog(this,
							       "Please select a group",
							       "Select group",
							       JOptionPane.PLAIN_MESSAGE,
							       null,
							       groupNames,
							       groupNames[0]);
		if (s == null)
		    return;

		comparisonEnv.setShowGroupTotalValue(s);
		comparisonEnv.resort();

		showTotalItem.setEnabled(true);
		showMeanItem.setEnabled(true);
		showGroupMeanItem.setEnabled(true);
		showGroupTotalItem.setEnabled(true);

		cwPanel.revalidate();
		cwPanel.repaint();
	    }
	    else if (commond.equals("function name")){ // sort by function name menu
		if(nameButton.isSelected()){
		    comparisonEnv.sortBy("function name");
		    cwPanel.revalidate();
		    cwPanel.repaint();		    
		}

	    }
	    else if (commond.equals("value")){ // sort by value menu
		if(valueButton.isSelected()){
		    comparisonEnv.sortBy("value");
		    cwPanel.revalidate();
		    cwPanel.repaint();
		}
	    }
	    else if (commond.equals("ascending")){
		if (ascendButton.isSelected()){
		    comparisonEnv.setSortOrder("ascending");
		    comparisonEnv.resort();

		    cwPanel.revalidate();
		    cwPanel.repaint();	
		}
	    }
	    else if (commond.equals("descending")){
		if (descendButton.isSelected()){
		    comparisonEnv.setSortOrder("descending");
		    comparisonEnv.resort();

		    cwPanel.revalidate();
		    cwPanel.repaint();	
		}
	    }	           
	}	
    }   	
}
