/* 
   CallPathTextWindow.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

public class CallPathTextWindow extends JFrame implements ActionListener, MenuListener, Observer{

    public CallPathTextWindow(){
	try{
	    setLocation(new java.awt.Point(0, 0));
	    setSize(new java.awt.Dimension(800, 600));
      
	    //Set the title indicating that there was a problem.
	    this.setTitle("Wrong constructor used!");
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW01");
	}
    }
  
    public CallPathTextWindow(Trial inTrial,
			      int inServerNumber,
			      int inContextNumber,
			      int inThreadNumber,
			      StaticMainWindowData inSMWData,
			      boolean global){
	try{
	    trial = inTrial;
	    sMWData = inSMWData;
	    this.global = global;
      
	    setLocation(new java.awt.Point(0, 0));
	    setSize(new java.awt.Dimension(800, 600));
      
	    //Now set the title.
	    if(global)
		this.setTitle("Call Path Data Relations - " + trial.getProfilePathName());
	    else
		this.setTitle("Call Path Data " + "n,c,t, " + inServerNumber + "," + inContextNumber + "," + inThreadNumber + " - " + trial.getProfilePathName());
      
	    server = inServerNumber;
	    context = inContextNumber;
	    thread = inThreadNumber;
      
	    //Add some window listener code
	    addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(java.awt.event.WindowEvent evt) {
			thisWindowClosing(evt);
		    }
		});
        
	    //Set the help window text if required.
	    if(ParaProf.helpWindow.isVisible()){
		ParaProf.helpWindow.clearText();
		//Since the data must have been loaded.  Tell them someting about
		//where they are.
		ParaProf.helpWindow.writeText("Call path text window.");
		ParaProf.helpWindow.writeText("");
		ParaProf.helpWindow.writeText("This window displays call path relationships in two ways:");
		ParaProf.helpWindow.writeText("1- If this window has been invoked from the \"windows\" menu of");
		ParaProf.helpWindow.writeText("ParaProf, the information displayed is all call path relations found.");
		ParaProf.helpWindow.writeText("That is, all the parent/child relationships.");
		ParaProf.helpWindow.writeText("Thus, in this case, given the parallel nature of ParaProf, this information");
		ParaProf.helpWindow.writeText("might not be valid for a particular thread. It is however useful to observe");
		ParaProf.helpWindow.writeText("all the realtionships that exist in the data.");
		ParaProf.helpWindow.writeText("");
		ParaProf.helpWindow.writeText("2- If this window has been invoked from the popup menu to the left of a thread bar");
		ParaProf.helpWindow.writeText("in the main ParaProf window, the information dispayed will be specific to this thread,");
		ParaProf.helpWindow.writeText("and will thus contain both parent/child relations and the data relating to those");
		ParaProf.helpWindow.writeText("relationships.");

	    }
      
      
	    //******************************
	    //Code to generate the menus.
	    //******************************
      
      
	    JMenuBar mainMenu = new JMenuBar();
      
	    //******************************
	    //File menu.
	    //******************************
	    JMenu fileMenu = new JMenu("File");
      
	    //Add a menu item.
	    JMenuItem closeItem = new JMenuItem("Close This Window");
	    closeItem.addActionListener(this);
	    fileMenu.add(closeItem);
      
	    //Add a menu item.
	    JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
	    exitItem.addActionListener(this);
	    fileMenu.add(exitItem);
	    //******************************
	    //End - File menu.
	    //******************************


	    //******************************
	    //Options menu.
	    //******************************
	    JMenu optionsMenu = new JMenu("Options");
	    optionsMenu.addMenuListener(this);
	    
	    //Add a submenu.
	    sortGroup = new ButtonGroup();
	    ascendingButton.addActionListener(this);
	    descendingButton.addActionListener(this);
	    sortGroup.add(ascendingButton);
	    sortGroup.add(descendingButton);
	    
	    JMenu sortOrderMenu = new JMenu("Sort Order");
	    sortOrderGroup = new ButtonGroup();
	    
	    nameButton.addActionListener(this);
	    inclusiveRadioButton.addActionListener(this);
	    exclusiveRadioButton.addActionListener(this);
	    numOfCallsRadioButton.addActionListener(this);
	    numOfSubRoutinesRadioButton.addActionListener(this);
	    userSecPerCallRadioButton.addActionListener(this);
	    
	    sortOrderGroup.add(ascendingButton);
	    sortOrderGroup.add(descendingButton);
	    sortOrderGroup.add(nameButton);
	    sortOrderGroup.add(inclusiveRadioButton);
	    sortOrderGroup.add(exclusiveRadioButton);
	    sortOrderGroup.add(numOfCallsRadioButton);
	    sortOrderGroup.add(numOfSubRoutinesRadioButton);
	    sortOrderGroup.add(userSecPerCallRadioButton);
	    
	    sortOrderMenu.add(ascendingButton);
	    sortOrderMenu.add(descendingButton);
	    sortOrderMenu.add(nameButton);
	    sortOrderMenu.add(inclusiveRadioButton);
	    sortOrderMenu.add(exclusiveRadioButton);
	    sortOrderMenu.add(numOfCallsRadioButton);
	    sortOrderMenu.add(numOfSubRoutinesRadioButton);
	    sortOrderMenu.add(userSecPerCallRadioButton);
	    
	    sortOrderMenu.insertSeparator(2);
	    
	    optionsMenu.add(sortOrderMenu);
	    //End Submenu.

	    //Add a submenu.
	    unitsMenu = new JMenu("Select Units");
	    unitsGroup = new ButtonGroup();
      
	    secondsButton = new JRadioButtonMenuItem("Seconds", false);
	    //Add a listener for this radio button.
	    secondsButton.addActionListener(this);
      
	    millisecondsButton = new JRadioButtonMenuItem("Milliseconds", false);
	    //Add a listener for this radio button.
	    millisecondsButton.addActionListener(this);
      
	    microsecondsButton = new JRadioButtonMenuItem("Microseconds", true);
	    //Add a listener for this radio button.
	    microsecondsButton.addActionListener(this);
      
	    hMSButton = new JRadioButtonMenuItem("hr:min:sec", false);
	    //Add a listener for this radio button.
	    hMSButton.addActionListener(this);

	    unitsGroup.add(secondsButton);
	    unitsGroup.add(millisecondsButton);
	    unitsGroup.add(microsecondsButton);
	    unitsGroup.add(hMSButton);
      
	    unitsMenu.add(secondsButton);
	    unitsMenu.add(millisecondsButton);
	    unitsMenu.add(microsecondsButton);
	    unitsMenu.add(hMSButton);
	    optionsMenu.add(unitsMenu);
	    //End Submenu.


	    //******************************
	    //End - Options menu.
	    //******************************
      
	    //******************************
	    //Window menu.
	    //******************************
	    JMenu windowsMenu = new JMenu("Windows");
	    windowsMenu.addMenuListener(this);
      
	    //Add a submenu.
	    JMenuItem mappingLedgerItem = new JMenuItem("Show Function Ledger");
	    mappingLedgerItem.addActionListener(this);
	    windowsMenu.add(mappingLedgerItem);
      
	    //Add a submenu.
	    mappingGroupLedgerItem = new JMenuItem("Show Group Ledger");
	    mappingGroupLedgerItem.addActionListener(this);
	    windowsMenu.add(mappingGroupLedgerItem);
      
	    //Add a submenu.
	    userEventLedgerItem = new JMenuItem("Show User Event Ledger");
	    userEventLedgerItem.addActionListener(this);
	    windowsMenu.add(userEventLedgerItem);
      
	    //Add a submenu.
	    JMenuItem closeAllSubwindowsItem = new JMenuItem("Close All Sub-Windows");
	    closeAllSubwindowsItem.addActionListener(this);
	    windowsMenu.add(closeAllSubwindowsItem);
	    //******************************
	    //End - Window menu.
	    //******************************
      
      
	    //******************************
	    //Help menu.
	    //******************************
	    JMenu helpMenu = new JMenu("Help");
      
	    //Add a menu item.
	    JMenuItem aboutItem = new JMenuItem("About ParaProf");
	    aboutItem.addActionListener(this);
	    helpMenu.add(aboutItem);
      
	    //Add a menu item.
	    JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
	    showHelpWindowItem.addActionListener(this);
	    helpMenu.add(showHelpWindowItem);
	    //******************************
	    //End - Help menu.
	    //******************************
      
      
	    //Now, add all the menus to the main menu.
	    mainMenu.add(fileMenu);
	    mainMenu.add(optionsMenu);
	    mainMenu.add(windowsMenu);
	    mainMenu.add(helpMenu);
      
	    setJMenuBar(mainMenu);
      
	    //******************************
	    //End - Code to generate the menus.
	    //******************************
      
      
	    //******************************
	    //Create and add the components.
	    //******************************
	    //Setting up the layout system for the main window.
	    Container contentPane = getContentPane();
	    GridBagLayout gbl = new GridBagLayout();
	    contentPane.setLayout(gbl);
	    GridBagConstraints gbc = new GridBagConstraints();
	    gbc.insets = new Insets(5, 5, 5, 5);
      
	    //Create some borders.
	    Border mainloweredbev = BorderFactory.createLoweredBevelBorder();
	    Border mainraisedbev = BorderFactory.createRaisedBevelBorder();
	    Border mainempty = BorderFactory.createEmptyBorder();
      
	    //**********
	    //Panel and ScrollPane definition.
	    //**********
	    panel = new CallPathTextWindowPanel(trial,
						   inServerNumber,
						   inContextNumber,
						   inThreadNumber, this, global);
      
	    //The scroll panes into which the list shall be placed.
	    JScrollPane sp = new JScrollPane(panel);
	    JScrollBar vScollBar = sp.getVerticalScrollBar();
	    vScollBar.setUnitIncrement(35);
	    sp.setBorder(mainloweredbev);
	    sp.setPreferredSize(new Dimension(500, 450));
	    //**********
	    //End - Panel and ScrollPane definition.
	    //**********
      
	    //Now add the componants to the main screen.
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 1;
	    gbc.weighty = 1;
	    addCompItem(sp, gbc, 0, 0, 1, 1);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW02");
	}
    }
  
    //******************************
    //Event listener code!!
    //******************************
  
    //ActionListener code.
    public void actionPerformed(ActionEvent evt){
	try{
	    Object EventSrc = evt.getSource();
	    
	    if(EventSrc instanceof JMenuItem){
		String arg = evt.getActionCommand();
		
		if(arg.equals("Close This Window")){
		    closeThisWindow();
		}
		else if(arg.equals("Exit ParaProf!")){
		    setVisible(false);
		    dispose();
		    System.exit(0);
		}
		else if(arg.equals("Name")){
		    if(nameButton.isSelected()){
			metric = "Name";
			panel.repaint();
		    }
		}
		else if(arg.equals("Descending")){
		    if(descendingButton.isSelected()){
			descendingOrder = true;
			panel.repaint();
		    }
		}
		else if(arg.equals("Ascending")){
		    if(ascendingButton.isSelected()){
			descendingOrder = false;
			panel.repaint();
		    }
		}
		else if(arg.equals("Inclusive")){
		    if(inclusiveRadioButton.isSelected()){
			metric = "Inclusive";
			panel.repaint();
		    }
		}
		else if(arg.equals("Exclusive")){
		    if(exclusiveRadioButton.isSelected()){
			metric = "Exclusive";
			panel.repaint();
		    }
		}
		else if(arg.equals("Number of Calls")){
		    if(numOfCallsRadioButton.isSelected()){
			metric = "Number of Calls";
			panel.repaint();
		    }
		}
		else if(arg.equals("Number of Subroutines")){
		    if(numOfSubRoutinesRadioButton.isSelected()){
			metric = "Number of Subroutines";
			panel.repaint();
		    }
		}
		else if(arg.equals("Per Call Value")){
		    if(userSecPerCallRadioButton.isSelected()){
			metric = "Per Call Value";
			panel.repaint();
		    }
		}
		else if(arg.equals("hr:min:sec")){
		    if(hMSButton.isSelected()){
			units = 3;
			//Call repaint.
			panel.repaint();
		    }
		}
		else if(arg.equals("Microseconds")){
			if(microsecondsButton.isSelected()){
			    units = 0;
			    //Call repaint.
			    panel.repaint();
			}
		}
		else if(arg.equals("Milliseconds")){
		    if(millisecondsButton.isSelected()){
			units = 1;
			//Call repaint.
			panel.repaint();
		    }
		}
		else if(arg.equals("Seconds")){
		    if(secondsButton.isSelected()){
			units = 2;
			//Call repaint.
			panel.repaint();
		    }
		}
		else if(arg.equals("Show Function Ledger")){
		    //In order to be in this window, I must have loaded the data. So,
		    //just show the mapping ledger window.
		    (trial.getGlobalMapping()).displayMappingLedger(0);
		}
		else if(arg.equals("Show Group Ledger")){
		    //In order to be in this window, I must have loaded the data. So,
		    //just show the mapping ledger window.
		    (trial.getGlobalMapping()).displayMappingLedger(1);
		}
		else if(arg.equals("Show User Event Ledger")){
		    //In order to be in this window, I must have loaded the data. So,
		    //just show the mapping ledger window.
		    (trial.getGlobalMapping()).displayMappingLedger(2);
		}
		else if(arg.equals("Close All Sub-Windows")){
		    //Close the all subwindows.
		    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
		}
		else if(arg.equals("About ParaProf")){
		    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
		}
		else if(arg.equals("Show Help Window")){
		    //Show the racy help window.
		    ParaProf.helpWindow.clearText();
		    ParaProf.helpWindow.show();
		    //Since the data must have been loaded.  Tell them someting about
		    //where they are.
		    ParaProf.helpWindow.writeText("Call path text window.");
		    ParaProf.helpWindow.writeText("");
		    ParaProf.helpWindow.writeText("This window displays call path relationships in two ways:");
		    ParaProf.helpWindow.writeText("1- If this window has been invoked from the \"windows\" menu of");
		    ParaProf.helpWindow.writeText("ParaProf, the information displayed is all call path relations found.");
		    ParaProf.helpWindow.writeText("That is, all the parent/child relationships.");
		    ParaProf.helpWindow.writeText("Thus, in this case, given the parallel nature of ParaProf, this information");
		    ParaProf.helpWindow.writeText("might not be valid for a particular thread. It is however useful to observe");
		    ParaProf.helpWindow.writeText("all the realtionships that exist in the data.");
		    ParaProf.helpWindow.writeText("");
		    ParaProf.helpWindow.writeText("2- If this window has been invoked from the popup menu to the left of a thread bar");
		    ParaProf.helpWindow.writeText("in the main ParaProf window, the information dispayed will be specific to this thread,");
		    ParaProf.helpWindow.writeText("and will thus contain both parent/child relations and the data relating to those");
		    ParaProf.helpWindow.writeText("relationships.");
		}
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW03");
	}
    }
    
    public int units(){
	return units;}
    
    //******************************
    //MenuListener code.
    //******************************
    public void menuSelected(MenuEvent evt){
	try{
	    String trialName = trial.getCounterName();
	    trialName = trialName.toUpperCase();
	    if((trial.isTimeMetric())&&(!global))
		unitsMenu.setEnabled(true);
	    else
		unitsMenu.setEnabled(false);


	    if(trial.groupNamesPresent())
		mappingGroupLedgerItem.setEnabled(true);
	    else
		mappingGroupLedgerItem.setEnabled(false);
	    
	    if(trial.userEventsPresent())
		userEventLedgerItem.setEnabled(true);
	    else{
		userEventLedgerItem.setEnabled(false);
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW03");
	}
	
    }
    public void menuDeselected(MenuEvent evt){}
    public void menuCanceled(MenuEvent evt){}
    //******************************
    //End - MenuListener code.
    //******************************
  
    //Observer functions.
    public void update(Observable o, Object arg){
	try{
	    String tmpString = (String) arg;
	    if(tmpString.equals("prefEvent")){
		panel.repaint();
	    }
	    if(tmpString.equals("colorEvent")){
		panel.repaint();
	    }
	    else if(tmpString.equals("dataEvent")){ 
		panel.repaint();
	    }
	    else if(tmpString.equals("subWindowCloseEvent")){ 
		closeThisWindow();
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW04");
	}
    } 
  
    //Helper functions.
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	try{
	    gbc.gridx = x;
	    gbc.gridy = y;
	    gbc.gridwidth = w;
	    gbc.gridheight = h;
	    
	    getContentPane().add(c, gbc);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW05");
	}
    }
  
    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e){
	closeThisWindow();
    }

    //This function passes the correct data list to its panel when asked for.
    //Note:  This is only meant to be called by the TotalStatWindowPanel.
    public Vector getData(){
	try{
	    if(metric.equals("Name")){
		if(descendingOrder)
		    return sMWData.getSMWThreadData(server, context, thread, "NDE");
		else
		    return sMWData.getSMWThreadData(server, context, thread, "NAE");
	    }
	    if(metric.equals("Inclusive")){
		if(descendingOrder)
		    return sMWData.getSMWThreadData(server, context, thread, "MDI");
		else
		    return sMWData.getSMWThreadData(server, context, thread, "MAI");
	    }
	    else if(metric.equals("Exclusive")){
		if(descendingOrder)
		    return sMWData.getSMWThreadData(server, context, thread, "MDE");
		else
		    return sMWData.getSMWThreadData(server, context, thread, "MAE");
	    }
	    else if(metric.equals("Number of Calls")){
		if(descendingOrder)
		    return sMWData.getSMWThreadData(server, context, thread, "MDNC");
		else
		    return sMWData.getSMWThreadData(server, context, thread, "MANC");
	    }
	    else if(metric.equals("Number of Subroutines")){
		if(descendingOrder)
		    return sMWData.getSMWThreadData(server, context, thread, "MDNS");
		else
		    return sMWData.getSMWThreadData(server, context, thread, "MANS");
	    }
	    else if(metric.equals("Per Call Value")){
		if(descendingOrder)
		    return sMWData.getSMWThreadData(server, context, thread, "MDUS");
		else
		    return sMWData.getSMWThreadData(server, context, thread, "MAUS");
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CTPW06");
	}
	return null;
    }

    public ListIterator getDataIterator(){
	return new ParaProfIterator(this.getData());
    }
  
    void closeThisWindow(){ 
	try{
	    if(ParaProf.debugIsOn){
		System.out.println("------------------------");
		System.out.println("A total stat window for: \"" + "n,c,t, " + server + "," + context + "," + thread + "\" is closing");
		System.out.println("Clearing resourses for this window.");
	    }
      
	    setVisible(false);
	    trial.getSystemEvents().deleteObserver(this);
	    dispose();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW06");
	}
    }
  
    //******************************
    //Instance data.
    //******************************
    private Trial trial = null;
    private CallPathTextWindowPanel panel;
    private StaticMainWindowData sMWData;
    private boolean global = false;
  
    private JMenu unitsMenu = null;;
    private JMenuItem mappingGroupLedgerItem = null;;
    private JMenuItem userEventLedgerItem = null;;

    private ButtonGroup sortGroup = null;;
    private ButtonGroup sortOrderGroup = null;;
    private ButtonGroup unitsGroup = null;
    private ButtonGroup metricGroup = null;
  
    JRadioButtonMenuItem ascendingButton = new JRadioButtonMenuItem("Ascending", false);
    JRadioButtonMenuItem descendingButton = new JRadioButtonMenuItem("Descending", true);
    
    private JRadioButtonMenuItem nameButton = new JRadioButtonMenuItem("Name", false);
    private JRadioButtonMenuItem inclusiveRadioButton =  new JRadioButtonMenuItem("Inclusive", false);
    private JRadioButtonMenuItem exclusiveRadioButton = new JRadioButtonMenuItem("Exclusive", true);
    private JRadioButtonMenuItem numOfCallsRadioButton =  new JRadioButtonMenuItem("Number of Calls", false);
    private JRadioButtonMenuItem numOfSubRoutinesRadioButton = new JRadioButtonMenuItem("Number of Subroutines", false);
    private JRadioButtonMenuItem userSecPerCallRadioButton = new JRadioButtonMenuItem("Per Call Value", false);
    private JRadioButtonMenuItem secondsButton = null;
    private JRadioButtonMenuItem millisecondsButton = null;
    private JRadioButtonMenuItem microsecondsButton = null;
    private JRadioButtonMenuItem hMSButton = null;
    
    private String metric = "Exclusive";
    
    boolean sortByMappingID = false;
    boolean sortByName = false;
    boolean sortByMillisecond = true;    
    boolean descendingOrder = true;
    boolean inclusive = false;
    private int units = 0; //0-microseconds,1-milliseconds,2-seconds,3-hr:min:sec.


    int server;
    int context;
    int thread;
    //******************************
    //End - Instance data.
    //******************************
}
