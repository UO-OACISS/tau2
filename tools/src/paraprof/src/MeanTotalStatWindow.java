/* 
   MeanTotalStatWindow.java

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

public class MeanTotalStatWindow extends JFrame implements ActionListener, MenuListener, Observer 
{
  
    public MeanTotalStatWindow()
    {
	try{
	    setLocation(new java.awt.Point(0, 0));
	    setSize(new java.awt.Dimension(800, 600));
      
	    //Set the title indicating that there was a problem.
	    this.setTitle("Wrong constructor used!");
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MTSW01");
	    }
    }
  
    public MeanTotalStatWindow(Trial inTrial, StaticMainWindowData inSMWData)
    {
	try{
	    trial = inTrial;
	    sMWData = inSMWData;
      
	    setLocation(new java.awt.Point(0, 0));
	    setSize(new java.awt.Dimension(800, 600));
      
	    //Now set the title.
	    this.setTitle("Mean Total Stat Window: " + trial.getProfilePathName());
      
	    //Add some window listener code
	    addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(java.awt.event.WindowEvent evt) {
			thisWindowClosing(evt);
		    }
		});
        
        
	    //Set the help window text if required.
	    if(ParaProf.helpWindow.isVisible())
		{
		    ParaProf.helpWindow.clearText();
		    //Since the data must have been loaded.  Tell them someting about
		    //where they are.
		    ParaProf.helpWindow.writeText("This is the mean total mapping statistics window.");
		    ParaProf.helpWindow.writeText("");
		    ParaProf.helpWindow.writeText("This window shows you the mean total statistics for all mappings.");
		    ParaProf.helpWindow.writeText("");
		    ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
		    ParaProf.helpWindow.writeText("");
		    ParaProf.helpWindow.writeText("Right click on any mapping within this window to bring up a popup");
		    ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
		    ParaProf.helpWindow.writeText("for the mapping, or to show more details about the mapping.");
		    ParaProf.helpWindow.writeText("You can also left click any mapping to hightlight it in the system.");
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
      
	    //Add a submenu.
	    JMenu sortMenu = new JMenu("Sort by ...");
	    sortGroup = new ButtonGroup();
      
	    mappingIDButton = new JRadioButtonMenuItem("function ID", false);
	    //Add a listener for this radio button.
	    mappingIDButton.addActionListener(this);
      
	    nameButton = new JRadioButtonMenuItem("name", false);
	    //Add a listener for this radio button.
	    nameButton.addActionListener(this);
      
	    millisecondButton = new JRadioButtonMenuItem("millisecond", true);
	    //Add a listener for this radio button.
	    millisecondButton.addActionListener(this);
      
	    sortGroup.add(mappingIDButton);
	    sortGroup.add(nameButton);
	    sortGroup.add(millisecondButton);
      
	    sortMenu.add(mappingIDButton);
	    sortMenu.add(nameButton);
	    sortMenu.add(millisecondButton);
	    optionsMenu.add(sortMenu);
	    //End Submenu.
      
	    //Add a submenu.
	    JMenu sortOrderMenu = new JMenu("Sort Order");
	    sortOrderGroup = new ButtonGroup();
      
	    ascendingButton = new JRadioButtonMenuItem("Ascending", false);
	    //Add a listener for this radio button.
	    ascendingButton.addActionListener(this);
      
	    descendingButton = new JRadioButtonMenuItem("Descending", true);
	    //Add a listener for this radio button.
	    descendingButton.addActionListener(this);
      
	    sortOrderGroup.add(ascendingButton);
	    sortOrderGroup.add(descendingButton);
      
	    sortOrderMenu.add(ascendingButton);
	    sortOrderMenu.add(descendingButton);
	    optionsMenu.add(sortOrderMenu);
	    //End Submenu.
      
	    //Add a submenu.
	    JMenu metricMenu = new JMenu("Select Metric");
	    metricGroup = new ButtonGroup();
      
	    //Add listeners
	    inclusiveRadioButton.addActionListener(this);
	    exclusiveRadioButton.addActionListener(this);
	    numOfCallsRadioButton.addActionListener(this);
	    numOfSubRoutinesRadioButton.addActionListener(this);
	    userSecPerCallRadioButton.addActionListener(this);
      
	    metricGroup.add(inclusiveRadioButton);
	    metricGroup.add(exclusiveRadioButton);
	    metricGroup.add(numOfCallsRadioButton);
	    metricGroup.add(numOfSubRoutinesRadioButton);
	    metricGroup.add(userSecPerCallRadioButton);
	    metricMenu.add(inclusiveRadioButton);
	    metricMenu.add(exclusiveRadioButton);
	    metricMenu.add(numOfCallsRadioButton);
	    metricMenu.add(numOfSubRoutinesRadioButton);
	    metricMenu.add(userSecPerCallRadioButton);
	    optionsMenu.add(metricMenu);
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
	    //Create and add the componants.
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
	    panel = new MeanTotalStatWindowPanel(trial, this);
      
	    //The scroll panes into which the list shall be placed.
	    JScrollPane totalStatWindowPanelScrollPane = new JScrollPane(panel);
	    totalStatWindowPanelScrollPane.setBorder(mainloweredbev);
	    totalStatWindowPanelScrollPane.setPreferredSize(new Dimension(500, 450));
	    //**********
	    //End - Panel and ScrollPane definition.
	    //**********
      
	    //Now add the componants to the main screen.
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 1;
	    gbc.weighty = 1;
	    addCompItem(totalStatWindowPanelScrollPane, gbc, 0, 0, 1, 1);
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MTSW02");
	    }
    }

    public int units(){
	return units;}
  
    //******************************
    //Event listener code!!
    //******************************
  
    //ActionListener code.
    public void actionPerformed(ActionEvent evt)
    {
	try{
	    Object EventSrc = evt.getSource();
      
	    if(EventSrc instanceof JMenuItem)
		{
		    String arg = evt.getActionCommand();

		    if(arg.equals("Close This Window"))
			{
			    closeThisWindow();
			}
		    else if(arg.equals("Exit ParaProf!"))
			{
			    setVisible(false);
			    dispose();
			    System.exit(0);
			}
		    else if(arg.equals("function ID"))
			{
			    if(mappingIDButton.isSelected())
				{
				    sortByMappingID = true;
				    sortByName = false;
				    sortByMillisecond = false;
				    //Call repaint.
				    panel.repaint();
				}
			}
		    else if(arg.equals("name"))
			{
			    if(nameButton.isSelected())
				{
				    sortByMappingID = false;
				    sortByName = true;
				    sortByMillisecond = false;
				    //Call repaint.
				    panel.repaint();
				}
			}
		    else if(arg.equals("millisecond"))
			{
			    if(millisecondButton.isSelected())
				{
				    sortByMappingID = false;
				    sortByName = false;
				    sortByMillisecond = true;
				    //Call repaint.
				    panel.repaint();
				}
			}
		    else if(arg.equals("Descending"))
			{
			    if(descendingButton.isSelected())
				{
				    descendingOrder = true;
				    panel.repaint();
				}
			}
		    else if(arg.equals("Ascending"))
			{
			    if(ascendingButton.isSelected())
				{
				    descendingOrder = false;
				    panel.repaint();
				}
			}
		    else if(arg.equals("Inclusive"))
			{
			    if(inclusiveRadioButton.isSelected())
				{
				    metric = "Inclusive";
				    //Call repaint.
				    panel.repaint();
				}
			}
		    else if(arg.equals("Exclusive"))
			{
			    if(exclusiveRadioButton.isSelected())
				{
				    metric = "Exclusive";
				    //Call repaint.
				    panel.repaint();
				}
			}
		    else if(arg.equals("Number of Calls"))
			{
			    if(numOfCallsRadioButton.isSelected())
				{
				    metric = "Number of Calls";
				    //Call repaint.
				    panel.repaint();
				}
			}
		    else if(arg.equals("Number of Subroutines"))
			{
			    if(numOfSubRoutinesRadioButton.isSelected())
				{
				    metric = "Number of Subroutines";
				    //Call repaint.
				    panel.repaint();
				}
			}
		    else if(arg.equals("Per Call Value"))
			{
			    if(userSecPerCallRadioButton.isSelected())
				{
				    metric = "Per Call Value";
				    //Call repaint.
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
		    else if(arg.equals("Show Function Ledger"))
			{
			    //In order to be in this window, I must have loaded the data. So,
			    //just show the mapping ledger window.
			    (trial.getGlobalMapping()).displayMappingLedger(0);
			}
		    else if(arg.equals("Show Group Ledger"))
			{
			    //In order to be in this window, I must have loaded the data. So,
			    //just show the mapping ledger window.
			    (trial.getGlobalMapping()).displayMappingLedger(1);
			}
		    else if(arg.equals("Show User Event Ledger"))
			{
			    //In order to be in this window, I must have loaded the data. So,
			    //just show the mapping ledger window.
			    (trial.getGlobalMapping()).displayMappingLedger(2);
			}
		    else if(arg.equals("Close All Sub-Windows"))
			{
			    //Close the all subwindows.
			    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
			}
		    else if(arg.equals("About Racy"))
			{
			    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
			}
		    else if(arg.equals("Show Help Window"))
			{
			    //Show the racy help window.
			    ParaProf.helpWindow.clearText();
			    ParaProf.helpWindow.show();
			    //Since the data must have been loaded.  Tell them someting about
			    //where they are.
			    ParaProf.helpWindow.writeText("This is the mean total mapping statistics window.");
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("This window shows you the mean total statistics for all mappings.");
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("Right click on any mapping within this window to bring up a popup");
			    ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
			    ParaProf.helpWindow.writeText("for the mapping, or to show more details about the mapping.");
			    ParaProf.helpWindow.writeText("You can also left click any mapping to hightlight it in the system.");
			}
		}
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MTSW03");
	    }
    }
  
    //******************************
    //MenuListener code.
    //******************************
    public void menuSelected(MenuEvent evt)
    {
	try
	    {
		if(trial.isTimeMetric())
		    unitsMenu.setEnabled(true);
		else
		    unitsMenu.setEnabled(false);
		
		if(trial.groupNamesPresent())
		    mappingGroupLedgerItem.setEnabled(true);
		else
		    mappingGroupLedgerItem.setEnabled(false);
		
		if(trial.userEventsPresent())
		    userEventLedgerItem.setEnabled(true);
		else
		    userEventLedgerItem.setEnabled(false);
	    }
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "SMW03");
	    }
    
    }
  
    public void menuDeselected(MenuEvent evt)
    {
    }
  
    public void menuCanceled(MenuEvent evt)
    {
    }
  
    //******************************
    //End - MenuListener code.
    //******************************
  
    //Observer functions.
    public void update(Observable o, Object arg)
    {
	try{
	    String tmpString = (String) arg;
	    if(tmpString.equals("prefEvent"))
		{
		    //Just need to call a repaint on the ThreadDataWindowPanel.
		    panel.repaint();
		}
	    else if(tmpString.equals("colorEvent"))
		{
		    //Just need to call a repaint on the ThreadDataWindowPanel.
		    panel.repaint();
		}
	    else if(tmpString.equals("dataEvent"))
		{
		    lastSorting = "";
		    if(!(trial.isTimeMetric()))
			units = 0;
		    panel.repaint();
		}
	    else if(tmpString.equals("subWindowCloseEvent"))
		{
		    closeThisWindow();
		}
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MTSW04");
	    }
    } 
  
    //Helper functions.
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h)
    {
	try{
	    gbc.gridx = x;
	    gbc.gridy = y;
	    gbc.gridwidth = w;
	    gbc.gridheight = h;
      
	    getContentPane().add(c, gbc);
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MTSW05");
	    }
    }
  
    //This function passes the correct data list to its panel when asked for.
    //Note:  This is only meant to be called by the TotalStatWindowPanel.
    public Vector getStaticMainWindowSystemData()
    {
	try
	    {
		if(sortByMappingID)
		    {
			if(metric.equals("Inclusive"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("FIdDI");
				else
				    return sMWData.getSMWMeanData("FIdAI");
			    }
			else if(metric.equals("Exclusive"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("FIdDE");
				else
				    return sMWData.getSMWMeanData("FIdAE");
			    }
			else if(metric.equals("Number of Calls"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("FIdDNC");
				else
				    return sMWData.getSMWMeanData("FIdANC");
			    }
			else if(metric.equals("Number of Subroutines"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("FIdDNS");
				else
				    return sMWData.getSMWMeanData("FIdANS");
			    }
			else if(metric.equals("Per Call Value"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("FIdDUS");
				else
				    return sMWData.getSMWMeanData("FIdAUS");
			    }
		    }
		else if(sortByName)
		    {
        
			if(metric.equals("Inclusive"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("NDI");
				else
				    return sMWData.getSMWMeanData("NAI");
			    }
			else if(metric.equals("Exclusive"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("NDE");
				else
				    return sMWData.getSMWMeanData("NAE");
			    }
			else if(metric.equals("Number of Calls"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("NDNC");
				else
				    return sMWData.getSMWMeanData("NANC");
			    }
			else if(metric.equals("Number of Subroutines"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("NDNS");
				else
				    return sMWData.getSMWMeanData("NANS");
			    }
			else if(metric.equals("Per Call Value"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("NDUS");
				else
				    return sMWData.getSMWMeanData("NAUS");
			    }
		    }
		else if(sortByMillisecond)
		    {
        
			if(metric.equals("Inclusive"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("MDI");
				else
				    return sMWData.getSMWMeanData("MAI");
			    }
			else if(metric.equals("Exclusive"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("MDE");
				else
				    return sMWData.getSMWMeanData("MAE");
			    }
			else if(metric.equals("Number of Calls"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("MDNC");
				else
				    return sMWData.getSMWMeanData("MANC");
			    }
			else if(metric.equals("Number of Subroutines"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("MDNS");
				else
				    return sMWData.getSMWMeanData("MANS");
			    }
			else if(metric.equals("Per Call Value"))
			    {
				if(descendingOrder)
				    return sMWData.getSMWMeanData("MDUS");
				else
				    return sMWData.getSMWMeanData("MAUS");
			    }
		    }
	    }
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MTSW06");
	    }
    
	return null;
    
    }
  
    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e)
    {
	closeThisWindow();
    }
  
    void closeThisWindow()
    { 
	try
	    {
		if(ParaProf.debugIsOn)
		    {
			System.out.println("------------------------");
			System.out.println("The Mean Total Stat Window is closing");
			System.out.println("Clearing resourses for this window.");
		    }
      
		setVisible(false);
		trial.getSystemEvents().deleteObserver(this);
		dispose();
	    }
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MTSW07");
	    }
    }
  
    //******************************
    //Instance data.
    //******************************
    private Trial trial = null;
    private MeanTotalStatWindowPanel panel;
    private StaticMainWindowData sMWData = null;
  
    private JMenu unitsMenu = null;
    private JMenuItem mappingGroupLedgerItem;
    private JMenuItem userEventLedgerItem;
  
    ButtonGroup sortGroup;
    ButtonGroup sortOrderGroup;
    private ButtonGroup unitsGroup = null;
    private ButtonGroup metricGroup = null;
  
    JRadioButtonMenuItem mappingIDButton;
    JRadioButtonMenuItem nameButton;
    JRadioButtonMenuItem millisecondButton;
  
    JRadioButtonMenuItem ascendingButton;
    JRadioButtonMenuItem descendingButton;
  
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
  
  
    String lastSorting = "";
    Vector lastList;
  
  
    
    //******************************
    //End - Instance data.
    //******************************

}
