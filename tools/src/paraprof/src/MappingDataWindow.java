/* 
   MappingDataWindow.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  The container for the MappingDataWindowPanel.
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

public class MappingDataWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener, AdjustmentListener
{
  
    public MappingDataWindow()
    {
	try{
	    setLocation(new java.awt.Point(300, 200));
	    setSize(new java.awt.Dimension(550, 550));
      
	    //Set the title indicating that there was a problem.
	    this.setTitle("Wrong constructor used");
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MDW01");
	    }
    }
  
    public MappingDataWindow(Trial inTrial, int inMappingID, StaticMainWindowData inSMWData)
    {
	try{
      
      
	    mappingID = inMappingID;
	    trial = inTrial;
	    sMWData = inSMWData;
      
      
	    setLocation(new java.awt.Point(300, 200));
	    setSize(new java.awt.Dimension(550, 550));
      
	    inclusive = false;
	    percent = true;
	    unitsString = "milliseconds";
      
      
	    //Grab the appropriate global mapping element.
	    GlobalMapping tmpGM = trial.getGlobalMapping();
	    GlobalMappingElement tmpGME = tmpGM.getGlobalMappingElement(inMappingID, 0);
      
	    mappingName = tmpGME.getMappingName();
      
	    //Now set the title.
	    this.setTitle("Function Data Window: " + trial.getProfilePathName());
      
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
		    ParaProf.helpWindow.writeText("This is the function data window for:");
		    ParaProf.helpWindow.writeText(mappingName);
		    ParaProf.helpWindow.writeText("");
		    ParaProf.helpWindow.writeText("This window shows you this function's statistics across all the threads.");
		    ParaProf.helpWindow.writeText("");
		    ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
		    ParaProf.helpWindow.writeText("");
		    ParaProf.helpWindow.writeText("Right click anywhere within this window to bring up a popup");
		    ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
		    ParaProf.helpWindow.writeText("for this function.");
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
	    JMenuItem showBinWindowItem = new JMenuItem("Bin Window");
	    showBinWindowItem.addActionListener(this);
	    fileMenu.add(showBinWindowItem);
      
	    //Add a menu item.
	    JMenuItem closeItem = new JMenuItem("Close This Window");
	    closeItem.addActionListener(this);
	    fileMenu.add(closeItem);
      
	    //Add a menu item.
	    JMenuItem exitItem = new JMenuItem("Exit Racy!");
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
	    valuePercentMenu = new JMenu("Select Value or Percent");
	    valuePercentGroup = new ButtonGroup();
      
	    percentButton = new JRadioButtonMenuItem("Percent", true);
	    //Add a listener for this radio button.
	    percentButton.addActionListener(this);
      
	    valueButton = new JRadioButtonMenuItem("Value", false);
	    //Add a listener for this radio button.
	    valueButton.addActionListener(this);
      
	    valuePercentGroup.add(percentButton);
	    valuePercentGroup.add(valueButton);
      
	    valuePercentMenu.add(percentButton);
	    valuePercentMenu.add(valueButton);
	    optionsMenu.add(valuePercentMenu);
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
      
	    unitsGroup.add(secondsButton);
	    unitsGroup.add(millisecondsButton);
	    unitsGroup.add(microsecondsButton);
      
	    unitsMenu.add(secondsButton);
	    unitsMenu.add(millisecondsButton);
	    unitsMenu.add(microsecondsButton);
	    optionsMenu.add(unitsMenu);
	    //End Submenu.
      
	    displaySlidersButton = new JRadioButtonMenuItem("Display Sliders", false);
	    //Add a listener for this radio button.
	    displaySlidersButton.addActionListener(this);
	    optionsMenu.add(displaySlidersButton);
          
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
      
	    //Add listeners
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
	    JMenuItem aboutItem = new JMenuItem("About Racy");
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
	    contentPane = getContentPane();
	    gbl = new GridBagLayout();
	    contentPane.setLayout(gbl);
	    gbc = new GridBagConstraints();
	    gbc.insets = new Insets(5, 5, 5, 5);
      
	    //Create some borders.
	    Border mainloweredbev = BorderFactory.createLoweredBevelBorder();
	    Border mainraisedbev = BorderFactory.createRaisedBevelBorder();
	    Border mainempty = BorderFactory.createEmptyBorder();
      
      
	    //**********
	    //Panel and ScrollPane definition.
	    //**********
	    mappingDataWinPanelRef = new MappingDataWindowPanel(trial, inMappingID, this);
	    //The scroll panes into which the list shall be placed.
	    mappingDataWinPanelScrollPane = new JScrollPane(mappingDataWinPanelRef);
	    mappingDataWinPanelScrollPane.setBorder(mainloweredbev);
	    mappingDataWinPanelScrollPane.setPreferredSize(new Dimension(500, 450));
	    //**********
	    //End - Panel and ScrollPane definition.
	    //**********
      
	    //Do the slider stuff, but don't add.  By default, sliders are off.
	    String sliderMultipleStrings[] = {"1.00", "0.75", "0.50", "0.25", "0.10"};
	    sliderMultiple = new JComboBox(sliderMultipleStrings);
	    sliderMultiple.addActionListener(this);
      
	    barLengthSlider.setPaintTicks(true);
	    barLengthSlider.setMajorTickSpacing(5);
	    barLengthSlider.setMinorTickSpacing(1);
	    barLengthSlider.setPaintLabels(true);
	    barLengthSlider.setSnapToTicks(true);
	    barLengthSlider.addChangeListener(this);
      
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 0.95;
	    gbc.weighty = 0.98;
	    addCompItem(mappingDataWinPanelScrollPane, gbc, 0, 0, 1, 1);
	    //addCompItem(mappingDataWinPanelRef, gbc, 0, 0, 1, 1);
      
	    //hsb.addAdjustmentListener(this);
      
	    //gbc.fill = GridBagConstraints.BOTH;
	    //gbc.anchor = GridBagConstraints.CENTER;
	    //gbc.weightx = 0.95;
	    //gbc.weighty = 0.02;
	    //addCompItem(mappingDataWinPanelScrollPane, gbc, 0, 0, 1, 1);
	    //addCompItem(hsb, gbc, 0, 1, 1, 1);
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MDW02");
	    }
    
    
    }
  
    //******************************
    //Event listener code!!
    //******************************
  
    //******************************
    //ActionListener code.
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
		    else if(arg.equals("Exit Racy!"))
			{
			    setVisible(false);
			    dispose();
			    System.exit(0);
			}
		    else if(arg.equals("Bin Window"))
			{
			    System.out.println("Mapping is in MDW is: " + mappingID);
			    BinWindow bW = new BinWindow(trial, sMWData, false, mappingID);
			    bW.show();
			}
		    else if(arg.equals("Inclusive"))
			{
			    if(inclusiveRadioButton.isSelected())
				{
				    metric = "Inclusive";
				    //Call repaint.
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Exclusive"))
			{
			    if(exclusiveRadioButton.isSelected())
				{
				    metric = "Exclusive";
				    //Call repaint.
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Number of Calls"))
			{
			    if(numOfCallsRadioButton.isSelected())
				{
				    metric = "Number of Calls";
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Number of Subroutines"))
			{
			    if(numOfSubRoutinesRadioButton.isSelected())
				{
				    metric = "Number of Subroutines";
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Per Call Value"))
			{
			    if(userSecPerCallRadioButton.isSelected())
				{
				    metric = "Per Call Value";
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Percent"))
			{
			    if(percentButton.isSelected())
				{
				    percent = true;
				    //Call repaint.
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Value"))
			{
			    if(valueButton.isSelected())
				{
				    percent = false;
				    //Call repaint.
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Seconds"))
			{
			    if(secondsButton.isSelected())
				{
				    unitsString = "Seconds";
				    //Call repaint.
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Microseconds"))
			{
			    if(microsecondsButton.isSelected())
				{
				    unitsString = "Microseconds";
				    //Call repaint.
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Milliseconds"))
			{
			    if(millisecondsButton.isSelected())
				{
				    unitsString = "Milliseconds";
				    //Call repaint.
				    mappingDataWinPanelRef.repaint();
				}
			}
		    else if(arg.equals("Display Sliders"))
			{
			    if(displaySlidersButton.isSelected())
				{ 
				    displaySiders(true);
				}
			    else
				{
				    displaySiders(false);
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
			    ParaProf.helpWindow.writeText("This is the mapping data window for:");
			    ParaProf.helpWindow.writeText(mappingName);
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("This window shows you this mapping's statistics across all the threads.");
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("Right click anywhere within this window to bring up a popup");
			    ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
			    ParaProf.helpWindow.writeText("for this mapping.");
			}
		}
	    else if(EventSrc == sliderMultiple)
		{
		    mappingDataWinPanelRef.changeInMultiples();
		}
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MDW03");
	    }
    }
    //******************************
    //End - ActionListener code.
    //******************************
  
  
    //******************************
    //Change listener code.
    //******************************
    public void stateChanged(ChangeEvent event)
    {
	mappingDataWinPanelRef.changeInMultiples();
    }
    //******************************
    //End - Change listener code.
    //******************************
  
  
    //******************************
    //Adjustment listener code.
    //******************************
    public void adjustmentValueChanged(AdjustmentEvent e){
	Object EventSrc = e.getSource();
	if(EventSrc == hsb){
	    JScrollBar sb = (JScrollBar) EventSrc;
      
	    System.out.println("min: " + hsb.getMinimum() + ", max: " + hsb.getMaximum() +
			       ", visible amount: " + hsb.getVisibleAmount() +
			       ", value: " + hsb.getValue());
                
	    hsbValue = hsb.getValue();
                
	    mappingDataWinPanelRef.repaint();
	}
    }
    //******************************
    //End - Adjustment listener code.
    //******************************
  
    //******************************
    //MenuListener code.
    //******************************
    public void menuSelected(MenuEvent evt)
    {
	try
	    {
		String trialName = trial.getCounterName();
		trialName = trialName.toUpperCase();
		boolean isDefault = false;
		boolean isTimeMetric = false;
      
		if(trialName.equals("DEFAULT")) 
		    isDefault = true;
		else if(trialName.indexOf("TIME") != -1)
		    isTimeMetric = true;
      
		if(trial.groupNamesPresent())
		    mappingGroupLedgerItem.setEnabled(true);
		else
		    mappingGroupLedgerItem.setEnabled(false);
        
		if(trial.userEventsPresent())
		    userEventLedgerItem.setEnabled(true);
		else
		    userEventLedgerItem.setEnabled(false);
        
        
      
		if((metric.equals("Number of Calls")) || (metric.equals("Number of Subroutines")) || (metric.equals("Per Call Value"))){
		    valuePercentMenu.setEnabled(false);
		    unitsMenu.setEnabled(false);}
		else if(percent){
		    valuePercentMenu.setEnabled(true);
		    unitsMenu.setEnabled(false);}
		else{
		    valuePercentMenu.setEnabled(true);
		    unitsMenu.setEnabled(true);}
	    }
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "TDW04");
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
		    mappingDataWinPanelRef.repaint();
		}
	    else if(tmpString.equals("colorEvent"))
		{
		    //Just need to call a repaint on the ThreadDataWindowPanel.
		    mappingDataWinPanelRef.repaint();
		}
	    else if(tmpString.equals("dataSetChangeEvent"))
		{
		    //Clear any locally saved data.
		    sMWGeneralData = null;
		}
	    else if(tmpString.equals("subWindowCloseEvent"))
		{
		    closeThisWindow();
		}
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MDW05");
	    }
    }
  
    //MappingDataWindowPanel call back functions.
    public Vector getStaticMainWindowSystemData()
    {
	try{
	    if(sMWGeneralData == null)
		{
		    sMWGeneralData = sMWData.getSMWMappingData(mappingID);
		    return sMWGeneralData;
		}
	    else
		{
		    return sMWGeneralData;
		}
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MDW06");
	    }
    
	return null;
    }
  
    public String getMetric(){
	return metric;
    }
    
    public boolean isInclusive()
    {
	return inclusive;
    }
  
    public boolean isPercent()
    {
	return percent;
    }
  
    public String units()
    {
	return unitsString;
    }
  
    public int getSliderValue()
    {
	int tmpInt = -1;
    
	try
	    {
		tmpInt = barLengthSlider.getValue();
	    }
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MDW07");
	    }
    
	return tmpInt;
    }
  
    public double getSliderMultiple()
    {
	String tmpString = null;
	try
	    {
		tmpString = (String) sliderMultiple.getSelectedItem();
		return Double.parseDouble(tmpString);
	    }
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MDW08");
	    }
    
	return 0;
    }

    public Rectangle getViewRectangle(){
	return mappingDataWinPanelScrollPane.getViewport().getViewRect();}
      
    private void displaySiders(boolean displaySliders)
    {
	if(displaySliders)
	    {
		//Since the menu option is a toggle, the only component that needs to be
		//removed is that scrollPane.  We then add back in with new parameters.
		//This might not be required as it seems to adjust well if left in, but just
		//to be sure.
		contentPane.remove(mappingDataWinPanelScrollPane);
      
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.EAST;
		gbc.weightx = 0.10;
		gbc.weighty = 0.01;
		addCompItem(sliderMultipleLabel, gbc, 0, 0, 1, 1);
      
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.WEST;
		gbc.weightx = 0.10;
		gbc.weighty = 0.01;
		addCompItem(sliderMultiple, gbc, 1, 0, 1, 1);
      
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.EAST;
		gbc.weightx = 0.10;
		gbc.weighty = 0.01;
		addCompItem(barLengthLabel, gbc, 2, 0, 1, 1);
      
		gbc.fill = GridBagConstraints.HORIZONTAL;
		gbc.anchor = GridBagConstraints.WEST;
		gbc.weightx = 0.70;
		gbc.weighty = 0.01;
		addCompItem(barLengthSlider, gbc, 3, 0, 1, 1);
      
		gbc.fill = GridBagConstraints.BOTH;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 0.95;
		gbc.weighty = 0.98;
		addCompItem(mappingDataWinPanelScrollPane, gbc, 0, 1, 4, 1);
	    }
	else
	    {
		contentPane.remove(sliderMultipleLabel);
		contentPane.remove(sliderMultiple);
		contentPane.remove(barLengthLabel);
		contentPane.remove(barLengthSlider);
		contentPane.remove(mappingDataWinPanelScrollPane);
      
		gbc.fill = GridBagConstraints.BOTH;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 0.95;
		gbc.weighty = 0.98;
		addCompItem(mappingDataWinPanelScrollPane, gbc, 0, 0, 1, 1);
	    }
    
	//Now call validate so that these componant changes are displayed.
	validate();
    }
        
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
		ParaProf.systemError(e, null, "MDW09");
	    }
    }
  
    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e)
    {
	closeThisWindow();
    }
  
    void closeThisWindow()
    {
	try{
	    if(ParaProf.debugIsOn)
		{
		    System.out.println("------------------------");
		    System.out.println("A mapping window for: \"" + mappingName + "\" is closing");
		    System.out.println("Clearing resourses for that window.");
		}
	    setVisible(false);
	    trial.getSystemEvents().deleteObserver(this);
	    dispose();
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "MDW10");
	    }
    }

    
  
    //******************************
    //Instance data.
    //******************************
    private int mappingID = -1;
    private String mappingName = null;
  
    private JMenu unitsMenu;
    JMenu valuePercentMenu;
    private JMenuItem mappingGroupLedgerItem;
    private JMenuItem userEventLedgerItem;
  
    private ButtonGroup inclusiveExclusiveGroup = null;
    private ButtonGroup valuePercentGroup = null;
    private ButtonGroup unitsGroup = null;
    private ButtonGroup metricGroup = null;
  
    private JRadioButtonMenuItem metricButton =  new JRadioButtonMenuItem("Selected Metric", true);
  
    private JRadioButtonMenuItem inclusiveRadioButton =  new JRadioButtonMenuItem("Inclusive", false);
    private JRadioButtonMenuItem exclusiveRadioButton = new JRadioButtonMenuItem("Exclusive", true);
    private JRadioButtonMenuItem numOfCallsRadioButton =  new JRadioButtonMenuItem("Number of Calls", false);
    private JRadioButtonMenuItem numOfSubRoutinesRadioButton = new JRadioButtonMenuItem("Number of Subroutines", false);
    private JRadioButtonMenuItem userSecPerCallRadioButton = new JRadioButtonMenuItem("Per Call Value", false);
  
    private JRadioButtonMenuItem valueButton = null;
    private JRadioButtonMenuItem percentButton = null;
  
    private JRadioButtonMenuItem secondsButton = null;
    private JRadioButtonMenuItem millisecondsButton = null;
    private JRadioButtonMenuItem microsecondsButton = null;
  
    private JRadioButtonMenuItem displaySlidersButton;
  
    private JLabel sliderMultipleLabel = new JLabel("Slider Mulitiple");
    private JComboBox sliderMultiple;
  
    private JLabel barLengthLabel = new JLabel("Bar Mulitiple");
    private JSlider barLengthSlider = new JSlider(0, 40, 1);
  
    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;
  
    private JScrollPane mappingDataWinPanelScrollPane;
  
    //private ThreadDataWindowPanel threadDataWindowPanelRef = null;
    int hsbValue = 0;
  
    private Trial trial = null;
    StaticMainWindowData sMWData = null;
  
    Vector sMWGeneralData = null;
    
    MappingDataWindowPanel mappingDataWinPanelRef = null;
    private JScrollBar hsb = new JScrollBar(JScrollBar.HORIZONTAL);
  
    private String metric = "Exclusive";
  
    boolean inclusive = false;
    boolean percent = true;
    private String unitsString = null;
  
    //******************************
    //End - Instance data.
    //******************************
}
