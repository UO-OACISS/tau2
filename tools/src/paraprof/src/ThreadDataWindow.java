/* 
  ThreadDataWindow.java

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
import java.awt.print.*;

public class ThreadDataWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener
{
  
  public ThreadDataWindow()
  {
    try
    {
      setLocation(new java.awt.Point(300, 200));
      setSize(new java.awt.Dimension(700, 450));
      
      //Set the title indicating that there was a problem.
      this.setTitle("Wrong constructor used!");
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "TDW01");
    }
  }
  
  public ThreadDataWindow(Trial inTrial,
              int inServerNumber,
              int inContextNumber,
              int inThreadNumber,
              StaticMainWindowData inSMWData)
  {
    try
    {
      
      trial = inTrial;
      sMWData = inSMWData;
      
      lPWindow = new LocalPrefWindow(trial, this);
      
      setLocation(new java.awt.Point(300, 200));
      setSize(new java.awt.Dimension(700, 450));
      
      server = inServerNumber;
      context = inContextNumber;
      thread = inThreadNumber;
      
      //Now set the title.
      this.setTitle("n,c,t, " + server + "," + context + "," + thread + " - " + trial.getProfilePathName());
      
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
        ParaProf.helpWindow.writeText("This is the thread data window");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("This window shows you the values for all functions on this thread.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
        ParaProf.helpWindow.writeText("");
        ParaProf.helpWindow.writeText("Right click on any function within this window to bring up a popup");
        ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
        ParaProf.helpWindow.writeText("for the function, or to show more details about the function.");
        ParaProf.helpWindow.writeText("You can also left click any function to hightlight it in the system.");
      }
      
      //Sort the local data.
      sortLocalData();
      
      //******************************
      //Code to generate the menus.
      //******************************
      
      
      JMenuBar mainMenu = new JMenuBar();
      
      //******************************
      //File menu.
      //******************************
      JMenu fileMenu = new JMenu("File");
      
      /*//Add a menu item.
      JMenuItem printItem = new JMenuItem("Print");
      printItem.addActionListener(this);
      fileMenu.add(printItem);*/
      
      //Add a menu item.
      JMenuItem editPrefItem = new JMenuItem("Edit ParaProf Preferences!");
      editPrefItem.addActionListener(this);
      fileMenu.add(editPrefItem);
      
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
      
      //***********
      //Submenu.
      JMenu sortMenu = new JMenu("Sort by ...");
      
      sortGroup = new ButtonGroup();
      //Add listeners
      nameButton.addActionListener(this);
      millisecondButton.addActionListener(this);
      sortGroup.add(nameButton);
      sortGroup.add(millisecondButton);
      sortMenu.add(nameButton);
      sortMenu.add(millisecondButton);
      
      
      sortOrderGroup = new ButtonGroup();
      //Add listeners
      ascendingButton.addActionListener(this);
      descendingButton.addActionListener(this);
      sortOrderGroup.add(ascendingButton);
      sortOrderGroup.add(descendingButton);
      sortMenu.add(ascendingButton);
      sortMenu.add(descendingButton);
      
      sortMenu.insertSeparator(2);
      
      optionsMenu.add(sortMenu);
      //End - Submenu.
      //***********
      
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
      
      //***********
      //Submenu.
      valuePercentMenu = new JMenu("Select Value or Percent");
      valuePercentGroup = new ButtonGroup();
      
      //Add listeners
      percentButton.addActionListener(this);
      valueButton.addActionListener(this);
      
      valuePercentGroup.add(percentButton);
      valuePercentGroup.add(valueButton);
      
      valuePercentMenu.add(percentButton);
      valuePercentMenu.add(valueButton);
      optionsMenu.add(valuePercentMenu);
      //End - Submenu.
      //***********
      
      //***********
      //Submenu.
      unitsMenu = new JMenu("Select Units");
      unitsGroup = new ButtonGroup();
      
      //Add listeners
      secondsButton.addActionListener(this);
      millisecondsButton.addActionListener(this);
      microsecondsButton.addActionListener(this);
      
      unitsGroup.add(secondsButton);
      unitsGroup.add(millisecondsButton);
      unitsGroup.add(microsecondsButton);
      
      unitsMenu.add(secondsButton);
      unitsMenu.add(millisecondsButton);
      unitsMenu.add(microsecondsButton);
      optionsMenu.add(unitsMenu);
      //End - Submenu.
      //***********
      
      displaySlidersButton = new JRadioButtonMenuItem("Display Sliders", false);
      //Add a listener for this radio button.
      displaySlidersButton.addActionListener(this);
      optionsMenu.add(displaySlidersButton);
      
      formatValuesButton = new JRadioButtonMenuItem("Format Values", true);
      //Add a listener for this radio button.
      formatValuesButton.addActionListener(this);
      optionsMenu.add(formatValuesButton);
      
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
      
      JMenuItem aboutItem = new JMenuItem("About Racy");
      
      JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
      
      //Add listeners
      aboutItem.addActionListener(this);
      showHelpWindowItem.addActionListener(this);
      
      
      helpMenu.add(aboutItem);
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
      threadDataWindowPanelRef = new ThreadDataWindowPanel(trial,
                                 inServerNumber,
                                 inContextNumber,
                                 inThreadNumber, this, sMWData);
      
      //**********
      //End - Panel and ScrollPane definition.
      //**********
      
      //The scroll panes into which the list shall be placed.
      threadDataWindowPanelScrollPane = new JScrollPane(threadDataWindowPanelRef);
      threadDataWindowPanelScrollPane.setBorder(mainloweredbev);
      threadDataWindowPanelScrollPane.setPreferredSize(new Dimension(500, 450));
      
      
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
      addCompItem(threadDataWindowPanelScrollPane, gbc, 0, 0, 1, 1);
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "TDW02");
    }
    
    
  }
  
  //******************************
  //Event listener code!!
  //******************************
  
  //******************************
  //ActionListener code.
  //******************************
  public void actionPerformed(ActionEvent evt)
  {
    try
    {
      Object EventSrc = evt.getSource();
      
      if(EventSrc instanceof JMenuItem)
      {
        String arg = evt.getActionCommand();
        if(arg.equals("Print"))
        {
          PrinterJob job = PrinterJob.getPrinterJob();
          PageFormat defaultFormat = job.defaultPage();
          PageFormat selectedFormat = job.pageDialog(defaultFormat);
          job.setPrintable(threadDataWindowPanelRef, selectedFormat);
          if(job.printDialog()){
            job.print();
          }
        }
        else if(arg.equals("Edit ParaProf Preferences!"))
        {
          lPWindow.show();
        }
        else if(arg.equals("Close This Window"))
        {
          closeThisWindow();
        }
        else if(arg.equals("Exit Racy!"))
        {
          setVisible(false);
          dispose();
          System.exit(0);
        }
        else if(arg.equals("name"))
        {
          if(nameButton.isSelected())
          {
            sortByMappingID = false;
            sortByName = true;
            sortByMillisecond = false;
            //Sort the local data.
            sortLocalData();
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Selected Metric"))  //Note the difference in case from the millisecond option below.
        {
          if(millisecondButton.isSelected())
          {
            sortByMappingID = false;
            sortByName = false;
            sortByMillisecond = true;
            //Sort the local data.
            sortLocalData();
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Descending"))
        {
          if(descendingButton.isSelected())
          {
            descendingOrder = true;
            //Sort the local data.
            sortLocalData();
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Ascending"))
        {
          if(ascendingButton.isSelected())
          {
            descendingOrder = false;
            //Sort the local data.
            sortLocalData();
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Inclusive"))
        {
          if(inclusiveRadioButton.isSelected())
          {
            metric = "Inclusive";
            //Sort the local data.
            sortLocalData();
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Exclusive"))
        {
          if(exclusiveRadioButton.isSelected())
          {
            metric = "Exclusive";
            //Sort the local data.
            sortLocalData();
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Number of Calls"))
        {
          if(numOfCallsRadioButton.isSelected())
          {
            metric = "Number of Calls";
            //Sort the local data.
            sortLocalData();
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Number of Subroutines"))
        {
          if(numOfSubRoutinesRadioButton.isSelected())
          {
            metric = "Number of Subroutines";
            //Sort the local data.
            sortLocalData();
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Per Call Value"))
        {
          if(userSecPerCallRadioButton.isSelected())
          {
            metric = "Per Call Value";
            //Sort the local data.
            sortLocalData();
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Percent"))
        {
          if(percentButton.isSelected())
          {
            percent = true;
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Value"))
        {
          if(valueButton.isSelected())
          {
            percent = false;
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Seconds"))
        {
          if(secondsButton.isSelected())
          {
            unitsString = "Seconds";
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Microseconds"))
        {
          if(microsecondsButton.isSelected())
          {
            unitsString = "Microseconds";
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
        }
        else if(arg.equals("Milliseconds"))
        {
          if(millisecondsButton.isSelected())
          {
            unitsString = "Milliseconds";
            //Call repaint.
            threadDataWindowPanelRef.repaint();
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
        else if(arg.equals("Format Values"))
        {
          if(formatValuesButton.isSelected())
          { 
            formatNumbers = true;
            //Call repaint.
            threadDataWindowPanelRef.repaint();
          }
          else
          {
            formatNumbers = false;
            //Call repaint.
            threadDataWindowPanelRef.repaint();
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
          ParaProf.helpWindow.writeText("This is the thread data window");
          ParaProf.helpWindow.writeText("");
          ParaProf.helpWindow.writeText("This window shows you the values for all mappings on this thread.");
          ParaProf.helpWindow.writeText("");
          ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
          ParaProf.helpWindow.writeText("");
          ParaProf.helpWindow.writeText("Right click on any mapping within this window to bring up a popup");
          ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
          ParaProf.helpWindow.writeText("for the mapping, or to show more details about the mapping.");
          ParaProf.helpWindow.writeText("You can also left click any mapping to hightlight it in the system.");
        }
      }
      else if(EventSrc == sliderMultiple)
      {
        threadDataWindowPanelRef.changeInMultiples();
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "TDW03");
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
    threadDataWindowPanelRef.changeInMultiples();
  }
  //******************************
  //End - Change listener code.
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
      else if(!(isDefault || isTimeMetric)){
        valuePercentMenu.setEnabled(true);
        unitsMenu.setEnabled(false);
      }
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
    try
    {
      String tmpString = (String) arg;
      if(tmpString.equals("prefEvent"))
      {
        //Just need to call a repaint on the ThreadDataWindowPanel.
        threadDataWindowPanelRef.repaint();
      }
      else if(tmpString.equals("colorEvent"))
      {
        //Just need to call a repaint on the ThreadDataWindowPanel.
        threadDataWindowPanelRef.repaint();
      }
      else if(tmpString.equals("dataEvent"))
      {
        sortLocalData();
        threadDataWindowPanelRef.repaint();
      }
      else if(tmpString.equals("subWindowCloseEvent"))
      {
        closeThisWindow();
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "TDW05");
    }
  }
  
  public void refreshPanel(){
    threadDataWindowPanelRef.repaint();
  }
  
  //Updates the sorted lists after a change of sorting method takes place.
  private void sortLocalData()
  { 
    try
    {
      if(sortByMappingID)
      {
        if(metric.equals("Inclusive"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdDI");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdAI");
        }
        else if(metric.equals("Exclusive"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdDE");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdAE");
        }
        else if(metric.equals("Number of Calls"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdDNC");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdANC");
        }
        else if(metric.equals("Number of Subroutines"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdDNS");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdANS");
        }
        else if(metric.equals("Per Call Value"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdDUS");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "FIdAUS");
        }
      }
      else if(sortByName)
      {
        
        if(metric.equals("Inclusive"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NDI");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NAI");
        }
        else if(metric.equals("Exclusive"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NDE");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NAE");
        }
        else if(metric.equals("Number of Calls"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NDNC");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NANC");
        }
        else if(metric.equals("Number of Subroutines"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NDNS");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NANS");
        }
        else if(metric.equals("Per Call Value"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NDUS");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "NAUS");
        }
      }
      else if(sortByMillisecond)
      {
        
        if(metric.equals("Inclusive"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MDI");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MAI");
        }
        else if(metric.equals("Exclusive"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MDE");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MAE");
        }
        else if(metric.equals("Number of Calls"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MDNC");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MANC");
        }
        else if(metric.equals("Number of Subroutines"))
        {
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MDNS");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MANS");
        }
        else if(metric.equals("Per Call Value"))
        {
          System.out.println("Here1");
          if(descendingOrder)
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MDUS");
          else
            currentSMWThreadData = sMWData.getSMWThreadData(server, context, thread, "MAUS");
        }
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "TDW06");
    }
  }
  
  //This function passes the correct data list to its panel when asked for.
  //Note:  This is only meant to be called by the ThreadDataWindowPanel.
  public Vector getStaticMainWindowSystemData()
  {
    return currentSMWThreadData;
  }
  
  public LocalPrefWindow getLocalPrefWindow(){
    return lPWindow;
  }
  
  public boolean isInclusive()
  {
    if(metric.equals("Inclusive"))
      return true;
    else
      return false;
  }
  
  public String getMetric(){
    return metric;
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
      ParaProf.systemError(e, null, "TDW07");
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
      ParaProf.systemError(e, null, "FDW08");
    }
    
    return 0;
  }
  
  private void displaySiders(boolean displaySliders)
  {
    if(displaySliders)
    {
      //Since the menu option is a toggle, the only component that needs to be
      //removed is that scrollPane.  We then add back in with new parameters.
      //This might not be required as it seems to adjust well if left in, but just
      //to be sure.
      contentPane.remove(threadDataWindowPanelScrollPane);
      
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
      addCompItem(threadDataWindowPanelScrollPane, gbc, 0, 1, 4, 1);
    }
    else
    {
      contentPane.remove(sliderMultipleLabel);
      contentPane.remove(sliderMultiple);
      contentPane.remove(barLengthLabel);
      contentPane.remove(barLengthSlider);
      contentPane.remove(threadDataWindowPanelScrollPane);
      
      gbc.fill = GridBagConstraints.BOTH;
      gbc.anchor = GridBagConstraints.CENTER;
      gbc.weightx = 0.95;
      gbc.weighty = 0.98;
      addCompItem(threadDataWindowPanelScrollPane, gbc, 0, 0, 1, 1);
    }
    
    //Now call validate so that these componant changes are displayed.
    validate();
  }
  
  public boolean getFormatNumbers(){
    return formatNumbers;}
  
  //Helper functions.
  private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h)
  {
    try
    {
      gbc.gridx = x;
      gbc.gridy = y;
      gbc.gridwidth = w;
      gbc.gridheight = h;
      
      getContentPane().add(c, gbc);
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "TDW09");
    }
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
        System.out.println("A thread window for: \"" + "n,c,t, " + server + "," + context + "," + thread + "\" is closing");
        System.out.println("Clearing resourses for that window.");
      }
      
      setVisible(false);
      trial.getSystemEvents().deleteObserver(this);
      dispose();
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "TDW10");
    }
  }
  
  //******************************
  //Instance data.
  //******************************
  private JMenu unitsMenu;
  private JMenu valuePercentMenu;
  private JMenuItem mappingGroupLedgerItem;
  private JMenuItem userEventLedgerItem;
  
  private ButtonGroup sortGroup = null;
  private ButtonGroup sortOrderGroup = null;
  private ButtonGroup metricGroup = null;
  private ButtonGroup valuePercentGroup = null;
  private ButtonGroup unitsGroup = null;
  
  private JRadioButtonMenuItem ascendingButton =  new JRadioButtonMenuItem("Ascending", false);
  private JRadioButtonMenuItem descendingButton = new JRadioButtonMenuItem("Descending", true);
  
  private JRadioButtonMenuItem nameButton = new JRadioButtonMenuItem("name", false);
  private JRadioButtonMenuItem millisecondButton =  new JRadioButtonMenuItem("Selected Metric", true);
  
  private JRadioButtonMenuItem inclusiveRadioButton =  new JRadioButtonMenuItem("Inclusive", false);
  private JRadioButtonMenuItem exclusiveRadioButton = new JRadioButtonMenuItem("Exclusive", true);
  private JRadioButtonMenuItem numOfCallsRadioButton =  new JRadioButtonMenuItem("Number of Calls", false);
  private JRadioButtonMenuItem numOfSubRoutinesRadioButton = new JRadioButtonMenuItem("Number of Subroutines", false);
  private JRadioButtonMenuItem userSecPerCallRadioButton = new JRadioButtonMenuItem("Per Call Value", false);
  
  private JRadioButtonMenuItem valueButton = new JRadioButtonMenuItem("Value", false);
  private JRadioButtonMenuItem percentButton = new JRadioButtonMenuItem("Percent", true);
  
  private JRadioButtonMenuItem secondsButton = new JRadioButtonMenuItem("Seconds", false);
  private JRadioButtonMenuItem millisecondsButton = new JRadioButtonMenuItem("Milliseconds", false);
  private JRadioButtonMenuItem microsecondsButton = new JRadioButtonMenuItem("Microseconds", true);
  
  private JRadioButtonMenuItem displaySlidersButton;
  private JRadioButtonMenuItem formatValuesButton;
  
  private JLabel sliderMultipleLabel = new JLabel("Slider Mulitiple");
  private JComboBox sliderMultiple;
  
  private JLabel barLengthLabel = new JLabel("Bar Mulitiple");
  private JSlider barLengthSlider = new JSlider(0, 40, 1);
  
  private Container contentPane = null;
  private GridBagLayout gbl = null;
  private GridBagConstraints gbc = null;
  
  private JScrollPane threadDataWindowPanelScrollPane;
  
  private ThreadDataWindowPanel threadDataWindowPanelRef = null;
  private LocalPrefWindow lPWindow = null;
  
  private Trial trial = null;
  private StaticMainWindowData sMWData = null;
  
  SMWThreadDataElement tmpSMWThreadDataElement = null;
  
  
  //Local data.
  Vector currentSMWThreadData = null;
  
  private boolean sortByMappingID = false;
  private boolean sortByName = false;
  private boolean sortByMillisecond = true;
  
  private boolean descendingOrder = true;
  
  private String metric = "Exclusive";
  private boolean percent = true;
  private String unitsString = "milliseconds";
  
  private boolean formatNumbers = true;
  
  private int server = -1;
  private int context = -1;
  private int thread = -1;
  
  //******************************
  //End - Instance data.
  //******************************


}
