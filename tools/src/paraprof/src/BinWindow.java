/*  
  BinWindow.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import javax.swing.colorchooser.*;

public class BinWindow extends JFrame implements ActionListener, MenuListener, Observer  
{ 
  //******************************
  //Instance data.
  //******************************
  
  Trial trial = null;
  
  //References for some of the componants for this frame.
  private BinWindowPanel bWPanel;
  private StaticMainWindowData sMWData = null;
  private int mappingID = -1;
  private Vector sMWGeneralData = null;
  
  private Container contentPane = null;
  private GridBagLayout gbl = null;
  private GridBagConstraints gbc = null;
  
  private JScrollPane scrollPane;
  
  private JMenuItem mappingGroupLedgerItem;
  private JMenuItem userEventLedgerItem;
  
  //Local data
  private Vector currentSMWGeneralData = null;
  private Vector currentSMWMeanData = null;
  
  
  //******************************
  //End - Instance data.
  //******************************
  
  
  public BinWindow(Trial inTrial, StaticMainWindowData inSMWData, boolean normal, int inMappingID)
  {
    try{
      trial = inTrial;
      mappingID = inMappingID;
      sMWData = inSMWData;
      
      //Window Stuff.
      setTitle("ParaProf: " + trial.getProfilePathName());
      int windowWidth = 750;
      int windowHeight = 500;
      setSize(new java.awt.Dimension(windowWidth, windowHeight));
      
      //Add some window listener code
      addWindowListener(new java.awt.event.WindowAdapter() {
        public void windowClosing(java.awt.event.WindowEvent evt) {
          thisWindowClosing(evt);
        }
      });
      
      //Grab the screen size.
      Toolkit tk = Toolkit.getDefaultToolkit();
      Dimension screenDimension = tk.getScreenSize();
      int screenHeight = screenDimension.height;
      int screenWidth = screenDimension.width;
      
      //Set the window to come up in the center of the screen.
      int xPosition = (screenWidth - windowWidth) / 2;
      int yPosition = (screenHeight - windowHeight) / 2;
      
      setLocation(xPosition, yPosition);
      
      //End - Window Stuff.

      //******************************
      //Code to generate the menus.
      //******************************
      
      
      JMenuBar mainMenu = new JMenuBar();
      
      //******************************
      //File menu.
      //******************************
      JMenu fileMenu = new JMenu("File");
      
      
      //Add a submenu.
      JMenu openMenu = new JMenu("Open ...");
        //Add a menu item.
        JMenuItem openExperimentManagerItem = new JMenuItem("ParaProf Manager");
        openExperimentManagerItem.addActionListener(this);
        openMenu.add(openExperimentManagerItem);
        
      fileMenu.add(openMenu);
      //End submenu.
      
      //Add a menu item.
      JMenuItem editPrefItem = new JMenuItem("Edit ParaProf Preferences!");
      editPrefItem.addActionListener(this);
      fileMenu.add(editPrefItem);
      
      
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
      JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
      showHelpWindowItem.addActionListener(this);
      helpMenu.add(showHelpWindowItem);
      
      //Add a menu item.
      JMenuItem aboutItem = new JMenuItem("About Racy");
      aboutItem.addActionListener(this);
      helpMenu.add(aboutItem);
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
      bWPanel = new BinWindowPanel(trial, this, normal, inMappingID);
      bWPanel.setPreferredSize(new Dimension(600,300));
      //The scroll panes into which the list shall be placed.
      scrollPane = new JScrollPane(bWPanel);
      scrollPane.setBorder(mainloweredbev);
      scrollPane.setPreferredSize(new Dimension(600, 300));
      
      //**********
      //End - Panel and ScrollPane definition.
      //**********
      
      
      gbc.fill = GridBagConstraints.BOTH;
      gbc.anchor = GridBagConstraints.CENTER;
      gbc.weightx = 1;
      gbc.weighty = 1;
      addCompItem(scrollPane, gbc, 0, 0, 1, 1);
      
      //******************************
      //End - Create and add the componants.
      //******************************
      
      //Call a repaint of the bWPanel
      bWPanel.repaint();
    }
    catch(Exception e)
    { 
      ParaProf.systemError(e, null, "BW01");
    }
      
  }
  
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
        if(arg.equals("ParaProf Manager"))
        {
          ParaProfManager jRM = new ParaProfManager();
          jRM.show();
        }
        else if(arg.equals("Edit ParaProf Preferences!"))
        {
          trial.getPreferences().showPreferencesWindow();
        } 
        else if(arg.equals("Exit ParaProf!"))
        {
          setVisible(false);
          dispose();
          System.exit(0);
        }
        else if(arg.equals("Show Function Ledger"))
        {
          //Grab the global mapping and bring up the mapping ledger window.
          (trial.getGlobalMapping()).displayMappingLedger(0);
        }
        else if(arg.equals("Show Group Ledger"))
        {
          //Grab the global mapping and bring up the mapping ledger window.
          (trial.getGlobalMapping()).displayMappingLedger(1);
        }
        else if(arg.equals("Show User Event Ledger"))
        {
          //Grab the global mapping and bring up the mapping ledger window.
          (trial.getGlobalMapping()).displayMappingLedger(2);
        }
        else if(arg.equals("Close All Sub-Windows"))
        {
          //Close the all subwindows.
          trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
        }
        else if(arg.equals("About ParaProf"))
        {
          JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
        }
        else if(arg.equals("Show Help Window"))
        {
          //Show the racy help window.
          ParaProf.helpWindow.show();
          //See if any system data has been loaded.  Give a helpful hint
          //if it has not.
          if(!(sMWData.isDataLoaded()))
          {
            ParaProf.helpWindow.writeText("Welcome to ParaProf");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("The first step is to load a pprof dump file."
                                      + "You can find this option in the file menu.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("To create a pprof dump file, simply run pprof" +
                          " with the -d option, and pipe the output to a file.");
          }
        }
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "BW02");
    }
  }
  
  //******************************
  //MenuListener code.
  //******************************
  public void menuSelected(MenuEvent evt){
    try{
      if(trial.groupNamesPresent())
        mappingGroupLedgerItem.setEnabled(true);
      else
        mappingGroupLedgerItem.setEnabled(false);
        
      if(trial.userEventsPresent())
        userEventLedgerItem.setEnabled(true);
      else
        userEventLedgerItem.setEnabled(false);
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "BW03");
    }
    
  }
  public void menuDeselected(MenuEvent evt){}
  public void menuCanceled(MenuEvent evt){} 
  //******************************
  //End - MenuListener code.
  //******************************
    
  //Observer functions.
  public void update(Observable o, Object arg)
  {
    try{
      String tmpString = (String) arg;
      if(tmpString.equals("prefEvent")){
        //Just need to call a repaint on the ThreadDataWindowPanel.
        bWPanel.repaint();
      }
      else if(tmpString.equals("colorEvent")){
        //Just need to call a repaint on the ThreadDataWindowPanel.
        bWPanel.repaint();
      }
      else if(tmpString.equals("dataEvent")){
        bWPanel.repaint();
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "BW04");
    }
  }
    
  private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
    try{
      gbc.gridx = x;
      gbc.gridy = y;
      gbc.gridwidth = w;
      gbc.gridheight = h;
      
      contentPane.add(c, gbc);
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "BW05");
    }
  }
  
  //******************************
  //End - Event listener code!!
  //******************************
  
  public Vector getStaticMainWindowSystemData()
  {
    try{
      if(sMWGeneralData == null){
        sMWGeneralData = sMWData.getSMWMappingData(mappingID);
        return sMWGeneralData;
      }
      else{
        return sMWGeneralData;
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MDW06");
    }
    
    return null;
  }
  
  private boolean mShown = false;
  public void addNotify() 
  {
    super.addNotify();
    
    if (mShown)
      return;
      
    // resize frame to account for menubar
    JMenuBar jMenuBar = getJMenuBar();
    if (jMenuBar != null) {
      int jMenuBarHeight = jMenuBar.getPreferredSize().height;
      Dimension dimension = getSize();
      dimension.height += jMenuBarHeight;
      setSize(dimension);
    }

    mShown = true;
  }

  // Close the window when the close box is clicked
  void thisWindowClosing(java.awt.event.WindowEvent e){
    setVisible(false);
    dispose();
  } 
}
