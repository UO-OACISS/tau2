/* 
  MappingLedgerWindow.java

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

public class MappingLedgerWindow extends JFrame implements ActionListener,  Observer
{
  
  
  public MappingLedgerWindow()
  {
    try{
      setLocation(new java.awt.Point(300, 200));
      setSize(new java.awt.Dimension(350, 450));
      
      //Set the title indicating that there was a problem.
      this.setTitle("Wrong constructor used");
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MLW01");
    }
  }
  
  
  public MappingLedgerWindow(Trial inTrial, Vector inNameIDMapping, int inMappingSelection)
  {
    try{
      
      trial = inTrial;
      mappingSelection = inMappingSelection;
      
      setLocation(new java.awt.Point(300, 200));
      setSize(new java.awt.Dimension(350, 450));
      
      //Now set the title.
      if(mappingSelection == 0)
        this.setTitle("Function Ledger Window: " + trial.getProfilePathName());
      else if(mappingSelection == 1)
        this.setTitle("Group Ledger Window: " + trial.getProfilePathName());
      else
        this.setTitle("User Event Window: " + trial.getProfilePathName());
      
      //Set the help window text if required.
      if(ParaProf.helpWindow.isVisible())
      {
        ParaProf.helpWindow.clearText();
 
	if(mappingSelection == 0){
	    ParaProf.helpWindow.writeText("This is the function ledger window.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("This window shows all the functions tracked in this profile.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("To see more information about any of the mappings shown here,");
	    ParaProf.helpWindow.writeText("right click on that function, and select from the popup menu.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("You can also left click any function to hightlight it in the system.");
	}
	else if(mappingSelection == 1){
	    ParaProf.helpWindow.writeText("This is the group ledger window.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("This window shows all the groups tracked in this profile.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("Left click any group to hightlight it in the system.");
	    ParaProf.helpWindow.writeText("Right click on any group, and select from the popup menu"
					  + " to display more options for masking or displaying functions in a particular group.");
	}
	else{
	    ParaProf.helpWindow.writeText("This is the user event ledger window.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("This window shows all the user events tracked in this profile.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("Left click any mapping to hightlight it in the system.");
	    ParaProf.helpWindow.writeText("Right click on any user event, and select from the popup menu.");
	}
      }
      
      //******************************
      //Add a window listener.
      //******************************
      this.addWindowListener(new WindowAdapter()
        {
          public void windowClosing(WindowEvent evt)
          {
            (trial.getGlobalMapping()).closeMappingLedger(mappingSelection);
          }
        }
        );
      //******************************
      //End - Add a window listener.
      //******************************
      
      
      //******************************
      //Code to generate the menus.
      //******************************
      
      
      JMenuBar mainMenu = new JMenuBar();
      
      //******************************
      //File menu.
      //******************************
      JMenu fileMenu = new JMenu("File");
      
      //Add a submenu.
      JMenu saveMenu = new JMenu("Save ...");
        
        //Add a menu item.
        JMenuItem saveParaProfPreferencesItem = new JMenuItem("ParaProf Preferrences");
        saveParaProfPreferencesItem.addActionListener(this);
        saveMenu.add(saveParaProfPreferencesItem);
        
      fileMenu.add(saveMenu);
      //End submenu.
      
      
      //Add a menu item.
      JMenuItem editPrefItem = new JMenuItem("Edit ParaProf Preferences");
      editPrefItem.addActionListener(this);
      fileMenu.add(editPrefItem);
      
      //Add a menu item.
      JMenuItem closeItem = null;
      if(mappingSelection == 0)
        closeItem = new JMenuItem("Close Function Ledger Window");
      else if(mappingSelection == 1)
        closeItem = new JMenuItem("Close Group Ledger Window");
      else
        closeItem = new JMenuItem("Close User Event Window");
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
      //Window menu.
      //******************************
      JMenu windowsMenu = new JMenu("Windows");
      
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
      mappingLedgerWinPanelRef = new MappingLedgerWindowPanel(trial, inNameIDMapping, mappingSelection);
      //The scroll panes into which the list shall be placed.
      JScrollPane mappingLedgerWinPanelScrollPane = new JScrollPane(mappingLedgerWinPanelRef);
      mappingLedgerWinPanelScrollPane.setBorder(mainloweredbev);
      mappingLedgerWinPanelScrollPane.setPreferredSize(new Dimension(350, 400));
      //**********
      //End - Panel and ScrollPane definition.
      //**********
      
      //Now add the componants to the main screen.
      gbc.fill = GridBagConstraints.BOTH;
      gbc.anchor = GridBagConstraints.CENTER;
      gbc.weightx = 1;
      gbc.weighty = 1;
      addCompItem(mappingLedgerWinPanelScrollPane, gbc, 0, 0, 1, 1);
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MLW02");
    }
  }
  
  //Helper mappings.
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
      ParaProf.systemError(e, null, "MLW03");
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
        
        if(arg.equals("Edit ParaProf Preferences"))
        {
          trial.getPreferences().showPreferencesWindow();
        }
        else if(arg.equals("Exit ParaProf!"))
        {
          setVisible(false);
          dispose();
          System.exit(0);
        } 
        else if(arg.equals("Close Function Ledger Window"))
        {
          (trial.getGlobalMapping()).closeMappingLedger(mappingSelection);
        }
        else if(arg.equals("Close Group Ledger Window"))
        {
          (trial.getGlobalMapping()).closeMappingLedger(mappingSelection);
        }
        else if(arg.equals("Close User Event Window"))
        {
          (trial.getGlobalMapping()).closeMappingLedger(mappingSelection);
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
          //Show the ParaProf help window.
          ParaProf.helpWindow.clearText();
          ParaProf.helpWindow.show();

	  if(mappingSelection == 0){
              ParaProf.helpWindow.writeText("This is the function ledger window.");
              ParaProf.helpWindow.writeText("");
              ParaProf.helpWindow.writeText("This window shows all the functions tracked in this profile.");
              ParaProf.helpWindow.writeText("");
              ParaProf.helpWindow.writeText("To see more information about any of the functions shown here,");
              ParaProf.helpWindow.writeText("right click on that function, and select from the popup menu.");
              ParaProf.helpWindow.writeText("");
              ParaProf.helpWindow.writeText("You can also left click any function to hightlight it in the system.");
	  }
	  else if(mappingSelection == 1){
              ParaProf.helpWindow.writeText("This is the group ledger window.");
              ParaProf.helpWindow.writeText("");
              ParaProf.helpWindow.writeText("This window shows all the groups tracked in this profile.");
              ParaProf.helpWindow.writeText("");
              ParaProf.helpWindow.writeText("Left click any group to hightlight it in the system.");
              ParaProf.helpWindow.writeText("Right click on any group, and select from the popup menu"
              + " to display more options for masking or displaying functions in a particular group.");
	  }
            else{
		ParaProf.helpWindow.writeText("This is the user event ledger window.");
		ParaProf.helpWindow.writeText("");
		ParaProf.helpWindow.writeText("This window shows all the user events tracked in this profile.");
		ParaProf.helpWindow.writeText("");
		ParaProf.helpWindow.writeText("Left click any user event to hightlight it in the system.");
		ParaProf.helpWindow.writeText("Right click on any user event, and select from the popup menu.");
            }
	}
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MLW04");
    }
  }
  
  //Observer functions.
  public void update(Observable o, Object arg)
  {
    try{
      String tmpString = (String) arg;
      if(tmpString.equals("prefEvent"))
      {
        //Just need to call a repaint on the ThreadDataWindowPanel.
        mappingLedgerWinPanelRef.repaint();
      }
      else if(tmpString.equals("colorEvent"))
      {
        //Just need to call a repaint on the ThreadDataWindowPanel.
        mappingLedgerWinPanelRef.repaint();
      }
      else if(tmpString.equals("subWindowCloseEvent"))
      {
        trial.getGlobalMapping().closeMappingLedger(mappingSelection);
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MLW05");
    }
  }
  
  //******************************
  //End - Event listener code!!
  //******************************
  
  Vector serverDataVector;
  
  int mappingSelection = -1;
  
  private Trial trial = null;
  MappingLedgerWindowPanel mappingLedgerWinPanelRef;

}
