/* 
	MappingLedgerWindow.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

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
			jRacy.systemError(null, "FLW01");
		}
	}
	
	
	public MappingLedgerWindow(Vector inNameIDMapping, int inMappingSelection)
	{
		try{
			setLocation(new java.awt.Point(300, 200));
			setSize(new java.awt.Dimension(350, 450));
			
			//Now set the title.
			this.setTitle("Mapping Ledger Window: " + jRacy.profilePathName);
			
			mappingSelection = inMappingSelection;
			
			//Set the help window text if required.
			if(jRacy.helpWindow.isVisible())
			{
				jRacy.helpWindow.clearText();
				
				if(!((jRacy.staticMainWindow.getSMWData()).isDataLoaded()))
					{
						if(mappingSelection == 0){
							jRacy.helpWindow.writeText("This is the mapping ledger window.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window shows all the mappings tracked in this profile.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window will be blank until you load data.");
							jRacy.helpWindow.writeText("You can do this from the file menu of the main window.");
						}
						else{
							jRacy.helpWindow.writeText("This is the group mapping ledger window.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window shows all the group mappings tracked in this profile.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window will be blank until you load data.");
							jRacy.helpWindow.writeText("You can do this from the file menu of the main window.");
						}
					}
					else
					{
						if(mappingSelection == 0){
							jRacy.helpWindow.writeText("This is the mapping ledger window.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window shows all the mappings tracked in this profile.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("To see more information about any of the mappings shown here,");
							jRacy.helpWindow.writeText("right click on that mapping, and select from the popup menu.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("You can also left click any mapping to hightlight it in the system.");
						}
						else{
							jRacy.helpWindow.writeText("This is the group mapping ledger window.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window shows all the group mappings tracked in this profile.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("Left click any mapping to hightlight it in the system.");
							jRacy.helpWindow.writeText("Right click on any mapping, and select from the popup menu"
							+ " to display more options for masking or displaying mappings in a particular group.");
						}
					}
			}
			
			//******************************
			//Add a window listener.
			//******************************
			this.addWindowListener(new WindowAdapter()
				{
					public void windowClosing(WindowEvent evt)
					{
						(jRacy.staticSystemData.getGlobalMapping()).closeMappingLedger(mappingSelection);
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
				JMenuItem savejRacyPreferencesItem = new JMenuItem("jRacy Preferrences");
				savejRacyPreferencesItem.addActionListener(this);
				saveMenu.add(savejRacyPreferencesItem);
				
			fileMenu.add(saveMenu);
			//End submenu.
			
			
			//Add a menu item.
			JMenuItem editPrefItem = new JMenuItem("Edit jRacy Preferences");
			editPrefItem.addActionListener(this);
			fileMenu.add(editPrefItem);
			
			//Add a menu item.
			JMenuItem closeItem = new JMenuItem("Close Mapping Ledger Window");
			closeItem.addActionListener(this);
			fileMenu.add(closeItem);
			
			
			//Add a menu item.
			JMenuItem exitItem = new JMenuItem("Exit jRacy!");
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
			mappingLedgerWinPanelRef = new MappingLedgerWindowPanel(inNameIDMapping, mappingSelection);
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
			jRacy.systemError(null, "FLW02");
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
			jRacy.systemError(null, "FLW03");
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
				
				if(arg.equals("Edit jRacy Preferences"))
				{
					jRacy.jRacyPreferences.showPreferencesWindow();
				}
				else if(arg.equals("Exit jRacy!"))
				{
					setVisible(false);
					dispose();
					System.exit(0);
				}	
				else if(arg.equals("Close Mapping Ledger Window"))
				{
					(jRacy.staticSystemData.getGlobalMapping()).closeMappingLedger(mappingSelection);
				}
				else if(arg.equals("Close All Sub-Windows"))
				{
					//Close the all subwindows.
					jRacy.systemEvents.updateRegisteredObjects("subWindowCloseEvent");
				}
				else if(arg.equals("About Racy"))
				{
					JOptionPane.showMessageDialog(this, jRacy.getInfoString());
				}
				else if(arg.equals("Show Help Window"))
				{
					//Show the racy help window.
					jRacy.helpWindow.clearText();
					jRacy.helpWindow.show();
					//See if any system data has been loaded.  Give a helpful hint
					//if it has not.
					if(!((jRacy.staticMainWindow.getSMWData()).isDataLoaded()))
					{
						if(mappingSelection == 0){
							jRacy.helpWindow.writeText("This is the mapping ledger window.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window shows all the mappings tracked in this profile.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window will be blank until you load data.");
							jRacy.helpWindow.writeText("You can do this from the file menu of the main window.");
						}
						else{
							jRacy.helpWindow.writeText("This is the group mapping ledger window.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window shows all the group mappings tracked in this profile.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window will be blank until you load data.");
							jRacy.helpWindow.writeText("You can do this from the file menu of the main window.");
						}
					}
					else
					{
						if(mappingSelection == 0){
							jRacy.helpWindow.writeText("This is the mapping ledger window.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window shows all the mappings tracked in this profile.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("To see more information about any of the mappings shown here,");
							jRacy.helpWindow.writeText("right click on that mapping, and select from the popup menu.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("You can also left click any mapping to hightlight it in the system.");
						}
						else{
							jRacy.helpWindow.writeText("This is the group mapping ledger window.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("This window shows all the group mappings tracked in this profile.");
							jRacy.helpWindow.writeText("");
							jRacy.helpWindow.writeText("Left click any mapping to hightlight it in the system.");
							jRacy.helpWindow.writeText("Right click on any mapping, and select from the popup menu"
							+ " to display more options for masking or displaying mappings in a particular group.");
						}
					}
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "FLW04");
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
				System.out.println("The mappingSelection is: " + mappingSelection);
				jRacy.staticSystemData.getGlobalMapping().closeMappingLedger(mappingSelection);
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "FLW05");
		}
	}
	
	//******************************
	//End - Event listener code!!
	//******************************
	
 	Vector serverDataVector;
 	
 	int mappingSelection = -1;
 	
 	MappingLedgerWindowPanel mappingLedgerWinPanelRef;

}
