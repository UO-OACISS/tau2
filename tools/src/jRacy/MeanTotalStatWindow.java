/* 
	MeanTotalStatWindow.java

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

public class MeanTotalStatWindow extends JFrame implements ActionListener, Observer 
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
			jRacy.systemError(null, "MTSW01");
		}
	}
	
	public MeanTotalStatWindow(StaticMainWindowData inSMWData)
	{
		try{
			setLocation(new java.awt.Point(0, 0));
			setSize(new java.awt.Dimension(800, 600));
			
			//Now set the title.
			this.setTitle("Mean Total Stat Window");
			
			
			sMWData = inSMWData;
			
			//Add some window listener code
				addWindowListener(new java.awt.event.WindowAdapter() {
					public void windowClosing(java.awt.event.WindowEvent evt) {
						thisWindowClosing(evt);
					}
				});
				
				
			//Set the help window text if required.
			if(jRacy.helpWindow.isVisible())
			{
				jRacy.helpWindow.clearText();
				//Since the data must have been loaded.  Tell them someting about
				//where they are.
				jRacy.helpWindow.writeText("This is the mean total function statistics window.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("This window shows you the mean total statistics for all functions.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("Right click on any function within this window to bring up a popup");
				jRacy.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
				jRacy.helpWindow.writeText("for the function, or to show more details about the function.");
				jRacy.helpWindow.writeText("You can also left click any function to hightlight it in the system.");
			}
			
			//******************************
			//Code to generate the menus.
			//******************************
			
			//Generic Duke Stuff.
			Icon dukeStandingSmall = new ImageIcon("Duke_Standing_Small.gif");
			Icon dukeWavingSmall = new ImageIcon("Duke_Waving_Small.gif");
			JRadioButton menubarDuke = new JRadioButton(dukeStandingSmall);
			JRadioButton fileMenuDuke = new JRadioButton(dukeStandingSmall);
			JRadioButton optionsMenuDuke = new JRadioButton(dukeStandingSmall);
			JRadioButton windowsMenuDuke = new JRadioButton(dukeStandingSmall);
			JRadioButton helpMenuDuke = new JRadioButton(dukeStandingSmall);
			menubarDuke.setRolloverIcon(dukeWavingSmall);
			fileMenuDuke.setRolloverIcon(dukeWavingSmall);
			optionsMenuDuke.setRolloverIcon(dukeWavingSmall);
			windowsMenuDuke.setRolloverIcon(dukeWavingSmall);
			helpMenuDuke.setRolloverIcon(dukeWavingSmall);
			
			
			JMenuBar mainMenu = new JMenuBar();
			
			//Duke placement.
			mainMenu.add(menubarDuke);
			
			//******************************
			//File menu.
			//******************************
			JMenu fileMenu = new JMenu("File");
			fileMenu.add(fileMenuDuke);
			
			//Add a menu item.
			JMenuItem closeItem = new JMenuItem("Close This Window", dukeWavingSmall);
			closeItem.addActionListener(this);
			fileMenu.add(closeItem);
			
			//Add a menu item.
			JMenuItem exitItem = new JMenuItem("Exit Racy!", dukeWavingSmall);
			exitItem.addActionListener(this);
			fileMenu.add(exitItem);
			//******************************
			//End - File menu.
			//******************************
			
			//******************************
			//Options menu.
			//******************************
			JMenu optionsMenu = new JMenu("Options");
			optionsMenu.add(optionsMenuDuke);
			
			//Add a submenu.
			JMenu sortMenu = new JMenu("Sort by ...");
			sortGroup = new ButtonGroup();
			
			functionIDButton = new JRadioButtonMenuItem("function ID", false);
			//Add a listener for this radio button.
			functionIDButton.addActionListener(this);
			
			nameButton = new JRadioButtonMenuItem("name", false);
			//Add a listener for this radio button.
			nameButton.addActionListener(this);
			
			millisecondButton = new JRadioButtonMenuItem("millisecond", true);
			//Add a listener for this radio button.
			millisecondButton.addActionListener(this);
			
			sortGroup.add(functionIDButton);
			sortGroup.add(nameButton);
			sortGroup.add(millisecondButton);
			
			sortMenu.add(functionIDButton);
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
			JMenu inclusiveExclusiveMenu = new JMenu("Select Inclusive or Exclusive");
			inclusiveExclusiveGroup = new ButtonGroup();
			inclusiveRadioButton = new JRadioButtonMenuItem("Inclusive", false);
			//Add a listener for this radio button.
			inclusiveRadioButton.addActionListener(this);
			exclusiveRadioButton = new JRadioButtonMenuItem("Exclusive", true);
			//Add a listener for this radio button.
			exclusiveRadioButton.addActionListener(this);
			inclusiveExclusiveGroup.add(inclusiveRadioButton);
			inclusiveExclusiveGroup.add(exclusiveRadioButton);
			inclusiveExclusiveMenu.add(inclusiveRadioButton);
			inclusiveExclusiveMenu.add(exclusiveRadioButton);
			optionsMenu.add(inclusiveExclusiveMenu);
			//End Submenu.
					
			
			//Add a menu item.
			JMenuItem colorItem = new JMenuItem("Adjust Racy Colors");
			colorItem.addActionListener(this);
			optionsMenu.add(colorItem);
			//******************************
			//End - Options menu.
			//******************************
			
			
			//******************************
			//Window menu.
			//******************************
			JMenu windowsMenu = new JMenu("Windows");
			windowsMenu.add(windowsMenuDuke);
			
			//Add a submenu.
			JMenuItem functionLedgerItem = new JMenuItem("Show Function Ledger");
			functionLedgerItem.addActionListener(this);
			windowsMenu.add(functionLedgerItem);
			
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
			helpMenu.add(helpMenuDuke);
			
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
			meanTotalStatWindowPanelRef = new MeanTotalStatWindowPanel(this);
			
			//The scroll panes into which the list shall be placed.
			JScrollPane totalStatWindowPanelScrollPane = new JScrollPane(meanTotalStatWindowPanelRef);
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
			jRacy.systemError(null, "MTSW02");
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
				else if(arg.equals("Adjust Racy Colors"))
				{
					jRacy.clrChooser.showColorChooser();	//The ColorChooser class maintains all the state.
				}
				else if(arg.equals("function ID"))
				{
					if(functionIDButton.isSelected())
					{
						sortByFunctionID = true;
						sortByName = false;
						sortByMillisecond = false;
						//Call repaint.
						meanTotalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("name"))
				{
					if(nameButton.isSelected())
					{
						sortByFunctionID = false;
						sortByName = true;
						sortByMillisecond = false;
						//Call repaint.
						meanTotalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("millisecond"))
				{
					if(millisecondButton.isSelected())
					{
						sortByFunctionID = false;
						sortByName = false;
						sortByMillisecond = true;
						//Call repaint.
						meanTotalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Descending"))
				{
					if(descendingButton.isSelected())
					{
						descendingOrder = true;
						meanTotalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Ascending"))
				{
					if(ascendingButton.isSelected())
					{
						descendingOrder = false;
						meanTotalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Inclusive"))
				{
					if(inclusiveRadioButton.isSelected())
					{
						inclusive = true;
						//Call repaint.
						meanTotalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Exclusive"))
				{
					if(exclusiveRadioButton.isSelected())
					{
						inclusive = false;
						//Call repaint.
						meanTotalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Show Function Ledger"))
				{
					//In order to be in this window, I must have loaded the data. So,
					//just show the function ledger window.
					(jRacy.staticSystemData.getGlobalMapping()).displayFunctionLedger();
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
					//Since the data must have been loaded.  Tell them someting about
					//where they are.
					jRacy.helpWindow.writeText("This is the mean total function statistics window.");
					jRacy.helpWindow.writeText("");
					jRacy.helpWindow.writeText("This window shows you the mean total statistics for all functions.");
					jRacy.helpWindow.writeText("");
					jRacy.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
					jRacy.helpWindow.writeText("");
					jRacy.helpWindow.writeText("Right click on any function within this window to bring up a popup");
					jRacy.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
					jRacy.helpWindow.writeText("for the function, or to show more details about the function.");
					jRacy.helpWindow.writeText("You can also left click any function to hightlight it in the system.");
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MTSW03");
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
				meanTotalStatWindowPanelRef.repaint();
			}
			else if(tmpString.equals("colorEvent"))
			{
				//Just need to call a repaint on the ThreadDataWindowPanel.
				meanTotalStatWindowPanelRef.repaint();
			}
			else if(tmpString.equals("subWindowCloseEvent"))
			{
				closeThisWindow();
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MTSW04");
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
			jRacy.systemError(null, "MTSW05");
		}
	}
	
	//This function passes the correct data list to its panel when asked for.
	//Note:  This is only meant to be called by the TotalStatWindowPanel.
	public Vector getStaticMainWindowSystemData()
	{
		try{
			if(sortByFunctionID)
			{
				if(descendingOrder)
				{
					if(!(lastSorting.equals("FIdDE")))
					{
						lastList = sMWData.getSMWMeanData("FIdDE");
						lastSorting = "FIdDE";
						return lastList;
					}
					else
						return lastList;
				}
				else
				{
					if(!(lastSorting.equals("FIdAE")))
					{
						lastList = sMWData.getSMWMeanData("FIdAE");
						lastSorting = "FIdAE";
						return lastList;
					}
					else
						return lastList;
				}
			}
			else if(sortByName)
			{
				if(descendingOrder)
				{
					if(!(lastSorting.equals("NDE")))
					{
						lastList = sMWData.getSMWMeanData("NDE");
						lastSorting = "NDE";
						return lastList;
					}
					else
						return lastList;
				}
				else
				{
					if(!(lastSorting.equals("NAE")))
					{
						lastList = sMWData.getSMWMeanData("NAE");
						lastSorting = "NAE";
						return lastList;
					}
					else
						return lastList;
				}
			}
			else if(sortByMillisecond)
			{
				if(inclusive)
				{
					
					if(descendingOrder)
					{
						if(!(lastSorting.equals("MDI")))
						{
							lastList = sMWData.getSMWMeanData("MDI");
							lastSorting = "MDI";
							return lastList;
						}
						else
							return lastList;
					}
					else
					{
						if(!(lastSorting.equals("MAI")))
						{
							lastList = sMWData.getSMWMeanData("MAI");
							lastSorting = "MAI";
							return lastList;
						}
						else
							return lastList;
					}
				}
				else
				{
					if(descendingOrder)
					{
						if(!(lastSorting.equals("MDE")))
						{
							lastList = sMWData.getSMWMeanData("MDE");
							lastSorting = "MDE";
							return lastList;
						}
						else
							return lastList;
					}
					else
					{
						if(!(lastSorting.equals("MAE")))
						{
							lastList = sMWData.getSMWMeanData("MAE");
							lastSorting = "MAE";
							return lastList;
						}
						else
							return lastList;
					}
				}
			}
		
			//Should not get here.  Return null ... will force and exception.
			return null;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MTSW06");
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
			if(jRacy.debugIsOn)
			{
				System.out.println("------------------------");
				System.out.println("The Mean Total Stat Window is closing");
				System.out.println("Clearing resourses for this window.");
			}
			
			setVisible(false);
			jRacy.systemEvents.deleteObserver(this);
			dispose();
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MTSW07");
		}
	}
	
	//******************************
	//Instance data.
	//******************************
 	private MeanTotalStatWindowPanel meanTotalStatWindowPanelRef;
 	private StaticMainWindowData sMWData = new StaticMainWindowData();
 	
 	ButtonGroup sortGroup;
	ButtonGroup sortOrderGroup;
	ButtonGroup inclusiveExclusiveGroup;
	
	JRadioButtonMenuItem functionIDButton;
	JRadioButtonMenuItem nameButton;
	JRadioButtonMenuItem millisecondButton;
	
	JRadioButtonMenuItem ascendingButton;
	JRadioButtonMenuItem descendingButton;
	
	JRadioButtonMenuItem inclusiveRadioButton;
	JRadioButtonMenuItem exclusiveRadioButton;
	
	boolean sortByFunctionID = false;
	boolean sortByName = false;
	boolean sortByMillisecond = true;
	
	boolean descendingOrder = true;
	boolean inclusive = false;
	
 	
 	String lastSorting = "";
 	Vector lastList;
 	
	
		
	//******************************
	//End - Instance data.
	//******************************

}