/* 
	TotalStatUEWindow.java

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

public class TotalStatUEWindow extends JFrame implements ActionListener, MenuListener, Observer 
{
	
	public TotalStatUEWindow()
	{
		try{
			setLocation(new java.awt.Point(0, 0));
			setSize(new java.awt.Dimension(800, 600));
			
			//Set the title indicating that there was a problem.
			this.setTitle("Wrong constructor used!");
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "TSUEW01");
		}
	}
	
	public TotalStatUEWindow(Trial inTrial,
							int inServerNumber,
						   int inContextNumber,
						   int inThreadNumber,
						   StaticMainWindowData inSMWData)
	{
		try{
			
			trial = inTrial;
			sMWData = inSMWData;
			
			setLocation(new java.awt.Point(0, 0));
			setSize(new java.awt.Dimension(800, 600));
			
			//Now set the title.
			this.setTitle("Total " + "n,c,t, " + inServerNumber + "," + inContextNumber + "," + inThreadNumber + " - " + trial.getProfilePathName());
			
			server = inServerNumber;
			context = inContextNumber;
			thread = inThreadNumber;
			
			FIdDE = null;
		 	FIdAE = null;
		 	NDE = null;
		 	NAE = null;
		 	MDE = null;
		 	MAE = null;
		 	MDI = null;
		 	MAI = null;
			
			
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
				jRacy.helpWindow.writeText("This is the total user event statistics window.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("This window shows you the total statistics for all user events on this thread.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("Right click on any user event within this window to bring up a popup");
				jRacy.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
				jRacy.helpWindow.writeText("for the user event, or to show more details about the user event.");
				jRacy.helpWindow.writeText("You can also left click any user event to hightlight it in the system.");
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
			
			/*
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
					
			*/
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
			totalStatUEWindowPanelRef = new TotalStatUEWindowPanel(trial, inServerNumber,
															   inContextNumber,
															   inThreadNumber, this);
			
			//The scroll panes into which the list shall be placed.
			JScrollPane totalStatWindowPanelScrollPane = new JScrollPane(totalStatUEWindowPanelRef);
			JScrollBar vScollBar = totalStatWindowPanelScrollPane.getVerticalScrollBar();
			vScollBar.setUnitIncrement(35);
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
			jRacy.systemError(e, null, "TSUEW02");
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
					trial.getColorChooser().showColorChooser();	//The ColorChooser class maintains all the state.
				}
				else if(arg.equals("function ID"))
				{
					if(mappingIDButton.isSelected())
					{
						sortByMappingID = true;
						sortByName = false;
						sortByMillisecond = false;
						//Call repaint.
						totalStatUEWindowPanelRef.repaint();
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
						totalStatUEWindowPanelRef.repaint();
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
						totalStatUEWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Descending"))
				{
					if(descendingButton.isSelected())
					{
						descendingOrder = true;
						totalStatUEWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Ascending"))
				{
					if(ascendingButton.isSelected())
					{
						descendingOrder = false;
						totalStatUEWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Inclusive"))
				{
					if(inclusiveRadioButton.isSelected())
					{
						inclusive = true;
						//Call repaint.
						totalStatUEWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Exclusive"))
				{
					if(exclusiveRadioButton.isSelected())
					{
						inclusive = false;
						//Call repaint.
						totalStatUEWindowPanelRef.repaint();
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
					JOptionPane.showMessageDialog(this, jRacy.getInfoString());
				}
				else if(arg.equals("Show Help Window"))
				{
					//Show the racy help window.
					jRacy.helpWindow.clearText();
					jRacy.helpWindow.show();
					//Since the data must have been loaded.  Tell them someting about
					//where they are.
					jRacy.helpWindow.writeText("This is the total function statistics window.");
					jRacy.helpWindow.writeText("");
					jRacy.helpWindow.writeText("This window shows you the total statistics for all functions on this thread.");
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
			jRacy.systemError(e, null, "TSUEW03");
		}
	}
	
	//******************************
	//MenuListener code.
	//******************************
	public void menuSelected(MenuEvent evt)
	{
		try
		{
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
			jRacy.systemError(e, null, "SMW03");
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
				totalStatUEWindowPanelRef.repaint();
			}
			if(tmpString.equals("colorEvent"))
			{
				//Just need to call a repaint on the ThreadDataWindowPanel.
				totalStatUEWindowPanelRef.repaint();
			}
			else if(tmpString.equals("dataEvent"))
			{ 
				totalStatUEWindowPanelRef.repaint();
			}
			else if(tmpString.equals("subWindowCloseEvent"))
			{ 
				closeThisWindow();
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "TSUEW04");
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
			jRacy.systemError(e, null, "TSUEW05");
		}
	}
	
	//This function passes the correct data list to its panel when asked for.
	//Note:  This is only meant to be called by the TotalStatWindowPanel.
	public Vector getStaticMainWindowSystemData()
	{
		/*try{
			if(sortByMappingID)
			{
				if(descendingOrder)
					return sMWData.getSMWThreadData(server, context, thread, "FIdDE");
				else
					return sMWData.getSMWThreadData(server, context, thread, "FIdAE");
			}
			else if(sortByName)
			{
				if(descendingOrder)
					return sMWData.getSMWThreadData(server, context, thread, "NDE");
				else
					return sMWData.getSMWThreadData(server, context, thread, "NAE");
			}
			else if(sortByMillisecond)
			{
				if(inclusive)
				{
					
					if(descendingOrder)
						return sMWData.getSMWThreadData(server, context, thread, "MDI");
					else
						return sMWData.getSMWThreadData(server, context, thread, "MAI");
				}
				else
				{
					if(descendingOrder)
						return sMWData.getSMWThreadData(server, context, thread, "MDE");
					else
						return sMWData.getSMWThreadData(server, context, thread, "MAE");
				}
			}
			
			//Should not get here.  Return null ... will force and exception.
			return null;
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "TSUEW06");
		}*/
		
		return sMWData.getSMWUEThreadData(server, context, thread);
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
				System.out.println("A total stat window for: \"" + "n,c,t, " + server + "," + context + "," + thread + "\" is closing");
				System.out.println("Clearing resourses for this window.");
			}
			
			setVisible(false);
			trial.getSystemEvents().deleteObserver(this);
			dispose();
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "TSUEW07");
		}
	}
	
	//******************************
	//Instance data.
	//******************************
	private Trial trial = null;
 	private TotalStatUEWindowPanel totalStatUEWindowPanelRef;
 	private StaticMainWindowData sMWData = new StaticMainWindowData(trial);
 	
 	private JMenuItem mappingGroupLedgerItem;
	private JMenuItem userEventLedgerItem;
 	
 	ButtonGroup sortGroup;
	ButtonGroup sortOrderGroup;
	ButtonGroup inclusiveExclusiveGroup;
	
	JRadioButtonMenuItem mappingIDButton;
	JRadioButtonMenuItem nameButton;
	JRadioButtonMenuItem millisecondButton;
	
	JRadioButtonMenuItem ascendingButton;
	JRadioButtonMenuItem descendingButton;
	
	JRadioButtonMenuItem inclusiveRadioButton;
	JRadioButtonMenuItem exclusiveRadioButton;
	
	boolean sortByMappingID = false;
	boolean sortByName = false;
	boolean sortByMillisecond = true;
	
	boolean descendingOrder = true;
	boolean inclusive = false;
	
	int server;
 	int context;
 	int thread;
 	
 	Vector FIdDE;
 	Vector FIdAE;
 	Vector NDE;
 	Vector NAE;
 	Vector MDE;
 	Vector MAE;
	Vector MDI;
	Vector MAI;
		
	//******************************
	//End - Instance data.
	//******************************

}