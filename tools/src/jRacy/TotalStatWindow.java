/* 
	TotalStatWindow.java

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

public class TotalStatWindow extends JFrame implements ActionListener, MenuListener, Observer 
{
	
	public TotalStatWindow()
	{
		try{
			setLocation(new java.awt.Point(0, 0));
			setSize(new java.awt.Dimension(800, 600));
			
			//Set the title indicating that there was a problem.
			this.setTitle("Wrong constructor used!");
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSW01");
		}
	}
	
	public TotalStatWindow(ExperimentRun inExpRun,
						   int inServerNumber,
						   int inContextNumber,
						   int inThreadNumber,
						   StaticMainWindowData inSMWData)
	{
		try{
			expRun = inExpRun;
			sMWData = inSMWData;
			
			setLocation(new java.awt.Point(0, 0));
			setSize(new java.awt.Dimension(800, 600));
			
			//Now set the title.
			this.setTitle("Total " + "n,c,t, " + inServerNumber + "," + inContextNumber + "," + inThreadNumber + " - " + expRun.getProfilePathName());
			
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
			totalStatWindowPanelRef = new TotalStatWindowPanel(expRun,
															   inServerNumber,
															   inContextNumber,
															   inThreadNumber, this);
			
			//The scroll panes into which the list shall be placed.
			JScrollPane totalStatWindowPanelScrollPane = new JScrollPane(totalStatWindowPanelRef);
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
			jRacy.systemError(null, "TSW02");
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
					expRun.getColorChooser().showColorChooser();	//The ColorChooser class maintains all the state.
				}
				else if(arg.equals("Name"))
				{
					if(nameButton.isSelected())
					{
						metric = "Name";
						//Call repaint.
						totalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Descending"))
				{
					if(descendingButton.isSelected())
					{
						descendingOrder = true;
						totalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Ascending"))
				{
					if(ascendingButton.isSelected())
					{
						descendingOrder = false;
						totalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Inclusive"))
				{
					if(inclusiveRadioButton.isSelected())
					{
						metric = "Inclusive";
						//Call repaint.
						totalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Exclusive"))
				{
					if(exclusiveRadioButton.isSelected())
					{
						metric = "Exclusive";
						//Call repaint.
						totalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Number of Calls"))
				{
					if(numOfCallsRadioButton.isSelected())
					{
						metric = "Number of Calls";
						//Call repaint.
						totalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Number of Subroutines"))
				{
					if(numOfSubRoutinesRadioButton.isSelected())
					{
						metric = "Number of Subroutines";
						//Call repaint.
						totalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Per Call Value"))
				{
					if(userSecPerCallRadioButton.isSelected())
					{
						metric = "Per Call Value";
						totalStatWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Show Function Ledger"))
				{
					//In order to be in this window, I must have loaded the data. So,
					//just show the mapping ledger window.
					(expRun.getGlobalMapping()).displayMappingLedger(0);
				}
				else if(arg.equals("Show Group Ledger"))
				{
					//In order to be in this window, I must have loaded the data. So,
					//just show the mapping ledger window.
					(expRun.getGlobalMapping()).displayMappingLedger(1);
				}
				else if(arg.equals("Show User Event Ledger"))
				{
					//In order to be in this window, I must have loaded the data. So,
					//just show the mapping ledger window.
					(expRun.getGlobalMapping()).displayMappingLedger(2);
				}
				else if(arg.equals("Close All Sub-Windows"))
				{
					//Close the all subwindows.
					expRun.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
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
			jRacy.systemError(null, "TSW03");
		}
	}
	
	//******************************
	//MenuListener code.
	//******************************
	public void menuSelected(MenuEvent evt)
	{
		try
		{
			if(expRun.groupNamesPresent())
				mappingGroupLedgerItem.setEnabled(true);
			else
				mappingGroupLedgerItem.setEnabled(false);
				
			if(expRun.userEventsPresent())
				userEventLedgerItem.setEnabled(true);
			else{
				userEventLedgerItem.setEnabled(false);
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSW03");
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
				totalStatWindowPanelRef.repaint();
			}
			if(tmpString.equals("colorEvent"))
			{
				//Just need to call a repaint on the ThreadDataWindowPanel.
				totalStatWindowPanelRef.repaint();
			}
			else if(tmpString.equals("dataEvent"))
			{ 
				totalStatWindowPanelRef.repaint();
			}
			else if(tmpString.equals("subWindowCloseEvent"))
			{ 
				closeThisWindow();
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSW04");
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
			jRacy.systemError(null, "TSW05");
		}
	}
	
	//This function passes the correct data list to its panel when asked for.
	//Note:  This is only meant to be called by the TotalStatWindowPanel.
	public Vector getStaticMainWindowSystemData()
	{
		try
		{
			if(metric.equals("Name"))
			{
				
				if(descendingOrder)
						return sMWData.getSMWThreadData(server, context, thread, "NDE");
				else
					return sMWData.getSMWThreadData(server, context, thread, "NAE");
			}
			if(metric.equals("Inclusive"))
			{
				if(descendingOrder)
					return sMWData.getSMWThreadData(server, context, thread, "MDI");
				else
					return sMWData.getSMWThreadData(server, context, thread, "MAI");
			}
			else if(metric.equals("Exclusive"))
			{
				if(descendingOrder)
					return sMWData.getSMWThreadData(server, context, thread, "MDE");
				else
					return sMWData.getSMWThreadData(server, context, thread, "MAE");
			}
			else if(metric.equals("Number of Calls"))
			{
				if(descendingOrder)
					return sMWData.getSMWThreadData(server, context, thread, "MDNC");
				else
					return sMWData.getSMWThreadData(server, context, thread, "MANC");
			}
			else if(metric.equals("Number of Subroutines"))
			{
				if(descendingOrder)
					return sMWData.getSMWThreadData(server, context, thread, "MDNS");
				else
					return sMWData.getSMWThreadData(server, context, thread, "MANS");
			}
			else if(metric.equals("Per Call Value"))
			{
				if(descendingOrder)
					return sMWData.getSMWThreadData(server, context, thread, "MDUS");
				else
					return sMWData.getSMWThreadData(server, context, thread, "MAUS");
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSW06");
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
				System.out.println("A total stat window for: \"" + "n,c,t, " + server + "," + context + "," + thread + "\" is closing");
				System.out.println("Clearing resourses for this window.");
			}
			
			setVisible(false);
			expRun.getSystemEvents().deleteObserver(this);
			dispose();
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSW07");
		}
	}
	
	//******************************
	//Instance data.
	//******************************
	private ExperimentRun expRun = null;
 	private TotalStatWindowPanel totalStatWindowPanelRef;
 	private StaticMainWindowData sMWData;
 	
 	ButtonGroup sortGroup;
	ButtonGroup sortOrderGroup;
	private ButtonGroup metricGroup = null;
	
	private JMenuItem mappingGroupLedgerItem;
	private JMenuItem userEventLedgerItem;

	JRadioButtonMenuItem ascendingButton = new JRadioButtonMenuItem("Ascending", false);
	JRadioButtonMenuItem descendingButton = new JRadioButtonMenuItem("Descending", true);
	
	private JRadioButtonMenuItem nameButton = new JRadioButtonMenuItem("Name", false);
	private JRadioButtonMenuItem inclusiveRadioButton =  new JRadioButtonMenuItem("Inclusive", false);
	private JRadioButtonMenuItem exclusiveRadioButton = new JRadioButtonMenuItem("Exclusive", true);
	private JRadioButtonMenuItem numOfCallsRadioButton =  new JRadioButtonMenuItem("Number of Calls", false);
	private JRadioButtonMenuItem numOfSubRoutinesRadioButton = new JRadioButtonMenuItem("Number of Subroutines", false);
	private JRadioButtonMenuItem userSecPerCallRadioButton = new JRadioButtonMenuItem("Per Call Value", false);
	
	private String metric = "Exclusive";
	
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