/* 
	ThreadDataWindow.java

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

public class MeanDataWindow extends JFrame implements ActionListener, Observer
{
	
	public MeanDataWindow()
	{
		try{
			setLocation(new java.awt.Point(300, 200));
			setSize(new java.awt.Dimension(700, 450));
			
			//Set the title indicating that there was a problem.
			this.setTitle("Wrong constructor used!");
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MDW01");
		}
	}
	
	public MeanDataWindow(StaticMainWindowData inSMWData)
	{
		try{
			setLocation(new java.awt.Point(300, 200));
			setSize(new java.awt.Dimension(700, 450));
			
			sMWData = inSMWData;
			
			currentSMWMeanData = null;
			
			inclusive = false;
	 		percent = true;
	 		unitsString = "milliseconds";
			
			//Now set the title.
			this.setTitle("Mean Data Window");
			
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
				jRacy.helpWindow.writeText("This is the mean data window");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("This window shows you the mean values for all functions.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("Right click on any function within this window to bring up a popup");
				jRacy.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
				jRacy.helpWindow.writeText("for the function, or to show more details about the function.");
				jRacy.helpWindow.writeText("You can also left click any function to hightlight it in the system.");
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
			
			//Add a submenu.
			JMenu valuePercentMenu = new JMenu("Select Value or Percent");
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
			JMenu unitsMenu = new JMenu("Select Units");
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
			
			//******************************
			//End - Options menu.
			//******************************
			
			
			//******************************
			//Window menu.
			//******************************
			JMenu windowsMenu = new JMenu("Windows");
			
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
			meanDataWindowPanelRef = new MeanDataWindowPanel(this, sMWData);
			
			//**********
			//End - Panel and ScrollPane definition.
			//**********
			
			//The scroll panes into which the list shall be placed.
			JScrollPane meanDataWindowPanelScrollPane = new JScrollPane(meanDataWindowPanelRef);
			meanDataWindowPanelScrollPane.setBorder(mainloweredbev);
			meanDataWindowPanelScrollPane.setPreferredSize(new Dimension(500, 450));
			
			barLengthSlider.setPaintTicks(true);
			barLengthSlider.setMajorTickSpacing(5);
			barLengthSlider.setMinorTickSpacing(1);
			barLengthSlider.setPaintLabels(true);
			barLengthSlider.setSnapToTicks(true);
			barLengthSlider.addChangeListener(meanDataWindowPanelRef);
			
			
			//Now add the componants to the main screen.
			gbc.fill = GridBagConstraints.NONE;
			gbc.anchor = GridBagConstraints.EAST;
			gbc.weightx = 0.475;
			gbc.weighty = 0.01;
			addCompItem(barLengthLabel, gbc, 0, 0, 1, 1);
			
			gbc.fill = GridBagConstraints.HORIZONTAL;
			gbc.anchor = GridBagConstraints.WEST;
			gbc.weightx = 0.475;
			gbc.weighty = 0.01;
			addCompItem(barLengthSlider, gbc, 1, 0, 1, 1);
			
			gbc.fill = GridBagConstraints.BOTH;
			gbc.anchor = GridBagConstraints.CENTER;
			gbc.weightx = 0.95;
			gbc.weighty = 0.98;
			addCompItem(meanDataWindowPanelScrollPane, gbc, 0, 1, 2, 1);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MDW02");
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
				else if(arg.equals("function ID"))
				{
					if(functionIDButton.isSelected())
					{
						sortByFunctionID = true;
						sortByName = false;
						sortByMillisecond = false;
						sortLocalData();
						//Call repaint.
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("name"))
				{
					if(nameButton.isSelected())
					{
						sortByFunctionID = false;
						sortByName = true;
						sortByMillisecond = false;
						sortLocalData();
						//Call repaint.
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("millisecond"))	//Note the difference in case from the millisecond option below.
				{
					if(millisecondButton.isSelected())
					{
						sortByFunctionID = false;
						sortByName = false;
						sortByMillisecond = true;
						sortLocalData();
						//Call repaint.
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Descending"))
				{
					if(descendingButton.isSelected())
					{
						descendingOrder = true;
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Ascending"))
				{
					if(ascendingButton.isSelected())
					{
						descendingOrder = false;
						sortLocalData();
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Inclusive"))
				{
					if(inclusiveRadioButton.isSelected())
					{
						inclusive = true;
						sortLocalData();
						//Call repaint.
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Exclusive"))
				{
					if(exclusiveRadioButton.isSelected())
					{
						inclusive = false;
						sortLocalData();
						//Call repaint.
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Percent"))
				{
					if(percentButton.isSelected())
					{
						percent = true;
						sortLocalData();
						//Call repaint.
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Value"))
				{
					if(valueButton.isSelected())
					{
						percent = false;
						//Call repaint.
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Seconds"))
				{
					if(secondsButton.isSelected())
					{
						unitsString = "Seconds";
						//Call repaint.
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Microseconds"))
				{
					if(microsecondsButton.isSelected())
					{
						unitsString = "Microseconds";
						//Call repaint.
						meanDataWindowPanelRef.repaint();
					}
				}
				else if(arg.equals("Milliseconds"))
				{
					if(millisecondsButton.isSelected())
					{
						unitsString = "Milliseconds";
						//Call repaint.
						meanDataWindowPanelRef.repaint();
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
					jRacy.helpWindow.writeText("This is the mean data window");
					jRacy.helpWindow.writeText("");
					jRacy.helpWindow.writeText("This window shows you the mean values for all functions.");
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
			jRacy.systemError(null, "MDW03");
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
				meanDataWindowPanelRef.repaint();
			}
			else if(tmpString.equals("colorEvent"))
			{
				//Just need to call a repaint on the ThreadDataWindowPanel.
				meanDataWindowPanelRef.repaint();
			}
			else if(tmpString.equals("subWindowCloseEvent"))
			{
				closeThisWindow();
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MDW04");
		}
	}
	
	//Updates the sorted lists after a change of sorting method takes place.
	private void sortLocalData()
	{
		try
		{
			if(sortByFunctionID)
			{
				if(inclusive)
				{
					if(descendingOrder)
						currentSMWMeanData = sMWData.getSMWMeanData("FIdDI");
					else
						currentSMWMeanData = sMWData.getSMWMeanData("FIdAI");
				}
				else
				{
					if(descendingOrder)
						currentSMWMeanData = sMWData.getSMWMeanData("FIdDE");
					else
						currentSMWMeanData = sMWData.getSMWMeanData("FIdAE");
				}
			}
			else if(sortByName)
			{
				
				if(inclusive)
				{
					if(descendingOrder)
						currentSMWMeanData = sMWData.getSMWMeanData("NDI");
					else
						currentSMWMeanData = sMWData.getSMWMeanData("NAI");
				}
				else
				{
					if(descendingOrder)
						currentSMWMeanData = sMWData.getSMWMeanData("NDE");
					else
						currentSMWMeanData = sMWData.getSMWMeanData("NAE");
				}
			}
			else if(sortByMillisecond)
			{
				if(inclusive)
				{
					if(descendingOrder)
						currentSMWMeanData = sMWData.getSMWMeanData("MDI");
					else
						currentSMWMeanData = sMWData.getSMWMeanData("MAI");
				}
				else
				{
					if(descendingOrder)
						currentSMWMeanData = sMWData.getSMWMeanData("MDE");
					else
						currentSMWMeanData = sMWData.getSMWMeanData("MAE");
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MDW05");
		}
	}
	
	//This function passes the correct data list to its panel when asked for.
	//Note:  This is only meant to be called by the MeanDataWindowPanel.
	public Vector getStaticMainWindowSystemData()
	{
		return currentSMWMeanData;
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
		return barLengthSlider.getValue();
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
			jRacy.systemError(null, "MDW06");
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
			if(jRacy.debugIsOn)
			{
				System.out.println("------------------------");
				System.out.println("The Mean Data Window is closing");
				System.out.println("Clearing resourses for this window.");
			}
			
			setVisible(false);
			jRacy.systemEvents.deleteObserver(this);
			dispose();
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MDW07");
		}
	}
	
	//******************************
	//Instance data.
	//******************************
	
	private ButtonGroup sortGroup;
	private ButtonGroup sortOrderGroup;
	private ButtonGroup inclusiveExclusiveGroup;
	private ButtonGroup valuePercentGroup;
	private ButtonGroup unitsGroup;
	
	private JRadioButtonMenuItem ascendingButton;
	private JRadioButtonMenuItem descendingButton;
	
	private JRadioButtonMenuItem functionIDButton;
	private JRadioButtonMenuItem nameButton;
	private JRadioButtonMenuItem millisecondButton;
	
	private JRadioButtonMenuItem inclusiveRadioButton;
	private JRadioButtonMenuItem exclusiveRadioButton;
	
	private JRadioButtonMenuItem valueButton;
	private JRadioButtonMenuItem percentButton;
	
	private JRadioButtonMenuItem secondsButton;
	private JRadioButtonMenuItem millisecondsButton;
	private JRadioButtonMenuItem microsecondsButton;
	
	private JRadioButtonMenuItem showZeroFunctionsItem;
	
	private JLabel barLengthLabel = new JLabel("Bar Mulitiple");
	
	private JSlider barLengthSlider = new JSlider(0, 40, 1);
	
 	private MeanDataWindowPanel meanDataWindowPanelRef;
 	
 	private StaticMainWindowData sMWData;
 	
 	//Local data.
 	Vector currentSMWMeanData = null;
 	
 	private boolean sortByFunctionID = false;
	private boolean sortByName = false;
	private boolean sortByMillisecond = true;
	
	private boolean descendingOrder = true;
 	
 	private boolean inclusive;
 	private boolean percent;
 	private String unitsString;
 	//******************************
	//End - Instance data.
	//******************************


}
