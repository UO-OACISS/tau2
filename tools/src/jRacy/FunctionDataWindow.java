/* 
	FunctionDataWindow.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	The container for the FunctionDataWindowPanel.
*/

package jRacy;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

public class FunctionDataWindow extends JFrame implements ActionListener, Observer, ChangeListener
{
	
	public FunctionDataWindow()
	{
		try{
			setLocation(new java.awt.Point(300, 200));
			setSize(new java.awt.Dimension(550, 550));
			
			//Set the title indicating that there was a problem.
			this.setTitle("Wrong constructor used");
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "FDW01");
		}
	}
	
	public FunctionDataWindow(String inFunctionName, StaticMainWindowData inSMWData)
	{
		try{
			setLocation(new java.awt.Point(300, 200));
			setSize(new java.awt.Dimension(550, 550));
			
			sMWData = inSMWData;
			
			inclusive = false;
	 		percent = true;
	 		unitsString = "milliseconds";
	 		
	 		functionName = inFunctionName;
			
			//Now set the title.
			this.setTitle("Function Data Window: " + jRacy.profilePathName);
			
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
				jRacy.helpWindow.writeText("This is the function data window for:");
				jRacy.helpWindow.writeText(functionName);
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("This window shows you this function's statistics across all the threads.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
				jRacy.helpWindow.writeText("");
				jRacy.helpWindow.writeText("Right click anywhere within this window to bring up a popup");
				jRacy.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
				jRacy.helpWindow.writeText("for this function.");
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
			funDataWinPanelRef = new FunctionDataWindowPanel(inFunctionName, this);
			//The scroll panes into which the list shall be placed.
			JScrollPane funDataWinPanelScrollPane = new JScrollPane(funDataWinPanelRef);
			funDataWinPanelScrollPane.setBorder(mainloweredbev);
			funDataWinPanelScrollPane.setPreferredSize(new Dimension(500, 450));
			//**********
			//End - Panel and ScrollPane definition.
			//**********
			
			String sliderMultipleStrings[] = {"1.00", "0.75", "0.50", "0.25", "0.10"};
			sliderMultiple = new JComboBox(sliderMultipleStrings);
			sliderMultiple.addActionListener(this);
			
			barLengthSlider.setPaintTicks(true);
			barLengthSlider.setMajorTickSpacing(5);
			barLengthSlider.setMinorTickSpacing(1);
			barLengthSlider.setPaintLabels(true);
			barLengthSlider.setSnapToTicks(true);
			barLengthSlider.addChangeListener(this);
			
			
			//Now add the componants to the main screen.
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
			addCompItem(funDataWinPanelScrollPane, gbc, 0, 1, 4, 1);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "FDW02");
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
				else if(arg.equals("Inclusive"))
				{
					if(inclusiveRadioButton.isSelected())
					{
						inclusive = true;
						//Call repaint.
						funDataWinPanelRef.repaint();
					}
				}
				else if(arg.equals("Exclusive"))
				{
					if(exclusiveRadioButton.isSelected())
					{
						inclusive = false;
						//Call repaint.
						funDataWinPanelRef.repaint();
					}
				}
				else if(arg.equals("Percent"))
				{
					if(percentButton.isSelected())
					{
						percent = true;
						//Call repaint.
						funDataWinPanelRef.repaint();
					}
				}
				else if(arg.equals("Value"))
				{
					if(valueButton.isSelected())
					{
						percent = false;
						//Call repaint.
						funDataWinPanelRef.repaint();
					}
				}
				else if(arg.equals("Seconds"))
				{
					if(secondsButton.isSelected())
					{
						unitsString = "Seconds";
						//Call repaint.
						funDataWinPanelRef.repaint();
					}
				}
				else if(arg.equals("Microseconds"))
				{
					if(microsecondsButton.isSelected())
					{
						unitsString = "Microseconds";
						//Call repaint.
						funDataWinPanelRef.repaint();
					}
				}
				else if(arg.equals("Milliseconds"))
				{
					if(millisecondsButton.isSelected())
					{
						unitsString = "Milliseconds";
						//Call repaint.
						funDataWinPanelRef.repaint();
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
					jRacy.helpWindow.writeText("This is the function data window for:");
					jRacy.helpWindow.writeText(functionName);
					jRacy.helpWindow.writeText("");
					jRacy.helpWindow.writeText("This window shows you this function's statistics across all the threads.");
					jRacy.helpWindow.writeText("");
					jRacy.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
					jRacy.helpWindow.writeText("");
					jRacy.helpWindow.writeText("Right click anywhere within this window to bring up a popup");
					jRacy.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
					jRacy.helpWindow.writeText("for this function.");
				}
			}
			else if(EventSrc == sliderMultiple)
			{
				funDataWinPanelRef.changeInMultiples();
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "FDW03");
		}
	}
	//******************************
	//End - ActionListener code.
	//******************************
	
	
	//Change listener code.
	public void stateChanged(ChangeEvent event)
	{
		funDataWinPanelRef.changeInMultiples();
	}
	
	//Observer functions.
	public void update(Observable o, Object arg)
	{
		try{
			String tmpString = (String) arg;
			if(tmpString.equals("prefEvent"))
			{
				//Just need to call a repaint on the ThreadDataWindowPanel.
				funDataWinPanelRef.repaint();
			}
			else if(tmpString.equals("colorEvent"))
			{
				//Just need to call a repaint on the ThreadDataWindowPanel.
				funDataWinPanelRef.repaint();
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
			jRacy.systemError(null, "FDW04");
		}
	}
	
	//FunctionDataWindowPanel call back functions.
	public Vector getStaticMainWindowSystemData()
	{
		
		try{
			if(sMWGeneralData == null)
			{
				//For the function data window, the ordering does not matter.
				sMWGeneralData = sMWData.getSMWGeneralData(null);
				return sMWGeneralData;
			}
			else
			{
				return sMWGeneralData;
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "FDW05");
		}
		
		return null;
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
			jRacy.systemError(null, "FDW06");
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
			jRacy.systemError(null, "FDW07");
		}
		
		return 0;
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
			jRacy.systemError(null, "FDW08");
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
			if(jRacy.debugIsOn)
			{
				System.out.println("------------------------");
				System.out.println("A funtion window for: \"" + functionName + "\" is closing");
				System.out.println("Clearing resourses for that window.");
			}
			setVisible(false);
			jRacy.systemEvents.deleteObserver(this);
			dispose();
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "FDW09");
		}
	}
	
	//******************************
	//Instance data.
	//******************************
	String functionName = null;
	
	ButtonGroup inclusiveExclusiveGroup = null;
	ButtonGroup valuePercentGroup = null;
	ButtonGroup unitsGroup = null;
	
	JRadioButtonMenuItem inclusiveRadioButton = null;
	JRadioButtonMenuItem exclusiveRadioButton = null;
	
	JRadioButtonMenuItem valueButton = null;
	JRadioButtonMenuItem percentButton = null;
	
	JRadioButtonMenuItem secondsButton = null;
	JRadioButtonMenuItem millisecondsButton = null;
	JRadioButtonMenuItem microsecondsButton = null;
	
	private JLabel sliderMultipleLabel = new JLabel("Slider Mulitiple");
	private JComboBox sliderMultiple;
	
	private JLabel barLengthLabel = new JLabel("Bar Mulitiple");
	private JSlider barLengthSlider = new JSlider(0, 40, 1);
	
 	private ThreadDataWindowPanel threadDataWindowPanelRef = null;
 	
 	StaticMainWindowData sMWData = null;
 	
 	Vector sMWGeneralData = null;
 	 	
 	FunctionDataWindowPanel funDataWinPanelRef = null;
 	
 	boolean inclusive = false;
 	boolean percent = false;
 	String unitsString = null;
 	
 	//******************************
	//End - Instance data.
	//******************************
}