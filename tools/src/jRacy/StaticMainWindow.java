/*	
	StaticMainWindow.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import javax.swing.colorchooser.*;

public class StaticMainWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener  
{	
	//******************************
	//Instance data.
	//******************************
	
	//Create a file chooser to allow the user to select files for loading data.
	JFileChooser fileChooser = new JFileChooser();
		
	//References for some of the componants for this frame.
	private StaticMainWindowPanel sMWPanel;
	private StaticMainWindowData sMWData = new StaticMainWindowData();
	
	private ButtonGroup sortGroup;
	private ButtonGroup sortOrderGroup;
	
	private JRadioButtonMenuItem mappingIDButton;
	private JRadioButtonMenuItem nameButton;
	private JRadioButtonMenuItem millisecondButton;
	
	private JRadioButtonMenuItem ascendingButton;
	private JRadioButtonMenuItem descendingButton;
	
	private JRadioButtonMenuItem displaySlidersButton;
	
	private JMenuItem mappingGroupLedgerItem;
	
	private JLabel sliderMultipleLabel = new JLabel("Slider Mulitiple");
	private JComboBox sliderMultiple;
	
	private JLabel barLengthLabel = new JLabel("Bar Mulitiple");
	private JSlider barLengthSlider = new JSlider(0, 40, 1);
	
	private Container contentPane = null;
	private GridBagLayout gbl = null;
	private GridBagConstraints gbc = null;
	
	private JScrollPane scrollPane;
	
	boolean sortByMappingID = false;
	boolean sortByName = false;
	boolean sortByMillisecond = true;
	
	boolean descendingOrder = true;
	
	boolean displaySliders = false;
	
	
	//Local data
 	private Vector currentSMWGeneralData = null;
 	private Vector currentSMWMeanData = null;
	
	
	//******************************
	//End - Instance data.
	//******************************
	
	
	public StaticMainWindow()
	{
		try{
			//Window Stuff.
			setTitle("jRacy: No Data Loaded");
			
			int windowWidth = 750;
			int windowHeight = 400;
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
				JMenuItem openPprofDumpFileItem = new JMenuItem("Pprof Dump File");
				openPprofDumpFileItem.addActionListener(this);
				openMenu.add(openPprofDumpFileItem);
				
				
				//Add a menu item.
				JMenuItem openjRacyOutputItem = new JMenuItem("jRacy Output File");
				openjRacyOutputItem.addActionListener(this);
				openMenu.add(openjRacyOutputItem);
				
			fileMenu.add(openMenu);
			//End submenu.
			
			//Add a submenu.
			JMenu saveMenu = new JMenu("Save ...");
				//Add a menu item.
				JMenuItem savejRacyDataFileFileItem = new JMenuItem("To A jRacy Output File");
				savejRacyDataFileFileItem.addActionListener(this);
				saveMenu.add(savejRacyDataFileFileItem);
				
				
				//Add a menu item.
				JMenuItem savejRacyPreferencesItem = new JMenuItem("jRacy Preferrences");
				savejRacyPreferencesItem.addActionListener(this);
				saveMenu.add(savejRacyPreferencesItem);
				
			fileMenu.add(saveMenu);
			//End submenu.
			
			
			//Add a menu item.
			JMenuItem editPrefItem = new JMenuItem("Edit jRacy Preferences!");
			editPrefItem.addActionListener(this);
			fileMenu.add(editPrefItem);
			
			
			//Add a menu item.
			JMenuItem exitItem = new JMenuItem("Exit jRacy!");
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
			
			mappingIDButton = new JRadioButtonMenuItem("mapping ID", false);
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
			JMenuItem mappingLedgerItem = new JMenuItem("Show Mapping Ledger");
			mappingLedgerItem.addActionListener(this);
			windowsMenu.add(mappingLedgerItem);
			
			//Add a submenu.
			mappingGroupLedgerItem = new JMenuItem("Show Group Mapping Ledger");
			mappingGroupLedgerItem.addActionListener(this);
			windowsMenu.add(mappingGroupLedgerItem);
			
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
			sMWPanel = new StaticMainWindowPanel(this);
			sMWPanel.setPreferredSize(new Dimension(600,300));
			//The scroll panes into which the list shall be placed.
			scrollPane = new JScrollPane(sMWPanel);
			scrollPane.setBorder(mainloweredbev);
			scrollPane.setPreferredSize(new Dimension(600, 300));
			
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
			gbc.weightx = 1;
			gbc.weighty = 1;
			addCompItem(scrollPane, gbc, 0, 0, 1, 1);
			
			//******************************
			//End - Create and add the componants.
			//******************************
			sMWPanel.repaint();
			
			//Ok, now check to see if a "pprof.dat" file exists.  If it does, load it.
			File testForPprofDat = new File("pprof.dat");
			if(testForPprofDat.exists())
			{
				System.out.println("Found pprof.dat ... loading");
				
				jRacy.profilePathName = testForPprofDat.getCanonicalPath();
				
				setTitle("jRacy: " + jRacy.profilePathName);
				
				//Initialize the static data object.
				jRacy.staticSystemData = new StaticSystemData();
				jRacy.staticSystemData.buildStaticData(testForPprofDat);
				
				//Build the static main window data lists.
				sMWData.buildStaticMainWindowDataLists();
				
				//Sort the local data.
				sortLocalData();
				
				//Call a repaint of the sMWPanel
				sMWPanel.repaint();
			}
			else
			{
				System.out.println("Did not find pprof.dat!");
			}
		}
		catch(Exception e)
		{
			
			jRacy.systemError(null, "SMW01");
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
				if(arg.equals("Pprof Dump File"))
				{
					//Create a file chooser to allow the user to select the pprof dump file.
					JFileChooser pprofDumpFileChooser = new JFileChooser();
					
					//Set the directory to the current directory.
					pprofDumpFileChooser.setCurrentDirectory(new File("."));
					
					//Bring up the file chooser.
					int resultValue = pprofDumpFileChooser.showOpenDialog(this);
					
					if(resultValue == JFileChooser.APPROVE_OPTION)
					{
						//Try and get the file name.
						File file = pprofDumpFileChooser.getSelectedFile();
						
						//Test to see if valid.
						if(file != null)
						{	
							
							//To avoid some null pointer stuff, for the moment, explicitly close the ledger windows.
							//Might want to re-think how the mapping ledger window is maintained.  Separate its data
							//for example!
							if(jRacy.debugIsOn)
								System.out.println("Closing the mapping ledger windows.");
							(jRacy.staticSystemData.getGlobalMapping()).closeMappingLedger(0);
							(jRacy.staticSystemData.getGlobalMapping()).closeMappingLedger(1);
							if(jRacy.debugIsOn)
								System.out.println("End - Closing the mapping ledger windows.");
							
							//Closing all subwindows.
							System.out.println("Closing all subwindows.");
							jRacy.systemEvents.updateRegisteredObjects("subWindowCloseEvent");
							
							
							jRacy.profilePathName = file.getCanonicalPath();
				
							setTitle("jRacy: " + jRacy.profilePathName);
							
							//Initialize the static data object.
							jRacy.staticSystemData = new StaticSystemData();
							sMWData = new StaticMainWindowData();
							
							//Call the garbage collector.
							if(jRacy.debugIsOn)
								System.out.println("Cleaning the virtual machine memory footprint.");
							jRacy.runtime.gc();
							
							System.out.println("Building internal data system ... please wait ...");
							
							jRacy.staticSystemData.buildStaticData(file);
							
							System.out.println("Finished building internal data system.");
							
							//Build the static main window data lists.
							System.out.println("Building internal lists ... please wait ...");
							sMWData.buildStaticMainWindowDataLists();
							System.out.println("Finished building internal lists.");
							
							//Reset the hightlight colour.
							jRacy.clrChooser.setHighlightColorMappingID(-1);
							
							//Indicate to the rest of the system that there has been a change of data.
							jRacy.systemEvents.updateRegisteredObjects("dataSetChangeEvent");
							
							System.out.println("Done ... loading complete!");
							
							//Call a repaint of the sMWPanel
							sMWPanel.repaint();
							
						}
						else
						{
							System.out.println("There was some sort of internal error!");
						}
					}
					
				}
				else if(arg.equals("jRacy Output File"))
				{	
					//Set the directory to the current directory.
					fileChooser.setCurrentDirectory(new File("."));
					
					//Bring up the open file chooser.
					int resultValue = fileChooser.showOpenDialog(this);
					
					if(resultValue == JFileChooser.APPROVE_OPTION)
					{
						//Get the file.
						File file = fileChooser.getSelectedFile();
						
						
						//Check to make sure that something was obtained.
						if(file != null)
						{
							
							jRacy.profilePathName = file.getCanonicalPath();
				
							setTitle("jRacy: " + jRacy.profilePathName);
							
							//Closing all subwindows.
							System.out.println("Closing all subwindows.");
							jRacy.systemEvents.updateRegisteredObjects("subWindowCloseEvent");
							
							
							//Get the data object to which this file refers.
							try
							{
								ObjectInputStream racyDataObjectIn = new ObjectInputStream(new FileInputStream(file));
								jRacy.staticSystemData = (StaticSystemData) racyDataObjectIn.readObject();
							}
							catch(Exception e)
							{
								jRacy.systemError(null, "SMW02A");
							}
							
							sMWData = new StaticMainWindowData();
							
							//Call the garbage collector.
							System.out.println("Cleaning the virtual machine memory footprint.");
							jRacy.runtime.gc();
							
							//Build the static main window data lists.
							System.out.println("Building internal lists ... please wait ...");
							sMWData.buildStaticMainWindowDataLists();
							System.out.println("Finished building internal lists.");
							
							//Reset the hightlight colour.
							jRacy.clrChooser.setHighlightColorMappingID(-1);
							
							//Indicate to the rest of the system that there has been a change of data.
							jRacy.systemEvents.updateRegisteredObjects("dataSetChangeEvent");
							
							System.out.println("Done ... loading complete!");
							
							//Call a repaint of the sMWPanel
							sMWPanel.repaint();
						}
						else
						{
							//Display an error
							JOptionPane.showMessageDialog(this, "No file was selected!", "Error!"
															  	,JOptionPane.ERROR_MESSAGE);
						}
					}
				}
				else if(arg.equals("To A jRacy Output File"))
				{
					
					
					//Set the directory to the current directory.
					fileChooser.setCurrentDirectory(new File("."));
					
					//Bring up the save file chooser.
					int resultValue = fileChooser.showSaveDialog(this);
					
					if(resultValue == JFileChooser.APPROVE_OPTION)
					{
						//Get the file.
						File file = fileChooser.getSelectedFile();
						
						
						//Check to make sure that something was obtained.
						if(file != null)
						{
							try
							{
								ObjectOutputStream experimentOut = new ObjectOutputStream(new FileOutputStream(file));
								experimentOut.writeObject(jRacy.staticSystemData);
								experimentOut.close();
							
							}
							catch(Exception e)
							{
								//Display an error
								JOptionPane.showMessageDialog(this, "An error occured whilst trying to save jRacy output file.", "Error!"
															  	,JOptionPane.ERROR_MESSAGE);
							}
						}
						else
						{
							//Display an error
							JOptionPane.showMessageDialog(this, "No filename was given!", "Error!"
															  	,JOptionPane.ERROR_MESSAGE);
						}
					}
				}
				else if(arg.equals("jRacy Preferrences"))
				{
					
					
					//Set the directory to the current directory.
					fileChooser.setCurrentDirectory(new File("."));
					fileChooser.setSelectedFile(new File("jRacyPreferences.dat"));
					
					//Bring up the save file chooser.
					int resultValue = fileChooser.showSaveDialog(this);
					
					if(resultValue == JFileChooser.APPROVE_OPTION)
					{
						//Get the file.
						File file = fileChooser.getSelectedFile();
						
						
						//Check to make sure that something was obtained.
						if(file != null)
						{
							try
							{
								//Write to the savedPreferences object.
								jRacy.clrChooser.setSavedColors();
								jRacy.jRacyPreferences.setSavedPreferences();
								
								ObjectOutputStream prefsOut = new ObjectOutputStream(new FileOutputStream(file));
								prefsOut.writeObject(jRacy.savedPreferences);
								prefsOut.close();
							
							}
							catch(Exception e)
							{
								//Display an error
								JOptionPane.showMessageDialog(this, "An error occured whilst trying to save jRacy preferences.", "Error!"
															  	,JOptionPane.ERROR_MESSAGE);
							}
						}
						else
						{
							//Display an error
							JOptionPane.showMessageDialog(this, "No filename was given!", "Error!"
															  	,JOptionPane.ERROR_MESSAGE);
						}
					}
				}
				else if(arg.equals("Edit jRacy Preferences!"))
				{
					jRacy.jRacyPreferences.showPreferencesWindow();
				}
					
				else if(arg.equals("Exit jRacy!"))
				{
					setVisible(false);
					dispose();
					System.exit(0);
				}
				else if(arg.equals("mapping ID"))
				{
					if(mappingIDButton.isSelected())
					{
						sortByMappingID = true;
						sortByName = false;
						sortByMillisecond = false;
						//Sort the local data.
						sortLocalData();
						//Call repaint.
						sMWPanel.repaint();
					}
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
						sMWPanel.repaint();
					}
				}
				else if(arg.equals("millisecond"))
				{
					if(millisecondButton.isSelected())
					{
						sortByMappingID = false;
						sortByName = false;
						sortByMillisecond = true;
						//Sort the local data.
						sortLocalData();
						//Call repaint.
						sMWPanel.repaint();
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
						sMWPanel.repaint();
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
						sMWPanel.repaint();
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
				else if(arg.equals("Show Mapping Ledger"))
				{
					
					//Check to make sure that the system data file has been loaded!
					if(sMWData.isDataLoaded())
					{
						//Grab the global mapping and bring up the mapping ledger window.
						(jRacy.staticSystemData.getGlobalMapping()).displayMappingLedger(0);
					}
					else
					{
						//Pop up an error!
						JOptionPane.showMessageDialog(this, "Sorry, but you must load a pprof data file first!", "Selection Error!"
																  ,JOptionPane.ERROR_MESSAGE);
					}
				}
				else if(arg.equals("Show Group Mapping Ledger"))
				{
					//Check to make sure that the system data file has been loaded!
					if(sMWData.isDataLoaded())
					{
						//Grab the global mapping and bring up the mapping ledger window.
						(jRacy.staticSystemData.getGlobalMapping()).displayMappingLedger(1);
					}
					else
					{
						//Pop up an error!
						JOptionPane.showMessageDialog(this, "Sorry, but you must load a pprof data file first!", "Selection Error!"
																  ,JOptionPane.ERROR_MESSAGE);
					}
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
					jRacy.helpWindow.show();
					//See if any system data has been loaded.  Give a helpful hint
					//if it has not.
					if(!(sMWData.isDataLoaded()))
					{
						jRacy.helpWindow.writeText("Welcome to jRacy");
						jRacy.helpWindow.writeText("");
						jRacy.helpWindow.writeText("The first step is to load a pprof dump file."
																			+ "You can find this option in the file menu.");
						jRacy.helpWindow.writeText("");
						jRacy.helpWindow.writeText("To create a pprof dump file, simply run pprof" +
												  " with the -d option, and pipe the output to a file.");
					}
				}
			}
			else if(EventSrc == sliderMultiple)
			{
				sMWPanel.changeInMultiples();
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SMW02");
		}
	}
	
	//******************************
	//MenuListener code.
	//******************************
	public void menuSelected(MenuEvent evt)
	{
		try
		{
			if(jRacy.staticSystemData.groupNamesPresent())
				mappingGroupLedgerItem.setEnabled(true);
			else
				mappingGroupLedgerItem.setEnabled(false);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SMW03");
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
	
	//******************************
	//Change listener code.
	//******************************
	public void stateChanged(ChangeEvent event)
	{
		sMWPanel.changeInMultiples();
	}
	//******************************
	//End - Change listener code.
	//******************************
	
	
	//Observer functions.
	public void update(Observable o, Object arg)
	{
		try{
			String tmpString = (String) arg;
			if(tmpString.equals("prefEvent"))
			{
				//Just need to call a repaint on the ThreadDataWindowPanel.
				sMWPanel.repaint();
			}
			else if(tmpString.equals("colorEvent"))
			{
				//Just need to call a repaint on the ThreadDataWindowPanel.
				sMWPanel.repaint();
			}
			else if(tmpString.equals("dataSetChangeEvent"))
			{
				//Clear any locally saved data.
	 			currentSMWGeneralData = null;
	 			currentSMWMeanData = null;
	 			
	 			//Now sort the data.
	 			sortLocalData();
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SMW04");
		}
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
			jRacy.systemError(null, "SMW05");
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
			jRacy.systemError(null, "SMW06");
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
			contentPane.remove(scrollPane);
			
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
			gbc.weightx = 1.0;
			gbc.weighty = 0.99;
			addCompItem(scrollPane, gbc, 0, 1, 4, 1);
		}
		else
		{
			contentPane.remove(sliderMultipleLabel);
			contentPane.remove(sliderMultiple);
			contentPane.remove(barLengthLabel);
			contentPane.remove(barLengthSlider);
			contentPane.remove(scrollPane);
			
			gbc.fill = GridBagConstraints.BOTH;
			gbc.anchor = GridBagConstraints.CENTER;
			gbc.weightx = 1;
			gbc.weighty = 1;
			addCompItem(scrollPane, gbc, 0, 0, 1, 1);
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
			
			contentPane.add(c, gbc);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SMW07");
		}
	}
	
	//******************************
	//End - Event listener code!!
	//******************************
	
	public StaticMainWindowData getSMWData()
	{
		return sMWData;
	}
	
	//Updates the sorted lists after a change of sorting method takes place.
	private void sortLocalData()
	{
		try{
			//First, do the currentSMWGeneralData.
			if(sortByMappingID)
			{
				if(descendingOrder)
					currentSMWGeneralData = sMWData.getSMWGeneralData("FIdDE");
				else
					currentSMWGeneralData = sMWData.getSMWGeneralData("FIdAE");
			}
			else if(sortByName)
			{
				if(descendingOrder)
					currentSMWGeneralData = sMWData.getSMWGeneralData("NDE");
				else
					currentSMWGeneralData = sMWData.getSMWGeneralData("NAE");
			}
			else if(sortByMillisecond)
			{
				if(descendingOrder)
					currentSMWGeneralData = sMWData.getSMWGeneralData("MDE");
				else
					currentSMWGeneralData = sMWData.getSMWGeneralData("MAE");
			}
			
			//Now do the currentSMWMeanData.
			if(sortByMappingID)
			{
				if(descendingOrder)
					currentSMWMeanData = sMWData.getSMWMeanData("FIdDE");
				else
					currentSMWMeanData = sMWData.getSMWMeanData("FIdAE");
			}
			else if(sortByName)
			{
				if(descendingOrder)
					currentSMWMeanData = sMWData.getSMWMeanData("NDE");
				else
					currentSMWMeanData = sMWData.getSMWMeanData("NAE");
			}
			else if(sortByMillisecond)
			{
				if(descendingOrder)
					currentSMWMeanData = sMWData.getSMWMeanData("MDE");
				else
					currentSMWMeanData = sMWData.getSMWMeanData("MAE");
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SMW08");
		}
		
	}
	
	//This function passes the correct data list to its panel when asked for.
	//Note:  This is only meant to be called by the StaticMainWindowPanel.
	public Vector getSMWGeneralData()
	{
		return currentSMWGeneralData;
	}
	
	//This function passes the correct data list to its panel when asked for.
	//Note:  This is only meant to be called by the StaticMainWindowPanel.
	public Vector getSMWMeanData()
	{
		return currentSMWMeanData;
	}
	
	public boolean isDataLoaded()
	{
		try{
			return sMWData.isDataLoaded();
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "SMW09");
		}
		
		return false;
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
	void thisWindowClosing(java.awt.event.WindowEvent e)
	{
		setVisible(false);
		dispose();
		System.exit(0);
	}	
}
