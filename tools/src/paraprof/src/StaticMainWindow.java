/*  
  StaticMainWindow.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

//import ParaProf.dss.*;
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

public class StaticMainWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener{ 
  
  public StaticMainWindow(Trial inTrial){
      try{
	  
	  //This window needs to maintain a reference to its trial.
	  trial = inTrial;
	  

	  //******************************
	  //Window Stuff.
	  //******************************
	  setTitle("ParaProf: " + trial.getProfilePathName());
	  
	  int windowWidth = 750;
	  int windowHeight = 400;
	  setSize(new java.awt.Dimension(windowWidth, windowHeight));
	  
	  sMWData = new StaticMainWindowData(trial);
	  
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
	  //******************************
	  //End -Window Stuff.
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
	  JMenu openMenu = new JMenu("Open ...");
	  /*
	  //Add a menu item.
	  JMenuItem openPprofDumpFileItem = new JMenuItem("Pprof Dump File");
	  openPprofDumpFileItem.addActionListener(this);
	  openMenu.add(openPprofDumpFileItem);
	  
	  
	  //Add a menu item.
	  JMenuItem openParaProfOutputItem = new JMenuItem("ParaProf Output File");
	  openParaProfOutputItem.addActionListener(this);
	  openMenu.add(openParaProfOutputItem);
	  */
	  
	  //Add a menu item.
	  JMenuItem openExperimentManagerItem = new JMenuItem("ParaProf Manager");
	  openExperimentManagerItem.addActionListener(this);
	  openMenu.add(openExperimentManagerItem);
	  
	  //Add a menu item.
	  JMenuItem showBinWindowItem = new JMenuItem("Bin Window");
	  showBinWindowItem.addActionListener(this);
	  openMenu.add(showBinWindowItem);
	  
	  //Add a menu item.
	  JMenuItem testItem = new JMenuItem("test");
	  testItem.addActionListener(this);
	  openMenu.add(testItem);
	  
	  fileMenu.add(openMenu);
	  //End submenu.
	  
	  //Add a submenu.
	  JMenu saveMenu = new JMenu("Save ...");
	  //Add a menu item.
	  //JMenuItem saveParaProfDataFileFileItem = new JMenuItem("To A ParaProf Output File");
	  //saveParaProfDataFileFileItem.addActionListener(this);
	  //saveMenu.add(saveParaProfDataFileFileItem);
	  
	  
	  //Add a menu item.
	  JMenuItem saveParaProfPreferencesItem = new JMenuItem("ParaProf Preferrences");
	  saveParaProfPreferencesItem.addActionListener(this);
	  saveMenu.add(saveParaProfPreferencesItem);
	  
	  fileMenu.add(saveMenu);
	  //End submenu.
	  
	  /*//Add a menu item.
	    JMenuItem printItem = new JMenuItem("Print");
	    printItem.addActionListener(this);
	    fileMenu.add(printItem);
	  */
	  
	  //Add a menu item.
	  JMenuItem editPrefItem = new JMenuItem("Edit ParaProf Preferences!");
	  editPrefItem.addActionListener(this);
	  fileMenu.add(editPrefItem);
	  
	  //Add a menu item.
	  JMenuItem saveImageItem = new JMenuItem("Save Image");
	  saveImageItem.addActionListener(this);
	  fileMenu.add(saveImageItem);

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
	  sMWPanel = new StaticMainWindowPanel(trial, this);
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
      
	  //******************************
	  //Setup the static main window data lists.
	  //******************************
	  sMWData = new StaticMainWindowData(trial);
	  sMWData.buildStaticMainWindowDataLists();
	  //******************************
	  //End - Setup the static main window data lists.
	  //******************************
      
	  //Sort the data for the main window.
	  sortLocalData();
      
	  //Call a repaint of the sMWPanel
	  sMWPanel.repaint();
      }
      catch(Exception e) { 
	  ParaProf.systemError(e, null, "SMW01");
      }
      
  }
    
    //******************************
    //Event listener code!!
    //******************************
    
    //ActionListener code.
    public void actionPerformed(ActionEvent evt){
	try{
	    Object EventSrc = evt.getSource();
	    
	    if(EventSrc instanceof JMenuItem){
		String arg = evt.getActionCommand();

		if(arg.equals("ParaProf Manager")){
		    ParaProfManager jRM = new ParaProfManager();
		    jRM.show();
		}
		else if(arg.equals("Bin Window")){
		    BinWindow bW = new BinWindow(trial, sMWData, true, -1);
		    bW.show();
		}
		else if(arg.equals("test")){
		    //DSSTraditionalTAUProfile test = new DSSTraditionalTAUProfile();
		    //test.loadData(new DMSAccessSession());
		}
		else if(arg.equals("testImage")){
		    //DSSTraditionalTAUProfile test = new DSSTraditionalTAUProfile();
		    //test.loadData(new DMSAccessSession());
		}
		else if(arg.equals("ParaProf Preferrences")){
		    
		    /*
		    //Set the directory to the current directory.
		    fileChooser.setCurrentDirectory(new File("."));
		    fileChooser.setSelectedFile(new File("ParaProfPreferences.dat"));
		    
		    //Bring up the save file chooser.
		    int resultValue = fileChooser.showSaveDialog(this);
		    
		    if(resultValue == JFileChooser.APPROVE_OPTION){
		    //Get the file.
		    File file = fileChooser.getSelectedFile();
		    
		    
		    //Check to make sure that something was obtained.
		    if(file != null){
		    try{
		    //Write to the savedPreferences object.
		    ParaProf.clrChooser.setSavedColors();
		    trial.getPreferences().setSavedPreferences();
		    
		    ObjectOutputStream prefsOut = new ObjectOutputStream(new FileOutputStream(file));
		    prefsOut.writeObject(ParaProf.savedPreferences);
		    prefsOut.close();
		    
		    }
		    catch(Exception e){
		    //Display an error
		    JOptionPane.showMessageDialog(this, "An error occured whilst trying to save ParaProf preferences.", "Error!"
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
		    */
		}
		else if(arg.equals("Edit ParaProf Preferences!")){
		    trial.getPreferences().showPreferencesWindow();
		}
		else if(arg.equals("Save Image")){
		    ParaProfImageOutput imageOutput = new ParaProfImageOutput();
		    imageOutput.saveImage((ParaProfImageInterface) sMWPanel);
		}
		else if(arg.equals("Exit ParaProf!")){
		    setVisible(false);
		    dispose();
		    System.exit(0);
		}
		else if(arg.equals("function ID")){
		    if(mappingIDButton.isSelected()){
			sortByMappingID = true;
			sortByName = false;
			sortByMillisecond = false;
			//Sort the local data.
			sortLocalData();
			//Call repaint.
			sMWPanel.repaint();
		    }
		}
		else if(arg.equals("name")){
		    if(nameButton.isSelected()){
			sortByMappingID = false;
			sortByName = true;
			sortByMillisecond = false;
			//Sort the local data.
			sortLocalData();
		    //Call repaint.
			sMWPanel.repaint();
		    }
		}
		else if(arg.equals("millisecond")){
		    if(millisecondButton.isSelected()){
			sortByMappingID = false;
			sortByName = false;
			sortByMillisecond = true;
			//Sort the local data.
			sortLocalData();
			//Call repaint.
			sMWPanel.repaint();
		    }
		}
		else if(arg.equals("Descending")){
		    if(descendingButton.isSelected()){
			descendingOrder = true;
			//Sort the local data.
			sortLocalData();
			//Call repaint.
			sMWPanel.repaint();
		    }
		}
		else if(arg.equals("Ascending")){
		    if(ascendingButton.isSelected()){
			descendingOrder = false;
			//Sort the local data.
			sortLocalData();
			//Call repaint.
			sMWPanel.repaint();
		    }
		}
		else if(arg.equals("Display Sliders")){
		    if(displaySlidersButton.isSelected()){ 
			displaySiders(true);
		    }
		    else{
			displaySiders(false);
		    }
		}
		else if(arg.equals("Show Function Ledger")){
		    //Grab the global mapping and bring up the mapping ledger window.
		(trial.getGlobalMapping()).displayMappingLedger(0);
		}
		else if(arg.equals("Show Group Ledger")){
		    (trial.getGlobalMapping()).displayMappingLedger(1);
		}
		else if(arg.equals("Show User Event Ledger")){
		    (trial.getGlobalMapping()).displayMappingLedger(2);
		}
		else if(arg.equals("Close All Sub-Windows")){
		//Close the all subwindows.
		    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
		}
		else if(arg.equals("About Racy")){
		    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
		}
		else if(arg.equals("Show Help Window")){
		    //Show the racy help window.
		    ParaProf.helpWindow.show();
		}
	    }
	    else if(EventSrc == sliderMultiple){
		sMWPanel.changeInMultiples();
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SMW02");
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
	    ParaProf.systemError(e, null, "SMW03");
	}
	
    }
    
    public void menuDeselected(MenuEvent evt){}
    public void menuCanceled(MenuEvent evt){}
    
    //******************************
    //End - MenuListener code.
    //******************************
    
    //******************************
    //Change listener code.
    //******************************
    public void stateChanged(ChangeEvent event){
	sMWPanel.changeInMultiples();
    }
    //******************************
    //End - Change listener code.
    //******************************
    
    
    //Observer functions.
    public void update(Observable o, Object arg){
	try{
	    String tmpString = (String) arg;
	    if(tmpString.equals("prefEvent")){
		//Just need to call a repaint on the ThreadDataWindowPanel.
		sMWPanel.repaint();
	    }
	    else if(tmpString.equals("colorEvent")){
		//Just need to call a repaint on the ThreadDataWindowPanel.
		sMWPanel.repaint();
	    }
	    else if(tmpString.equals("dataEvent")){
		sortLocalData();
		sMWPanel.repaint();
	    }
	    else if(tmpString.equals("dataSetChangeEvent")){
		//Clear any locally saved data.
		currentSMWGeneralData = null;
		currentSMWMeanData = null;
		
		//Now sort the data.
		sortLocalData();
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SMW04");
	}
    }
    
    public int getSliderValue(){
	int tmpInt = -1;
	try{
	    tmpInt = barLengthSlider.getValue();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SMW05");
	}
	return tmpInt;
    }
  
    public double getSliderMultiple(){
	String tmpString = null;
	try{
	    tmpString = (String) sliderMultiple.getSelectedItem();
	    return Double.parseDouble(tmpString);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SMW06");
	}
	
	return 0;
    }
    
    private void displaySiders(boolean displaySliders){
	if(displaySliders){
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
	else{
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
    
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	try{
	    gbc.gridx = x;
	    gbc.gridy = y;
	    gbc.gridwidth = w;
	    gbc.gridheight = h;
	    
	    contentPane.add(c, gbc);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SMW07");
	}
    }
    
    //******************************
    //End - Event listener code!!
    //******************************
    
    public StaticMainWindowData getSMWData(){
	return sMWData;
    }
    
    //Updates the sorted lists after a change of sorting method takes place.
    private void sortLocalData(){
	try{
	    //First, do the currentSMWGeneralData.
	    if(sortByMappingID){
		if(descendingOrder)
		    currentSMWGeneralData = sMWData.getSMWGeneralData("FIdDE");
		else
		    currentSMWGeneralData = sMWData.getSMWGeneralData("FIdAE");
	    }
	    else if(sortByName){
		if(descendingOrder)
		    currentSMWGeneralData = sMWData.getSMWGeneralData("NDE");
		else
		    currentSMWGeneralData = sMWData.getSMWGeneralData("NAE");
	    }
	    else if(sortByMillisecond){
		if(descendingOrder)
		    currentSMWGeneralData = sMWData.getSMWGeneralData("MDE");
		else
		    currentSMWGeneralData = sMWData.getSMWGeneralData("MAE");
	    }
	    
	    //Now do the currentSMWMeanData.
	    if(sortByMappingID){
		if(descendingOrder)
		    currentSMWMeanData = sMWData.getSMWMeanData("FIdDE");
		else
		    currentSMWMeanData = sMWData.getSMWMeanData("FIdAE");
	    }
	    else if(sortByName){
		if(descendingOrder)
		    currentSMWMeanData = sMWData.getSMWMeanData("NDE");
		else{
		    currentSMWMeanData = sMWData.getSMWMeanData("NAE");
		}
	    }
	    else if(sortByMillisecond){
		if(descendingOrder)
		    currentSMWMeanData = sMWData.getSMWMeanData("MDE");
		else
		    currentSMWMeanData = sMWData.getSMWMeanData("MAE");
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SMW08");
	}
	
    }
    
    //This function passes the correct data list to its panel when asked for.
    //Note:  This is only meant to be called by the StaticMainWindowPanel.
    public Vector getSMWGeneralData(){
	return currentSMWGeneralData;
    }
    
    //This function passes the correct data list to its panel when asked for.
    //Note:  This is only meant to be called by the StaticMainWindowPanel.
    public Vector getSMWMeanData(){
	return currentSMWMeanData;
    }
    
    private boolean mShown = false;
    
    public void addNotify(){
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
	System.exit(0);
    }

    //******************************
    //Instance data.
    //******************************
    
    Trial trial = null;
    
    //Create a file chooser to allow the user to select files for loading data.
    JFileChooser fileChooser = new JFileChooser();
    
    //References for some of the componants for this frame.
    private StaticMainWindowPanel sMWPanel;
    private StaticMainWindowData sMWData;
    
    private ButtonGroup sortGroup;
    private ButtonGroup sortOrderGroup;
    
    private JRadioButtonMenuItem mappingIDButton;
    private JRadioButtonMenuItem nameButton;
    private JRadioButtonMenuItem millisecondButton;
    
    private JRadioButtonMenuItem ascendingButton;
    private JRadioButtonMenuItem descendingButton;
    
    private JRadioButtonMenuItem displaySlidersButton;
    
    private JMenuItem mappingGroupLedgerItem;
    private JMenuItem userEventLedgerItem;
    
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
}
