/* 
   ThreadDataWindow.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  The container for the MappingDataWindowPanel.
*/

/*
  To do: 
  1) Change the name of this class to reflect the fact that it handles more than
  just thread displays.

  2) Update the help text for this window.
  
  3) Add some comments to some of the code.
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import java.awt.print.*;

public class ThreadDataWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener{
  
    public ThreadDataWindow(){
	try{
	    setLocation(new java.awt.Point(300, 200));
	    setSize(new java.awt.Dimension(700, 450));
	    
	    //Set the title indicating that there was a problem.
	    this.setTitle("Wrong constructor used!");
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW01");
	}
    }
    
    public ThreadDataWindow(ParaProfTrial trial, int nodeID, int contextID, int threadID, StaticMainWindowData sMWData, int windowType, boolean debug){
	try{
	    this.trial = trial;
	    this.sMWData = sMWData;
	    this.nodeID = nodeID;
	    this.contextID = contextID;
	    this.threadID = threadID;
	    this.windowType = windowType;
	    this.debug = debug;

	    setLocation(new java.awt.Point(300, 200));
	    setSize(new java.awt.Dimension(700, 450));
	    //Now set the title.
	    if(windowType==0)
		this.setTitle("Mean Data Window: " + trial.getTrialIdentifier(true));
	    else
		this.setTitle("n,c,t, " + nodeID + "," + contextID + "," + threadID + " - " + trial.getTrialIdentifier(true));
	    
	    //Add some window listener code
	    addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(java.awt.event.WindowEvent evt) {
			thisWindowClosing(evt);
		    }
		});
	    
	    //Set the help window text if required.
	    if(ParaProf.helpWindow.isVisible()){
		this.help(false);
	    }
	    
	    //Sort the local data.
	    sortLocalData();

	    //####################################
	    //Code to generate the menus.
	    //####################################
	    JMenuBar mainMenu = new JMenuBar();
	    JMenu subMenu = null;
	    JMenuItem menuItem = null;

	    //######
	    //File menu.
	    //######
	    JMenu fileMenu = new JMenu("File");
	    
	    //Open menu.
	    subMenu = new JMenu("Open ...");
	    
	    menuItem = new JMenuItem("ParaProf Manager");
	    menuItem.addActionListener(this);
	    subMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("Bin Window");
	    menuItem.addActionListener(this);
	    subMenu.add(menuItem);
	    
	    fileMenu.add(subMenu);
	    //End - Open menu.
	    
	    //Save menu.
	    subMenu = new JMenu("Save ...");
	    
	    menuItem = new JMenuItem("ParaProf Preferrences");
	    menuItem.addActionListener(this);
	    subMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("Save Image");
	    menuItem.addActionListener(this);
	    subMenu.add(menuItem);
	    
	    fileMenu.add(subMenu);
	    //End - Save menu.
	    
	    menuItem = new JMenuItem("Edit ParaProf Preferences!");
	    menuItem.addActionListener(this);
	    fileMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("Print");
	    menuItem.addActionListener(this);
	    fileMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("Close This Window");
	    menuItem.addActionListener(this);
	    fileMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("Exit ParaProf!");
	    menuItem.addActionListener(this);
	    fileMenu.add(menuItem);

	    fileMenu.addMenuListener(this);
	    //######
	    //End - File menu.
	    //######

	    //######
	    //Options menu.
	    //######
	    optionsMenu = new JMenu("Options");

	    JCheckBoxMenuItem box = null;
	    ButtonGroup group = null;
	    JRadioButtonMenuItem button = null;
	    
	    sortByName = new JCheckBoxMenuItem("Sort By Name", false);
	    sortByName.addActionListener(this);
	    optionsMenu.add(sortByName);

	    descendingOrder = new JCheckBoxMenuItem("Descending Order", true);
	    descendingOrder.addActionListener(this);
	    optionsMenu.add(descendingOrder);
	    
	    showValuesAsPercent = new JCheckBoxMenuItem("Show Values as Percent", true);
	    showValuesAsPercent.addActionListener(this);
	    optionsMenu.add(showValuesAsPercent);
	    
	    //Units submenu.
	    unitsSubMenu = new JMenu("Select Units");
	    group = new ButtonGroup();
	    
	    button = new JRadioButtonMenuItem("hr:min:sec", false);
	    button.addActionListener(this);
	    group.add(button);
	    unitsSubMenu.add(button);
	    
	    button = new JRadioButtonMenuItem("Seconds", false);
	    button.addActionListener(this);
	    group.add(button);
	    unitsSubMenu.add(button);
	    
	    button = new JRadioButtonMenuItem("Milliseconds", false);
	    button.addActionListener(this);
	    group.add(button);
	    unitsSubMenu.add(button);
	    
	    button = new JRadioButtonMenuItem("Microseconds", true);
	    button.addActionListener(this);
	    group.add(button);
	    unitsSubMenu.add(button);
	    
	    optionsMenu.add(unitsSubMenu);
	    //End - Units submenu.

	    //Set the value type options.
	    subMenu = new JMenu("Select Value Type");
	    group = new ButtonGroup();
	    
	    button = new JRadioButtonMenuItem("Exclusive", true);
	    button.addActionListener(this);
	    group.add(button);
	    subMenu.add(button);
	    
	    button = new JRadioButtonMenuItem("Inclusive", false);
	    button.addActionListener(this);
	    group.add(button);
	    subMenu.add(button);
	    
	    button = new JRadioButtonMenuItem("Number of Calls", false);
	    button.addActionListener(this);
	    group.add(button);
	    subMenu.add(button);
	    
	    button = new JRadioButtonMenuItem("Number of Subroutines", false);
	    button.addActionListener(this);
	    group.add(button);
	    subMenu.add(button);
	    
	    button = new JRadioButtonMenuItem("Per Call Value", false);
	    button.addActionListener(this);
	    group.add(button);
	    subMenu.add(button);
	    
	    optionsMenu.add(subMenu);
	    //End - Set the value type options.
	    
	    box = new JCheckBoxMenuItem("Display Sliders", false);
	    box.addActionListener(this);
	    optionsMenu.add(box);

	    showPathTitleInReverse = new JCheckBoxMenuItem("Show Path Title in Reverse", true);
	    showPathTitleInReverse.addActionListener(this);
	    optionsMenu.add(showPathTitleInReverse);
	    
	    showMetaData = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
	    showMetaData.addActionListener(this);
	    optionsMenu.add(showMetaData);
	    
	    optionsMenu.addMenuListener(this);
	    //######
	    //End - Options menu.
	    //######

	    //######
	    //Windows menu
	    //######
	    windowsMenu = new JMenu("Windows");
	    
	    menuItem = new JMenuItem("Show Function Ledger");
	    menuItem.addActionListener(this);
	    windowsMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("Show Group Ledger");
	    menuItem.addActionListener(this);
	    windowsMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("Show User Event Ledger");
	    menuItem.addActionListener(this);
	    windowsMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("Show Call Path Relations");
	    menuItem.addActionListener(this);
	    windowsMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("Close All Sub-Windows");
	    menuItem.addActionListener(this);
	    windowsMenu.add(menuItem);
	    
	    windowsMenu.addMenuListener(this);
	    //######
	    //End - Windows menu
	    //######

	    //######
	    //Help menu.
	    //######
	    JMenu helpMenu = new JMenu("Help");

	    menuItem = new JMenuItem("Show Help Window");
	    menuItem.addActionListener(this);
	    helpMenu.add(menuItem);
	    
	    menuItem = new JMenuItem("About ParaProf");
	    menuItem.addActionListener(this);
	    helpMenu.add(menuItem);
	    
	    helpMenu.addMenuListener(this);
	    //######
	    //End - Help menu.
	    //######
	    
	    //Now, add all the menus to the main menu.
	    mainMenu.add(fileMenu);
	    mainMenu.add(optionsMenu);
	    mainMenu.add(windowsMenu);
	    mainMenu.add(helpMenu);
	    
	    setJMenuBar(mainMenu);
	    //####################################
	    //End - Code to generate the menus.
	    //####################################

	    //####################################
	    //Create and add the components.
	    //####################################
	    //Setting up the layout system for the main window.
	    contentPane = getContentPane();
	    gbl = new GridBagLayout();
	    contentPane.setLayout(gbl);
	    gbc = new GridBagConstraints();
	    gbc.insets = new Insets(5, 5, 5, 5);
	    	    
	    //######
	    //Panel and ScrollPane definition.
	    //######
	    panel = new ThreadDataWindowPanel(trial, nodeID, contextID, threadID, this, sMWData, windowType, this.debug());
	    sp = new JScrollPane(panel);
	    this.setHeader();
	    //######
	    //End - Panel and ScrollPane definition.
	    //######
	    	    
	    //######
	    //Slider setup.
	    //Do the slider stuff, but don't add.  By default, sliders are off.
	    //######
	    String sliderMultipleStrings[] = {"1.00", "0.75", "0.50", "0.25", "0.10"};
	    sliderMultiple = new JComboBox(sliderMultipleStrings);
	    sliderMultiple.addActionListener(this);
	    
	    barLengthSlider.setPaintTicks(true);
	    barLengthSlider.setMajorTickSpacing(5);
	    barLengthSlider.setMinorTickSpacing(1);
	    barLengthSlider.setPaintLabels(true);
	    barLengthSlider.setSnapToTicks(true);
	    barLengthSlider.addChangeListener(this);
	    //######
	    //End - Slider setup.
	    //Do the slider stuff, but don't add.  By default, sliders are off.
	    //######
	    
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 0.95;
	    gbc.weighty = 0.98;
	    addCompItem(sp, gbc, 0, 0, 1, 1);
	    //####################################
	    //End - Create and add the components.
	    //####################################
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW02");
	}
    }
    
    //####################################
    //Interface code.
    //####################################
    
    //######
    //ActionListener.
    //######
    public void actionPerformed(ActionEvent evt){
	try{
	    Object EventSrc = evt.getSource();
	    
	    if(EventSrc instanceof JMenuItem){
		String arg = evt.getActionCommand();
		if(arg.equals("Print")){
		    PrinterJob job = PrinterJob.getPrinterJob();
		    PageFormat defaultFormat = job.defaultPage();
		    PageFormat selectedFormat = job.pageDialog(defaultFormat);
		    job.setPrintable(panel, selectedFormat);
		    if(job.printDialog()){
			job.print();
		    }
		}
		if(arg.equals("ParaProf Manager")){
		    ParaProfManager jRM = new ParaProfManager();
		    jRM.show();
		}
		else if(arg.equals("Edit ParaProf Preferences!")){
		    trial.getPreferences().showPreferencesWindow();
		}
		else if(arg.equals("Save Image")){
		    ParaProfImageOutput imageOutput = new ParaProfImageOutput();
		    imageOutput.saveImage((ParaProfImageInterface) panel);
		}
		else if(arg.equals("Close This Window")){
		    closeThisWindow();
		}
		else if(arg.equals("Exit ParaProf!")){
		    setVisible(false);
		    dispose();
		    System.exit(0);
		}
		else if(arg.equals("Sort By Name")){
		    if(sortByName.isSelected())
			name = true;
		    else
			name = false;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Descending Order")){
		    if(descendingOrder.isSelected())
			order = 0;
		    else
			order = 1;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Show Values as Percent")){
		    if(showValuesAsPercent.isSelected())
			percent = true;
		    else
			percent = false;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Exclusive")){
		    valueType = 2;
		    this.setHeader();
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Inclusive")){
		    valueType = 4;
		    this.setHeader();
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Number of Calls")){
		    valueType = 6;
		    this.setHeader();
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Number of Subroutines")){
		    valueType = 8;
		    this.setHeader();
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Per Call Value")){
		    valueType = 10;
		    this.setHeader();
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Microseconds")){
		    units = 0;
		    this.setHeader();
		    panel.repaint();
		}
		else if(arg.equals("Milliseconds")){
		    units = 1;
		    this.setHeader();
		    panel.repaint();
		}
		else if(arg.equals("Seconds")){
		    units = 2;
		    this.setHeader();
		    panel.repaint();
		}
		else if(arg.equals("hr:min:sec")){
		    units = 3;
		    this.setHeader();
		    panel.repaint();
		}
		else if(arg.equals("Display Sliders")){
		    if(((JCheckBoxMenuItem)optionsMenu.getItem(5)).isSelected())
			displaySiders(true);
		    else
			displaySiders(false);
		}
		else if(arg.equals("Show Path Title in Reverse")){
		    if(windowType==0)
			this.setTitle("Mean Data Window: " + 
				      trial.getTrialIdentifier(showPathTitleInReverse.isSelected()));
		    else
			this.setTitle("n,c,t, " + nodeID + "," + contextID + "," 
				      + threadID + " - " + trial.getTrialIdentifier(showPathTitleInReverse.isSelected()));
		}
		else if(arg.equals("Show Meta Data in Panel"))
		    this.setHeader();
		else if(arg.equals("Show Function Ledger")){
		    (new MappingLedgerWindow(trial, 0, this.debug())).show();
		}
		else if(arg.equals("Show Group Ledger")){
		    (new MappingLedgerWindow(trial, 1, this.debug())).show();
		}
		else if(arg.equals("Show User Event Ledger")){
		    (new MappingLedgerWindow(trial, 2, this.debug())).show();
		}
		else if(arg.equals("Close All Sub-Windows")){
		    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
		}
		else if(arg.equals("About ParaProf")){
		    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
		}
		else if(arg.equals("Show Help Window")){
		    this.help(true);
		}
		    }
	    else if(EventSrc == sliderMultiple){
		panel.changeInMultiples();
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW03");
	}
    }
    //######
    //End - ActionListener
    //######

    //######
    //ChangeListener.
    //######
    public void stateChanged(ChangeEvent event){
	panel.changeInMultiples();}
    //######
    //End - ChangeListener.
    //######
  
    //######
    //MenuListener.
    //######
    public void menuSelected(MenuEvent evt){
	try{
	    if(valueType > 4){
		showValuesAsPercent.setEnabled(false);
		unitsSubMenu.setEnabled(false);}
	    else if(percent){
		showValuesAsPercent.setEnabled(true);
		unitsSubMenu.setEnabled(false);}
	    else if(trial.isTimeMetric()){
		showValuesAsPercent.setEnabled(true);
		unitsSubMenu.setEnabled(true);}
	    else{
		showValuesAsPercent.setEnabled(true);
		unitsSubMenu.setEnabled(false);}

	    if(trial.groupNamesPresent())
		((JMenuItem)windowsMenu.getItem(1)).setEnabled(true);
	    else
		((JMenuItem)windowsMenu.getItem(1)).setEnabled(false);
	    
	    if(trial.userEventsPresent())
		((JMenuItem)windowsMenu.getItem(2)).setEnabled(true);
	    else
		((JMenuItem)windowsMenu.getItem(2)).setEnabled(false);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW04");
	}
    }
    
    public void menuDeselected(MenuEvent evt){}
    public void menuCanceled(MenuEvent evt){}
    //######
    //End - MenuListener.
    //######
    
    //######
    //Observer.
    //######
    public void update(Observable o, Object arg){
	try{
	    String tmpString = (String) arg;
	    if(tmpString.equals("prefEvent")){
		panel.repaint();
	    }
	    else if(tmpString.equals("colorEvent")){
		panel.repaint();
	    }
	    else if(tmpString.equals("dataEvent")){
		sortLocalData();
		if(!(trial.isTimeMetric()))
		    units = 0;
		this.setHeader();
		panel.repaint();
	    }
	    else if(tmpString.equals("subWindowCloseEvent")){
		closeThisWindow();
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW05");
	}
    }
    //######
    //End - Observer.
    //######

    //####################################
    //End - Interface code.
    //####################################

    private void help(boolean display){
	//Show the ParaProf help window.
	ParaProf.helpWindow.clearText();
	if(display)
	    ParaProf.helpWindow.show();
	ParaProf.helpWindow.writeText("This is the thread data window");
	ParaProf.helpWindow.writeText("");
	ParaProf.helpWindow.writeText("This window shows you the values for all mappings on this thread.");
	ParaProf.helpWindow.writeText("");
	ParaProf.helpWindow.writeText("Use the options menu to select different ways of displaying the data.");
	ParaProf.helpWindow.writeText("");
	ParaProf.helpWindow.writeText("Right click on any mapping within this window to bring up a popup");
	ParaProf.helpWindow.writeText("menu. In this menu you can change or reset the default colour");
	ParaProf.helpWindow.writeText("for the mapping, or to show more details about the mapping.");
	ParaProf.helpWindow.writeText("You can also left click any mapping to hightlight it in the system.");
    }
    
    //Updates this window's data copy.
    private void sortLocalData(){ 
	try{
	    //The name selection behaves slightly differently. Thus the check for it.
	    if(name){
		if(windowType==0)
		    list = sMWData.getMeanData(order);
		else
		    list = sMWData.getThreadData(nodeID, contextID, threadID, windowType, order);
	    }
	    else{
		if(windowType==0)
		    list = sMWData.getMeanData(18+valueType+order);
		else
		    list = sMWData.getThreadData(nodeID, contextID, threadID, windowType, valueType+order);
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW06");
	}
    }

    public Vector getData(){
	return list;}
    
    public int getWindowType(){
	return windowType;}

    public boolean isPercent(){
	return percent;}
    
    public int getValueType(){
	return valueType;}
    
    public int units(){
	return units;}

    //######
    //Panel header.
    //######
    //This process is separated into two functions to provide the option
    //of obtaining the current header string being used for the panel
    //without resetting the actual header. Printing and image generation
    //use this functionality for example.
    public void setHeader(){
	if(showMetaData.isSelected()){
	    JTextArea jTextArea = new JTextArea();
	    jTextArea.setLineWrap(true);
	    jTextArea.setWrapStyleWord(true);
	    jTextArea.setEditable(false);
	    jTextArea.append(this.getHeaderString());
	    sp.setColumnHeaderView(jTextArea);
	}
	else
	    sp.setColumnHeaderView(null);
    }

    public String getHeaderString(){
	return "Metric Name: " + (trial.getMetricName(trial.getSelectedMetricID()))+"\n" +
	    "Value Type: "+UtilFncs.getValueTypeString(valueType)+"\n"+
	    "Units: "+UtilFncs.getUnitsString(units, trial.isTimeMetric())+"\n";
    }
    //######
    //End - Panel header.
    //######

    
    public int getSliderValue(){
	int tmpInt = -1;
	
	try{
	    tmpInt = barLengthSlider.getValue();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW07");
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
	    UtilFncs.systemError(e, null, "FDW08");
	}
	
	return 0;
    }
    
    private void displaySiders(boolean displaySliders){
	if(displaySliders){
	    contentPane.remove(sp);
	    
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
	    addCompItem(sp, gbc, 0, 1, 4, 1);
	}
	else{
	    contentPane.remove(sliderMultipleLabel);
	    contentPane.remove(sliderMultiple);
	    contentPane.remove(barLengthLabel);
	    contentPane.remove(barLengthSlider);
	    contentPane.remove(sp);
	    
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 0.95;
	    gbc.weighty = 0.98;
	    addCompItem(sp, gbc, 0, 0, 1, 1);
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
	    
	    getContentPane().add(c, gbc);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW09");
	}
    }
    
    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e){
	closeThisWindow();}
  
    void closeThisWindow(){ 
	try{
	    if(this.debug()){
		System.out.println("------------------------");
		System.out.println("A thread window for: \"" + "n,c,t, " + nodeID + "," + contextID + "," + threadID + "\" is closing");
		System.out.println("Clearing resourses for that window.");
	    }
	    
	    setVisible(false);
	    trial.getSystemEvents().deleteObserver(this);
	    dispose();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW10");
	}
    }
 
    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    private StaticMainWindowData sMWData = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private int windowType = 1; //0: mean data,1: function data.

    private JMenu optionsMenu = null;
    private JMenu windowsMenu = null;
    private JMenu unitsSubMenu = null;

    private JCheckBoxMenuItem sortByName = null;
    private JCheckBoxMenuItem descendingOrder = null;
    private JCheckBoxMenuItem showValuesAsPercent = null;
    private JCheckBoxMenuItem displaySliders = null;
    private JCheckBoxMenuItem  showPathTitleInReverse = null;
    private JCheckBoxMenuItem  showMetaData = null;
    private JMenuItem groupLedger = null;
    private JMenuItem usereventLedger = null;
    private JMenuItem callPathRelations = null;
    

    private JLabel sliderMultipleLabel = new JLabel("Slider Mulitiple");
    private JComboBox sliderMultiple;
    private JLabel barLengthLabel = new JLabel("Bar Mulitiple");
    private JSlider barLengthSlider = new JSlider(0, 40, 1);
  
    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;
  
    private JScrollPane sp = null;
    private ThreadDataWindowPanel panel = null;
 
    private Vector list = null;
  
    private boolean name = false; //true: sort by name,false: sort by value.
    private int order = 0; //0: descending order,1: ascending order.
    private boolean percent = true; //true: show values as percent,false: show actual values.
    private int valueType = 2; //2-exclusive,4-inclusive,6-number of calls,8-number of subroutines,10-per call value.
    private int units = 0; //0-microseconds,1-milliseconds,2-seconds.

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################
}
