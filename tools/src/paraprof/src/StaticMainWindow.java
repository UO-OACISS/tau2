/*  
  StaticMainWindow.java

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
import java.awt.print.*;

public class StaticMainWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener{ 
  
  public StaticMainWindow(ParaProfTrial inParaProfTrial){
      try{
	  
	  //This window needs to maintain a reference to its trial.
	  trial = inParaProfTrial;
	  

	  //####################################
	  //Window Stuff.
	  //####################################
	  setTitle("ParaProf: " + trial.getTrialIdentifier(true));
	  
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
	  //####################################
	  //End -Window Stuff.
	  //####################################

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
	  
	  nameCheckBox = new JCheckBoxMenuItem("Sort By Name", false);
	  nameCheckBox.addActionListener(this);
	  optionsMenu.add(nameCheckBox);

	  orderCheckBox = new JCheckBoxMenuItem("Decending Order", true);
	  orderCheckBox.addActionListener(this);
	  optionsMenu.add(orderCheckBox);

	  slidersCheckBox = new JCheckBoxMenuItem("Display Sliders", false);
	  slidersCheckBox.addActionListener(this);
	  optionsMenu.add(slidersCheckBox);

	  pathTitleCheckBox = new JCheckBoxMenuItem("Show Path Title in Reverse", true);
	  pathTitleCheckBox.addActionListener(this);
	  optionsMenu.add(pathTitleCheckBox);

	  metaDataCheckBox = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
	  metaDataCheckBox.addActionListener(this);
	  optionsMenu.add(metaDataCheckBox);

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
	  panel = new StaticMainWindowPanel(trial, this);
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
	  gbc.weightx = 1;
	  gbc.weighty = 1;
	  addCompItem(sp, gbc, 0, 0, 1, 1);
	  //####################################
	  //End - Create and add the components.
	  //####################################

	  //####################################
	  //Setup the static main window data lists.
	  //####################################
	  sMWData = new StaticMainWindowData(trial);
	  sMWData.buildSMWGeneralData();
	  //####################################
	  //End - Setup the static main window data lists.
	  //####################################
      
	  //Sort the data for the main window.
	  sortLocalData();
      
	  //Call a repaint of the panel
	  panel.repaint();
      }
      catch(Exception e) { 
	  UtilFncs.systemError(e, null, "SMW01");
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
		else if(arg.equals("ParaProf Manager")){
		    ParaProfManager jRM = new ParaProfManager();
		    jRM.show();
		}
		else if(arg.equals("Bin Window")){
		    //BinWindow bW = new BinWindow(trial, sMWData, true, -1);
		    //bW.show();
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
		else if(arg.equals("name")){
		    if(nameCheckBox.isSelected())
			name = true;
		    else
			name = false;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Decending Order")){
		    if(orderCheckBox.isSelected())
			order = 0;
		    else
			order = 1;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Display Sliders")){
		    if(slidersCheckBox.isSelected())
			displaySiders(true);
		    else
			displaySiders(false);
		}
		else if(arg.equals("Show Path Title in Reverse"))
		    this.setTitle("ParaProf: " + trial.getTrialIdentifier(pathTitleCheckBox.isSelected()));
		else if(arg.equals("Show Meta Data in Panel"))
		    this.setHeader();
		else if(arg.equals("Show Function Ledger")){
		    (new MappingLedgerWindow(trial, 0)).show();
		}
		else if(arg.equals("Show Group Ledger")){
		    (new MappingLedgerWindow(trial, 1)).show();
		}
		else if(arg.equals("Show User Event Ledger")){
		    (new MappingLedgerWindow(trial, 2)).show();
		}
		else if(arg.equals("Show Call Path Relations")){
		    CallPathTextWindow tmpRef = new CallPathTextWindow(trial, -1, -1, -1, this.getSMWData(),true);
		    trial.getSystemEvents().addObserver(tmpRef);
		    tmpRef.show();
		}
		else if(arg.equals("Close All Sub-Windows")){
		    //Close the all subwindows.
		    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
		}
		else if(arg.equals("About ParaProf")){
		    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
		}
		else if(arg.equals("Show Help Window")){
		    ParaProf.helpWindow.show();
		}
	    }
	    else if(EventSrc == sliderMultiple){
		panel.changeInMultiples();
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMW02");
	}
    }
    //######
    //End - ActionListener.
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
	    UtilFncs.systemError(e, null, "SMW03");
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
		//Just need to call a repaint on the ThreadDataWindowPanel.
		panel.repaint();
	    }
	    else if(tmpString.equals("colorEvent")){
		//Just need to call a repaint on the ThreadDataWindowPanel.
		panel.repaint();
	    }
	    else if(tmpString.equals("dataEvent")){
		sortLocalData();
		this.setHeader();
		panel.repaint();
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMW04");
	}
    }
    //######
    //End - Observer.
    //######

    //####################################
    //End - Interface code.
    //####################################
    
    //######
    //Panel header.
    //######
    //This process is separated into two functions to provide the option
    //of obtaining the current header string being used for the panel
    //without resetting the actual header. Printing and image generation
    //use this functionality for example.
    public void setHeader(){
	if(metaDataCheckBox.isSelected()){
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
	    "Value Type: "+UtilFncs.getValueTypeString(2)+"\n";
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
	    UtilFncs.systemError(e, null, "SMW05");
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
	    UtilFncs.systemError(e, null, "SMW06");
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
	    gbc.weightx = 1.0;
	    gbc.weighty = 0.99;
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
	    gbc.weightx = 1;
	    gbc.weighty = 1;
	    addCompItem(sp, gbc, 0, 0, 1, 1);
	}
    
	//Now call validate so that these component changes are displayed.
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
	    UtilFncs.systemError(e, null, "SMW07");
	}
    }

    public StaticMainWindowData getSMWData(){
	return sMWData;
    }
    
    //Updates the sorted lists after a change of sorting method takes place.
    private void sortLocalData(){
	try{
	    if(name){
		list[0] = sMWData.getSMWGeneralData(0+order);
		list[1] = sMWData.getMeanData(18+order);
	    }
	    else{
		list[0] = sMWData.getSMWGeneralData(2+order);
		list[1] = sMWData.getMeanData(20+order);
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMW08");
	}
	
    }

    public Vector[] getData(){
	return list;}
    
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
    
    //Close the window when the close box is clicked
    void thisWindowClosing(java.awt.event.WindowEvent e){
	closeThisWindow();}

    void closeThisWindow(){ 
	try{
	    this.setVisible(false);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDW10");
	}
    }

    //####################################
    //Instance data.
    //####################################    
    ParaProfTrial trial = null;
    
    //Create a file chooser to allow the user to select files for loading data.
    JFileChooser fileChooser = new JFileChooser();
    
    //References for some of the components for this frame.
    private StaticMainWindowPanel panel = null;
    private StaticMainWindowData sMWData = null;

    private JMenu optionsMenu = null;
    private JMenu windowsMenu = null;
    private JCheckBoxMenuItem nameCheckBox = null;
    private JCheckBoxMenuItem orderCheckBox = null;
    private JCheckBoxMenuItem slidersCheckBox = null;
    private JCheckBoxMenuItem pathTitleCheckBox = null;
    private JCheckBoxMenuItem metaDataCheckBox = null;
    
    private JLabel sliderMultipleLabel = new JLabel("Slider Mulitiple");
    private JComboBox sliderMultiple;
    private JLabel barLengthLabel = new JLabel("Bar Mulitiple");
    private JSlider barLengthSlider = new JSlider(0, 40, 1);
    
    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;
    private JScrollPane sp;
    
    private boolean name = false; //true: sort by name,false: sort by value.
    private int order = 0; //0: descending order,1: ascending order.

    boolean displaySliders = false;
    
    private Vector[] list = new Vector[2]; //list[0]:The result of a call to getSMWGeneralData in StaticMainWindowData
                                           //list[1]:The result of a call to getMeanData in StaticMainWindowData
    //####################################
    //End - Instance data.
    //#################################### 
}
