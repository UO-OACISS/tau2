/*  
    Preferences.java

    Title:      ParaProf
    Author:     Robert Bell
    Description:  
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.colorchooser.*;
import edu.uoregon.tau.dms.dss.*;

public class Preferences extends JFrame implements ActionListener, Observer{ 
    
    public Preferences(ParaProfTrial trial, SavedPreferences savedPreferences){ 
	this.trial = trial;
	if(savedPreferences != null){
	    //######
	    //Set the saved values.
	    //######
	    paraProfFont = savedPreferences.getParaProfFont();
	    barSpacing = savedPreferences.getBarSpacing();
	    barHeight = savedPreferences.getBarHeight();
	    inExValue = savedPreferences.getInclusiveOrExclusive();
	    sortBy = savedPreferences.getSortBy();
	    
	    fontStyle = savedPreferences.getFontStyle();
	    
	    barDetailsSet = savedPreferences.getBarDetailsSet();
	    //######
	    //End - Set the saved values.
	    //######
	}
	    
	//Add some window listener code
	addWindowListener(new java.awt.event.WindowAdapter(){
		public void windowClosing(java.awt.event.WindowEvent evt) {
		    thisWindowClosing(evt);
		}
	    });
	
	//Get available fonts and initialize the fontComboBox..
	GraphicsEnvironment gE = GraphicsEnvironment.getLocalGraphicsEnvironment();
	String[] fontFamilyNames = gE.getAvailableFontFamilyNames();
	fontComboBox = new JComboBox(fontFamilyNames);
	
	int tmpInt = fontComboBox.getItemCount();
	int counter = 0;
	//We should always have some fonts available, so this should be safe.
	String tmpString = (String) fontComboBox.getItemAt(counter);
	while((counter < tmpInt) && (!(paraProfFont.equals(tmpString)))){
	    counter++;
	    tmpString = (String) fontComboBox.getItemAt(counter);
	}
	
	if(counter == tmpInt){
	    //The default font was not available.  Indicate an error.
	    System.out.println("The default font was not found!!  This is not a good thing as it is a default Java font!!");
	}
	else{
	    fontComboBox.setSelectedIndex(counter);
	}
	
	//Set the sliders.
	barHeightSlider.setValue(fontSize);
	
	fontComboBox.addActionListener(this);
	
	//Now initialize the panels.
	pSPanel = new PrefSpacingPanel(trial);
	
	//Window Stuff.
	setTitle("ParaProf Preferences: No Data Loaded");
	
	int windowWidth = 650;
	int windowHeight = 350;
	setSize(new java.awt.Dimension(windowWidth, windowHeight));
	
	//There is really no need to resize this window.
	setResizable(true);
	
	
	//Grab the screen size.
	Toolkit tk = Toolkit.getDefaultToolkit();
	Dimension screenDimension = tk.getScreenSize();
	int screenHeight = screenDimension.height;
	int screenWidth = screenDimension.width;
	
	//Set the window to come up in the center of the screen.
	int xPosition = 0;
	int yPosition = 0;
	
	setLocation(xPosition, yPosition);
    
	//End - Window Stuff.


	//####################################
	//Code to generate the menus.
	//####################################
	JMenuBar mainMenu = new JMenuBar();
    
	//######
	//File menu.
	//######
	JMenu fileMenu = new JMenu("File");
    
	//Add a menu item.
	JMenuItem editColorItem = new JMenuItem("Edit Color Map");
	editColorItem.addActionListener(this);
	fileMenu.add(editColorItem);
    
	//Add a menu item.
	JMenuItem saveColorItem = new JMenuItem("Save Color Map");
	saveColorItem.addActionListener(this);
	fileMenu.add(saveColorItem);
    
	//Add a menu item.
	JMenuItem loadColorItem = new JMenuItem("Load Color Map");
	loadColorItem.addActionListener(this);
	fileMenu.add(loadColorItem);
    
	//Add a menu item.
	JMenuItem closeItem = new JMenuItem("Apply and Close Window");
	closeItem.addActionListener(this);
	fileMenu.add(closeItem);
    
	//Add a menu item.
	JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
	exitItem.addActionListener(this);
	fileMenu.add(exitItem);
	//######
	//End - File menu.
	//######
    
	//######
	//Help menu.
	//######
	/*JMenu helpMenu = new JMenu("Help");
    
	//Add a menu item.
	JMenuItem aboutItem = new JMenuItem("About ParaProf");
	helpMenu.add(aboutItem);
    
	//Add a menu item.
	JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
	showHelpWindowItem.addActionListener(this);
	helpMenu.add(showHelpWindowItem);*/
	//######
	//End - Help menu.
	//######
    
    
	//Now, add all the menus to the main menu.
	mainMenu.add(fileMenu);
	//mainMenu.add(helpMenu);
    
	setJMenuBar(mainMenu);
	//####################################
	//Code to generate the menus.
	//####################################

	//####################################
	//Create and add the components
	//####################################
	
	//Setup the layout system for the main window.
	Container contentPane = getContentPane();
	GridBagLayout gbl = new GridBagLayout();
	contentPane.setLayout(gbl);
	GridBagConstraints gbc = new GridBagConstraints();
	gbc.insets = new Insets(5, 5, 5, 5);
    
	//######
	//Panel and ScrollPane definition.
	//######
	JScrollPane scrollPaneS = new JScrollPane(pSPanel);
	scrollPaneS.setPreferredSize(new Dimension(200, 200));
	//######
	//End - Panel and ScrollPane definition.
	//######
    
	//######
	//Slider Setup
	//######
	barHeightSlider.setPaintTicks(true);
	barHeightSlider.setMajorTickSpacing(20);
	barHeightSlider.setMinorTickSpacing(5);
	barHeightSlider.setPaintLabels(true);
	barHeightSlider.addChangeListener(pSPanel);
	//######
	//End - Slider Setup
	//######
    
	//######
	//RadioButton and ButtonGroup Setup
	//######
	normal = new JRadioButton("Plain Font", ((fontStyle == Font.PLAIN) || (fontStyle == (Font.PLAIN|Font.ITALIC))));
	normal.addActionListener(this);
	bold = new JRadioButton("Bold Font", ((fontStyle == Font.BOLD) || (fontStyle == (Font.BOLD|Font.ITALIC))));
	bold.addActionListener(this);
	italic = new JRadioButton("Italic Font", ((fontStyle == (Font.PLAIN|Font.ITALIC)) || (fontStyle == (Font.BOLD|Font.ITALIC))));
	italic.addActionListener(this);
    
	buttonGroup = new ButtonGroup();
	buttonGroup.add(normal);
	buttonGroup.add(bold);
	//######
	//End - RadioButton and ButtonGroup Setup
	//######
	    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.EAST;
	gbc.weightx = 1;
	gbc.weighty = 1;
	addCompItem(fontLabel, gbc, 0, 0, 1, 1);
    
	gbc.fill = GridBagConstraints.HORIZONTAL;
	gbc.anchor = GridBagConstraints.WEST;
	gbc.weightx = 1;
	gbc.weighty = 1;
	addCompItem(fontComboBox, gbc, 1, 0, 1, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 1;
	gbc.weighty = 1;
	addCompItem(normal, gbc, 2, 0, 1, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 1;
	gbc.weighty = 1;
	addCompItem(bold, gbc, 3, 0, 1, 1);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 1;
	gbc.weighty = 1;
	addCompItem(italic, gbc, 4, 0, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 1;
	gbc.weighty = 1;
	addCompItem(scrollPaneS, gbc, 0, 2, 2, 2);
    
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.NORTH;
	gbc.weightx = 1;
	gbc.weighty = 1;
	addCompItem(barHeightLabel, gbc, 2, 2, 1, 1);
    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.NORTH;
	gbc.weightx = 1;
	gbc.weighty = 1;
	addCompItem(barHeightSlider, gbc, 2, 3, 1, 1);
	//####################################
	//Code to generate the menus.
	//####################################
    }
  
    public void showPreferencesWindow(){
	//The path to data might have changed, therefore, reset the title.
	this.setTitle("ParaProf Preferences: " + ParaProf.profilePathName);
	this.show();
    }
  
    public void setSavedPreferences(){
	ParaProf.savedPreferences.setParaProfFont(paraProfFont);
	ParaProf.savedPreferences.setBarSpacing(barSpacing);
	ParaProf.savedPreferences.setBarHeight(barHeight);
	ParaProf.savedPreferences.setInclusiveOrExclusive(inExValue);
	ParaProf.savedPreferences.setSortBy(sortBy);
	ParaProf.savedPreferences.setFontStyle(fontStyle);
	ParaProf.savedPreferences.setBarDetailsSet(barDetailsSet);
    
    }

    public boolean areBarDetailsSet(){
	return barDetailsSet;}
  
    public String getParaProfFont(){
	return paraProfFont;}
  
    public int getFontStyle(){
	return fontStyle;}

    public int getFontSize(){
	return fontSize;}
  
    public void setBarDetails(Graphics2D g2D){
	if(!barDetailsSet){
	    Font font = new Font(paraProfFont, fontStyle, fontSize);
	    g2D.setFont(font);
	    FontMetrics fmFont = g2D.getFontMetrics(font);
	    int maxFontAscent = fmFont.getAscent();
	    int maxFontDescent = fmFont.getMaxDescent();
	    this.barHeight = maxFontAscent;
	    this.barSpacing = maxFontAscent + maxFontDescent + 2;
	    barDetailsSet = true;
	}
    }

    public void setFontSize(int fontSize){
	this.fontSize = fontSize;
	barDetailsSet = false;
    }

    public void updateFontSize(){
	fontSize = barHeightSlider.getValue();
	barDetailsSet = false;
    }
  
    public int getBarSpacing(){
	return barSpacing;}
  
    public int getBarHeight(){
	return barHeight;}
  
    public void setInExValue(String inExValue){
	this.inExValue = inExValue;}
  
    public String getInExValue(){
	return inExValue;}
  
    public void setSortBy(String sortBy){
	this.sortBy = sortBy;}
  
    public String getSortBy(){
	return sortBy;}
  
    //####################################
    //Interface code.
    //####################################
    
    //######
    //ActionListener.
    //######
    public void actionPerformed(ActionEvent evt){
	Object EventSrc = evt.getSource();
	String arg = evt.getActionCommand();
	if(EventSrc instanceof JMenuItem){
	    
	    if(arg.equals("Edit Color Map")){
		trial.getColorChooser().showColorChooser();
	    }
	    else if(arg.equals("Load Color Map")){
		JFileChooser fileChooser = new JFileChooser();
		
		//Set the directory to the current directory.
		fileChooser.setCurrentDirectory(new File("."));
		
		//Bring up the file chooser.
		int resultValue = fileChooser.showOpenDialog(this);
		
		if(resultValue == JFileChooser.APPROVE_OPTION){
		    //Try and get the file name.
		    File file = fileChooser.getSelectedFile();
		    
		    //Test to see if valid.
		    if(file != null){ 
			System.out.println("Loading color map ...");
			loadColorMap(file);
			trial.getSystemEvents().updateRegisteredObjects("prefEvent");
			System.out.println("Done loading color map ...");
		    }
		    else{
			System.out.println("There was some sort of internal error!");
		    }
		}
		
	    }
	    else if(arg.equals("Save Color Map")){
		JFileChooser fileChooser = new JFileChooser();
		
		//Set the directory to the current directory.
		fileChooser.setCurrentDirectory(new File("."));
		fileChooser.setSelectedFile(new File("colorMap.dat"));
		
		//Display the save file chooser.
		int resultValue = fileChooser.showSaveDialog(this);
		
		if(resultValue == JFileChooser.APPROVE_OPTION){
		    //Get the file.
		    File file = fileChooser.getSelectedFile();
		    
		    
		    //Check to make sure that something was obtained.
		    if(file != null){
			try{
			    //Just output the data for the moment to have a look at it.
			    Vector nameColorVector = new Vector();
			    GlobalMapping tmpGlobalMapping = trial.getGlobalMapping();
			    
			    int numOfMappings = tmpGlobalMapping.getNumberOfMappings(0);
			    
			    for(int i=0; i<numOfMappings; i++){
				GlobalMappingElement tmpGME = (GlobalMappingElement) tmpGlobalMapping.getGlobalMappingElement(i,0);
				if((tmpGME.getMappingName()) != null){ 
				    ColorPair tmpCP = new ColorPair(tmpGME.getMappingName(),tmpGME.getColor());
				    nameColorVector.add(tmpCP);
				}
			    }
			    Collections.sort(nameColorVector);
			    
			    
			    PrintWriter out = new PrintWriter(new FileWriter(file));
			    
			    System.out.println("Saving color map ...");
			    if(UtilFncs.debug){
				System.out.println("**********************");
				System.out.println("Color values loaded were:");
			    }
			    for(Enumeration e1 = nameColorVector.elements(); e1.hasMoreElements() ;){
				ColorPair tmpCP = (ColorPair) e1.nextElement();
				Color tmpColor = tmpCP.getColor();
				if(UtilFncs.debug){
				    System.out.println("MAPPING_NAME=\"" + (tmpCP.getMappingName()) + "\"" +
						       " RGB=\"" +
						       tmpColor.getRed() +
						       "," + tmpColor.getGreen() +
						       "," + tmpColor.getBlue() + "\"");
				}
				out.println("MAPPING_NAME=\"" + (tmpCP.getMappingName()) + "\"" +
					    " RGB=\"" +
					    tmpColor.getRed() +
					    "," + tmpColor.getGreen() +
					    "," + tmpColor.getBlue() + "\"");
			    }
			    if(UtilFncs.debug){
				System.out.println("**********************");
			    }
			    System.out.println("Done saving color map!");
			    out.close();
			}
			catch(Exception e){
			    //Display an error
			    JOptionPane.showMessageDialog(this, "An error occured whilst trying to save the color map.", "Error!"
							  ,JOptionPane.ERROR_MESSAGE);
			}
		    }
		    else{
			//Display an error
			JOptionPane.showMessageDialog(this, "No filename was given!", "Error!"
						      ,JOptionPane.ERROR_MESSAGE);
		    }
		}
	    }
	    else if(arg.equals("Exit ParaProf!")){
		setVisible(false);
		dispose();
		System.exit(0);
	    }
	    else if(arg.equals("Apply and Close Window")){
		setVisible(false);
		trial.getSystemEvents().updateRegisteredObjects("prefEvent");
	    }
	}
	else if(EventSrc instanceof JRadioButton){
	    if(arg.equals("Plain Font")){
		if(italic.isSelected())
		    fontStyle = Font.PLAIN|Font.ITALIC;
		else
		    fontStyle = Font.PLAIN;
		
		pSPanel.repaint();
	    }
	    else if(arg.equals("Bold Font")){
		if(italic.isSelected())
		    fontStyle = Font.BOLD|Font.ITALIC;
		else
		    fontStyle = Font.BOLD;
		
		pSPanel.repaint();
	    }
	    else if(arg.equals("Italic Font")){
		if(italic.isSelected()){
		    if(normal.isSelected())
			fontStyle = Font.PLAIN|Font.ITALIC;
		    else
			fontStyle = Font.BOLD|Font.ITALIC;
		}
		else{
		    if(normal.isSelected())
			fontStyle = Font.PLAIN;
		    else
			fontStyle = Font.BOLD;
		}
		
		pSPanel.repaint();
	    }
	}
	else if(EventSrc == fontComboBox){
	    paraProfFont = (String) fontComboBox.getSelectedItem();
	    pSPanel.repaint();
	}
    }
    //######
    //End - ActionListener.
    //######

    //######
    //Observer.
    //######
    public void update(Observable o, Object arg){
	String tmpString = (String) arg;
	if(tmpString.equals("colorEvent")){     
	    //Just need to call a repaint.
	    pSPanel.repaint();
	}
    }
    //######
    //End - Observer.
    //######

    //####################################
    //End - Interface code.
    //####################################

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	gbc.gridx = x;
	gbc.gridy = y;
	gbc.gridwidth = w;
	gbc.gridheight = h;
	getContentPane().add(c, gbc);
    }
    
    public void loadColorMap(File inFile){
	try{
	    //First, get the file stuff.
	    BufferedReader br = new BufferedReader(new FileReader(inFile));
	    
	    Vector nameColorVector = new Vector();
	    String tmpString;
	    int red = 0;
	    int green = 0;
	    int blue = 0;
	    
	    GlobalMapping tmpGlobalMapping = trial.getGlobalMapping(); 
	    
	    
	    //Read in the file line by line!
	    while((tmpString = br.readLine()) != null){ 
		StringTokenizer getMappingNameTokenizer = new StringTokenizer(tmpString, "\"");
		ColorPair tmpCP = new ColorPair();
		
		//The mapping name will be within the first set of quotes.
		//Grab the first token.
		tmpString = getMappingNameTokenizer.nextToken();
		//Grab the second token.
		tmpString = getMappingNameTokenizer.nextToken();
		
		tmpCP.setMappingName(tmpString);
		
		//The RGB values will be within the next set of quotes.
		//Grab the third token.
		tmpString = getMappingNameTokenizer.nextToken();
		//Grab the forth token.
		tmpString = getMappingNameTokenizer.nextToken();
		
		StringTokenizer getColorTokenizer = new StringTokenizer(tmpString, ",");
		
		tmpString = getColorTokenizer.nextToken();
		red = Integer.parseInt(tmpString);
		
		tmpString = getColorTokenizer.nextToken();
		green = Integer.parseInt(tmpString);
		
		tmpString = getColorTokenizer.nextToken();
		blue = Integer.parseInt(tmpString);
		
		Color tmpColor = new Color(red,green,blue);
		tmpCP.setMappingColor(tmpColor);
		
		nameColorVector.add(tmpCP);
	    }
	    
	    if(UtilFncs.debug){
		System.out.println("**********************");
		System.out.println("Color values loaded were:");
	    }
	    for(Enumeration e1 = nameColorVector.elements(); e1.hasMoreElements() ;){
		ColorPair tmpCP = (ColorPair) e1.nextElement();
		int mappingID = tmpGlobalMapping.getMappingID(tmpCP.getMappingName(), 0);
		if(mappingID != -1){
		    GlobalMappingElement tmpGME = tmpGlobalMapping.getGlobalMappingElement(mappingID, 0);
		    
		    Color tmpColor = tmpCP.getColor();
		    tmpGME.setSpecificColor(tmpColor);
		    tmpGME.setColorFlag(true);
		    if(UtilFncs.debug){
			System.out.println("MAPPING_NAME=\"" + (tmpCP.getMappingName()) + "\"" +
					   " RGB=\"" +
					   tmpColor.getRed() +
					   "," + tmpColor.getGreen() +
					   "," + tmpColor.getBlue() + "\"");
		    }
		}
	    }
	    if(UtilFncs.debug){
		System.out.println("**********************");
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "P01");
	}
    }
    
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
    }
    
    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    
    private boolean mShown = false;

    //References for some of the components for this frame.
    private PrefSpacingPanel pSPanel;
    
    private JCheckBox loadPprofDat;
  
  
    private JRadioButton normal;
    private JRadioButton bold;
    private JRadioButton italic;
  
    private ButtonGroup buttonGroup;
  
    private JLabel fontLabel = new JLabel("Font Selection");
  
    private JComboBox fontComboBox;
  
    private JLabel barHeightLabel = new JLabel("Adjust Bar Height");
    private JSlider barHeightSlider = new JSlider(SwingConstants.VERTICAL, 0, 100, 0);
    
    int fontStyle = Font.PLAIN;
    int fontSize = 12;
  
    private boolean barDetailsSet = false;
    private int barSpacing = 0;
    private int barHeight = 0;
  
    String paraProfFont = "SansSerif";
  
    //Whether we are doing inclusive or exclusive.
    String inExValue = "Exclusive";
  
    //Variable to determine which sorting paradigm has been chosen.
    String sortBy = "mappingID";  //Possible values are:
    //mappingID
    //millDes
    //millAsc

    //####################################
    //End - Instance data.
    //####################################
}
