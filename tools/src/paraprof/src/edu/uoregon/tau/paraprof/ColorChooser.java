/* 
   ParaProf.java

   Title:      Racy
   Author:     Robert Bell
   Description:  
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.colorchooser.*;
import javax.swing.event.*;
import edu.uoregon.tau.dms.dss.*;


public class ColorChooser implements WindowListener{
    public ColorChooser(ParaProfTrial trial, SavedPreferences savedPreferences){
	try{
	    this.trial = trial;
      
	    if(savedPreferences != null){
		colors = savedPreferences.getColors();
		groupColors = savedPreferences.getGroupColors();
		highlightColor = savedPreferences.getHighlightColor();
		groupHighlightColor = savedPreferences.getGroupHighlightColor();
		userEventHighlightColor = savedPreferences.getUserEventHightlightColor();
		miscMappingsColor = savedPreferences.getMiscMappingsColor();
	    }
	    else{
		//Set the default colours.
		this.setDefaultColors();
		this.setDefaultGroupColors();
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC01");
	}
    }
  
    public void showColorChooser(){
	try{
	    if(!clrChooserFrameShowing){
		//Bring up the color chooser frame.
		clrChooserFrame = new ColorChooserFrame(trial, this);
		clrChooserFrame.addWindowListener(this);
		clrChooserFrame.show();
		clrChooserFrameShowing = true;
	    }
	    else{
		//Just bring it to the foreground.
		clrChooserFrame.show();
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC02");
	}
    }
  
    public void setSavedColors(){
	try{
	    ParaProf.savedPreferences.setColors(colors);
	    ParaProf.savedPreferences.setGroupColors(groupColors);
	    ParaProf.savedPreferences.setHighlightColor(highlightColor);
	    ParaProf.savedPreferences.setGroupHighlightColor(groupHighlightColor);
	    ParaProf.savedPreferences.setMiscMappingsColor(miscMappingsColor);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC03");
	}
    }
  
    public int getNumberOfColors(){
	int tmpInt = -1;
	try{
	    tmpInt = colors.size();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC04");
	}
	
	return tmpInt;
    }
  
    public int getNumberOfMappingGroupColors(){
	int tmpInt = -1;
 	try{
	    tmpInt = groupColors.size();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC05");
	}
	return tmpInt;
    }

    public void addColor(Color color){
	try{
	    colors.add(color);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC10");
	}
    }
  
    public void setColor(Color color, int location){
	try{
	    colors.setElementAt(color, location);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC06");
	}
    }
  
    public Color getColor(int location){
	Color color = null;
	try{
	    color = (Color) colors.elementAt(location);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC08");
	}
    
	return color;
    }

    public Vector getColors(){
	return colors;}

    public void addGroupColor(Color color){
	try{
	    groupColors.add(color);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC11");
	}
    }

    public void setGroupColor(Color color, int location){
	try{
	    groupColors.setElementAt(color, location);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC07");
	}
    }
  
    public Color getGroupColor(int location){
	Color color = null;
	try{
	    color = (Color) groupColors.elementAt(location);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC09");
	}
    
	return color;
    }
 
    public Vector getGroupColors(){
	return groupColors;}

    public void setHighlightColor(Color highlightColor){
	this.highlightColor = highlightColor;}
  
    public Color getHighlightColor(){
	return highlightColor;}
  
    public void setHighlightColorID(int highlightColorID){
	this.highlightColorID = highlightColorID;
	trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }
  
    public int getHighlightColorID(){
	return highlightColorID;}

    public void setGroupHighlightColor(Color groupHighlightColor){
	this.groupHighlightColor = groupHighlightColor;}
  
    public Color getGroupHighlightColor(){
	return groupHighlightColor;}
  
    public void setGroupHighlightColorID(int groupHighlightColorID){
	    this.groupHighlightColorID = groupHighlightColorID;
	    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }
  
    public int getGroupHighlightColorID(){
	return groupHighlightColorID;}
  
    public void setUserEventHightlightColor(Color userEventHighlightColor){
	this.userEventHighlightColor = userEventHighlightColor;}
  
    public Color getUserEventHightlightColor(){
	return userEventHighlightColor;}
    
    public void setUserEventHighlightColorID(int userEventHighlightColorID){
	    this.userEventHighlightColorID = userEventHighlightColorID;
	    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }
  
    public int getUserEventHightlightColorID(){
	return userEventHighlightColorID;}

    public void setMiscMappingsColor(Color miscMappingsColor){
	this.miscMappingsColor = miscMappingsColor;}
  
    public Color getMiscMappingsColor(){
	return miscMappingsColor;}

    //A function which sets the colors vector to be the default set.
    public void setDefaultColors(){
	try{
	    //Clear the colors vector.
	    colors.clear();
	    
	    //Add the default colours.
	    addColor(new Color(61,104,63));
	    addColor(new Color(102,0,51));
	    addColor(new Color(0,102,102));
	    addColor(new Color(0,51,255));
	    addColor(new Color(102,132,25));
	    addColor(new Color(119,71,145));
	    addColor(new Color(221,232,30));
	    addColor(new Color(70,156,168));
	    addColor(new Color(255,153,0));
	    addColor(new Color(0,255,0));
	    addColor(new Color(121,196,144));
	    addColor(new Color(86,88,112));
      
	    addColor(new Color(151,204,255));
	    addColor(new Color(102,102,255));
	    addColor(new Color(204,255,51));
	    addColor(new Color(255,204,153));
	    addColor(new Color(204,0,204));
	    addColor(new Color(0,102,102));
	    addColor(new Color(204,204,255));
	    addColor(new Color(102,255,255));
	    addColor(new Color(255,102,102));
	    addColor(new Color(255,204,204));
	    addColor(new Color(240,97,159));
	    addColor(new Color(0,102,153));
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC12");
	}
    }
  
    //A function which sets the groupColors vector to be the default set.
    public void setDefaultGroupColors(){
	try{
	    //Clear the globalColors vector.
	    groupColors.clear();
	    
	    //Add the default colours.
	    addGroupColor(new Color(102,0,102));
	    addGroupColor(new Color(51,51,0));
	    addGroupColor(new Color(204,0,51));
	    addGroupColor(new Color(0,102,102));
	    addGroupColor(new Color(255,255,102));
	    addGroupColor(new Color(0,0,102));
	    addGroupColor(new Color(153,153,255));
	    addGroupColor(new Color(255,51,0));
	    addGroupColor(new Color(255,153,0));
	    addGroupColor(new Color(255,102,102));
	    addGroupColor(new Color(51,0,51));
	    addGroupColor(new Color(255,255,102));
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CC13");
	}
    }

    //Sets the colors of the given GlobalMapping.
    //If the mapping selection is equal to -1, then set the colors in all the mappings,
    //otherwise, just set the ones for the specified mapping.
    public void setColors(GlobalMapping globalMapping, int mappingSelection){
	if((mappingSelection == -1) || (mappingSelection == 0)){
	    int numberOfColors = this.getNumberOfColors();
	    for(Enumeration e = globalMapping.getMapping(0).elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		globalMappingElement.setColor(this.getColor((globalMappingElement.getMappingID()) % numberOfColors));
	    }
	}
	
	if((mappingSelection == -1) || (mappingSelection == 1)){
	    int numberOfColors = this.getNumberOfMappingGroupColors();
	    for(Enumeration e = globalMapping.getMapping(1).elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		globalMappingElement.setColor(this.getGroupColor((globalMappingElement.getMappingID()) % numberOfColors));
	    }
	}

	if((mappingSelection == -1) || (mappingSelection == 2)){
	    int numberOfColors = this.getNumberOfColors();
	    for(Enumeration e = globalMapping.getMapping(2).elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		globalMappingElement.setColor(this.getColor((globalMappingElement.getMappingID()) % numberOfColors));
	    }
	}
    }
    
    //Window Listener code.
    public void windowClosed(WindowEvent winevt){}
    public void windowIconified(WindowEvent winevt){}
    public void windowOpened(WindowEvent winevt){}
    public void windowClosing(WindowEvent winevt){
	if(winevt.getSource() == clrChooserFrame){
	    clrChooserFrameShowing = false;
	}
    }
    public void windowDeiconified(WindowEvent winevt){}
    public void windowActivated(WindowEvent winevt){}
    public void windowDeactivated(WindowEvent winevt){}
    
    //####################################
    //Instance Data.
    //####################################
    private ParaProfTrial trial = null;
    private Vector colors = new Vector();
    private Vector groupColors = new Vector();
    private Color highlightColor = Color.red;
    private int highlightColorID = -1;
    private Color groupHighlightColor = new Color(0,255,255);
    private int groupHighlightColorID = -1;
    private Color userEventHighlightColor = new Color(255,255, 0);
    private int userEventHighlightColorID = -1;
    private Color miscMappingsColor = Color.black;
    private boolean clrChooserFrameShowing = false; //For determining whether the clrChooserFrame is showing.
    private ColorChooserFrame clrChooserFrame;
    //####################################
    //End - Instance Data.
    //####################################
}
  
    
class ColorChooserFrame extends JFrame implements ActionListener{ 
    public ColorChooserFrame(ParaProfTrial trial, ColorChooser colorChooser){
	try{
	    this.trial = trial;
	    this.colorChooser = colorChooser;
	    numberOfColors = trial.getColorChooser().getNumberOfColors();
      
	    //Window Stuff.
	    setLocation(new Point(100, 100));
	    setSize(new Dimension(850, 450));
      
	    //####################################
	    //Code to generate the menus.
	    //####################################
	    JMenuBar mainMenu = new JMenuBar();
      
	    //######
	    //File menu.
	    //######
	    JMenu fileMenu = new JMenu("File");
      
	    JMenuItem closeItem = new JMenuItem("Close This Window");
	    closeItem.addActionListener(this);
	    fileMenu.add(closeItem);
      
	    JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
	    exitItem.addActionListener(this);
	    fileMenu.add(exitItem);
	    //######
	    //File menu.
	    //######
      
	    //######
	    //Help menu.
	    //######
	    /*JMenu helpMenu = new JMenu("Help");
      
	    //Add a menu item.
	    JMenuItem aboutItem = new JMenuItem("About Racy");
	    helpMenu.add(aboutItem);
      
	    //Add a menu item.
	    JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
	    showHelpWindowItem.addActionListener(this);
	    helpMenu.add(showHelpWindowItem);*/
	    //######
	    //Help menu.
	    //######
      
	    //Now, add all the menus to the main menu.
	    mainMenu.add(fileMenu);
	    //mainMenu.add(helpMenu);
	    setJMenuBar(mainMenu);
	    //####################################
	    //Code to generate the menus.
	    //####################################

  	    //####################################
	    //Create and add the components.
	    //####################################
	    //Setting up the layout system for the main window.
	    Container contentPane = getContentPane();
	    GridBagLayout gbl = new GridBagLayout();
	    contentPane.setLayout(gbl);
	    GridBagConstraints gbc = new GridBagConstraints();
	    gbc.insets = new Insets(5, 5, 5, 5);
 
	    //Create a new ColorChooser.
	    clrChooser = new JColorChooser();
	    clrModel = clrChooser.getSelectionModel();
      
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
      
	    //First add the label.
	    JLabel titleLabel = new JLabel("ParaProf Color Set.");
	    titleLabel.setFont(new Font("SansSerif", Font.ITALIC, 14));
	    addCompItem(titleLabel, gbc, 0, 0, 1, 1);
      
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
      
	    //Create and add color list.
	    listModel = new DefaultListModel();
	    colorList = new JList(listModel);
	    colorList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
	    colorList.setCellRenderer(new CustomCellRenderer(trial));
	    JScrollPane sp = new JScrollPane(colorList);
	    addCompItem(sp, gbc, 0, 1, 1, 5);
      
      
	    gbc.fill = GridBagConstraints.HORIZONTAL;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    addColorButton = new JButton("Add Color");
	    addColorButton.addActionListener(this);
	    addCompItem(addColorButton, gbc, 1, 1, 1, 1);
      
	    gbc.fill = GridBagConstraints.HORIZONTAL;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    addGroupColorButton = new JButton("Add Mapping Gr. Color");
	    addGroupColorButton.addActionListener(this);
	    addCompItem(addGroupColorButton, gbc, 1, 2, 1, 1);
      
	    gbc.fill = GridBagConstraints.HORIZONTAL;
	    gbc.anchor = GridBagConstraints.NORTH;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    deleteColorButton = new JButton("Delete Selected Color");
	    deleteColorButton.addActionListener(this);
	    addCompItem(deleteColorButton, gbc, 1, 3, 1, 1);
      
	    gbc.fill = GridBagConstraints.HORIZONTAL;
	    gbc.anchor = GridBagConstraints.NORTH;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    updateColorButton = new JButton("Update Selected Color");
	    updateColorButton.addActionListener(this);
	    addCompItem(updateColorButton, gbc, 1, 4, 1, 1);
      
	    gbc.fill = GridBagConstraints.HORIZONTAL;
	    gbc.anchor = GridBagConstraints.NORTH;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    restoreDefaultsButton = new JButton("Restore Defaults");
	    restoreDefaultsButton.addActionListener(this);
	    addCompItem(restoreDefaultsButton, gbc, 1, 5, 1, 1);
      
	    //Add the JColorChooser.
	    addCompItem(clrChooser, gbc, 2, 0, 1, 6);
	    //####################################
	    //End - Create and add the components.
	    //####################################
      
	    //Now populate the colour list.
	    populateColorList();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CCF01");
	}
    } 
  
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
	    if(arg.equals("Exit ParaProf!")){
		setVisible(false);
		dispose();
		System.exit(0);
	    }
	    else if(arg.equals("Close This Window")){
		setVisible(false);
	    }
	}
	else if(EventSrc instanceof JButton){
	    if(arg.equals("Add Color")){
		Color color = clrModel.getSelectedColor();
		(colorChooser.getColors()).add(color);
		listModel.clear();
		populateColorList();
		//Update the GlobalMapping.
		colorChooser.setColors(trial.getGlobalMapping(),0);
		//Update the listeners.
		trial.getSystemEvents().updateRegisteredObjects("colorEvent");
	    }
	    else if(arg.equals("Add Mapping Gr. Color")){
		Color color = clrModel.getSelectedColor();
		(colorChooser.getGroupColors()).add(color);
		listModel.clear();
		populateColorList();
		//Update the GlobalMapping.
		colorChooser.setColors(trial.getGlobalMapping(),1);
		//Update the listeners.
		trial.getSystemEvents().updateRegisteredObjects("colorEvent");
	    }
	    else if(arg.equals("Delete Selected Color")){
		//Get the currently selected items and cycle through them.
		int [] values = colorList.getSelectedIndices();
		for(int i = 0; i < values.length; i++){
		    if((values[i]) < trial.getColorChooser().getNumberOfColors()){
			System.out.println("The value being deleted is: " + values[i]);
			listModel.removeElementAt(values[i]);
			(colorChooser.getColors()).removeElementAt(values[i]);
			//Update the GlobalMapping.
			colorChooser.setColors(trial.getGlobalMapping(),0);
		    }
		    else if((values[i]) < (trial.getColorChooser().getNumberOfColors()) + (trial.getColorChooser().getNumberOfMappingGroupColors())){
			System.out.println("The value being deleted is: " + values[i]);
			listModel.removeElementAt(values[i]);
			(colorChooser.getGroupColors()).removeElementAt(values[i] - (trial.getColorChooser().getNumberOfColors()));
			//Update the GlobalMapping.
			colorChooser.setColors(trial.getGlobalMapping(),1);
		    }
		}
		
		//Update the listeners.
		trial.getSystemEvents().updateRegisteredObjects("colorEvent");
	    }
	    else if(arg.equals("Update Selected Color")){
		Color color = clrModel.getSelectedColor();
		//Get the currently selected items and cycle through them.
		int [] values = colorList.getSelectedIndices();
		for(int i = 0; i < values.length; i++){
		    listModel.setElementAt(color, values[i]);
		    int totalNumberOfColors = (trial.getColorChooser().getNumberOfColors()) + (trial.getColorChooser().getNumberOfMappingGroupColors());
		    if((values[i]) == (totalNumberOfColors)){
			trial.getColorChooser().setHighlightColor(color);
		    }
		    else if((values[i]) == (totalNumberOfColors+1)){
			trial.getColorChooser().setGroupHighlightColor(color);
		    }
		    else if((values[i]) == (totalNumberOfColors+2)){
			trial.getColorChooser().setUserEventHightlightColor(color);
		    }
		    else if((values[i]) == (totalNumberOfColors+3)){
			trial.getColorChooser().setMiscMappingsColor(color);
		    }
		    else if((values[i]) < trial.getColorChooser().getNumberOfColors()){
			colorChooser.setColor(color, values[i]);
			//Update the GlobalMapping.
			colorChooser.setColors(trial.getGlobalMapping(),0);
		    }
		    else{
			colorChooser.setGroupColor(color, (values[i] - trial.getColorChooser().getNumberOfColors()));
			//Update the GlobalMapping.
			colorChooser.setColors(trial.getGlobalMapping(),1);
		    }
		}
		//Update the listeners.
		trial.getSystemEvents().updateRegisteredObjects("colorEvent");
	    }
	    else if(arg.equals("Restore Defaults")){
		colorChooser.setDefaultColors();
		colorChooser.setDefaultGroupColors();
		colorChooser.setHighlightColor(Color.red);
		colorChooser.setGroupHighlightColor(new Color(0,255,255));
		colorChooser.setUserEventHightlightColor(new Color(255,255,0));
		colorChooser.setMiscMappingsColor(Color.black);
		listModel.clear();
		populateColorList();
		//Update the GlobalMapping.
		colorChooser.setColors(trial.getGlobalMapping(),0);
		colorChooser.setColors(trial.getGlobalMapping(),1);
		//Update the listeners.
		trial.getSystemEvents().updateRegisteredObjects("colorEvent");
	    }
	}
    }
  
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	gbc.gridx = x;
	gbc.gridy = y;
	gbc.gridwidth = w;
	gbc.gridheight = h;
	
	getContentPane().add(c, gbc);
    }
    
    void populateColorList(){
	Color color;
	for(Enumeration e = (colorChooser.getColors()).elements(); e.hasMoreElements() ;){
	    color = (Color) e.nextElement();
	    listModel.addElement(color);
	}
    
	for(Enumeration e = (colorChooser.getGroupColors()).elements(); e.hasMoreElements() ;){
	    color = (Color) e.nextElement();
	    listModel.addElement(color);
	}
    
	color = trial.getColorChooser().getHighlightColor();
	listModel.addElement(color);
	
	color = trial.getColorChooser().getGroupHighlightColor();
	listModel.addElement(color);
	
	color = trial.getColorChooser().getUserEventHightlightColor();
	listModel.addElement(color);
    
	color = trial.getColorChooser().getMiscMappingsColor();
	listModel.addElement(color);
    }
  
    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    private ColorChooser colorChooser;
    private ColorSelectionModel clrModel;
    private JColorChooser clrChooser;
    private DefaultListModel listModel;
    private JList colorList;
    private JButton addColorButton;
    private JButton addGroupColorButton;
    private JButton deleteColorButton;
    private JButton updateColorButton;
    private JButton restoreDefaultsButton;
    private int numberOfColors = -1;
    //####################################
    //End - Instance data.
    //####################################
}

class CustomCellRenderer implements ListCellRenderer{
    CustomCellRenderer(ParaProfTrial trial){
	this.trial = trial;
    }
  
    public Component getListCellRendererComponent(final JList list, final Object value,
						  final int index, final boolean isSelected,
						  final boolean cellHasFocus){
	return new JPanel(){
		public void paintComponent(Graphics g){
		    super.paintComponent(g);
		    Color inColor = (Color) value;
		    
		    int xSize = 0;
		    int ySize = 0;
		    int maxXNumFontSize = 0;
		    int maxXFontSize = 0;
		    int maxYFontSize = 0;
		    int thisXFontSize = 0;
		    int thisYFontSize = 0;
		    int barHeight = 0;
          
		    //For this, I will not allow changes in font size.
		    barHeight = 12;
		    
		    //Create font.
		    Font font = new Font(trial.getPreferences().getParaProfFont(), Font.PLAIN, barHeight);
		    g.setFont(font);
		    FontMetrics fmFont = g.getFontMetrics(font);
		    
		    maxXFontSize = fmFont.getAscent();
		    maxYFontSize = fmFont.stringWidth("0000,0000,0000");
		    
		    xSize = getWidth();
		    ySize = getHeight();
		    
		    String tmpString1 = new String("00" + (trial.getColorChooser().getNumberOfColors()));
		    maxXNumFontSize = fmFont.stringWidth(tmpString1);
		    
          
		    String tmpString2 = new String(inColor.getRed() + "," + inColor.getGreen() + "," + inColor.getBlue());
		    thisXFontSize = fmFont.stringWidth(tmpString2);
		    thisYFontSize = maxYFontSize;
          
          
		    g.setColor(isSelected ? list.getSelectionBackground() : list.getBackground());
		    g.fillRect(0, 0, (5 + maxXNumFontSize + 5), ySize);
            
		    int xStringPos1 = 5;
		    int yStringPos1 = (ySize - 5);
		    g.setColor(isSelected ? list.getSelectionForeground() : list.getForeground());
          
		    int totalNumberOfColors = (trial.getColorChooser().getNumberOfColors()) + (trial.getColorChooser().getNumberOfMappingGroupColors());
          
		    if(index == totalNumberOfColors){
			g.drawString(("" + ("MHC")), xStringPos1, yStringPos1);
		    }
		    else if(index == (totalNumberOfColors+1)){
			g.drawString(("" + ("GHC")), xStringPos1, yStringPos1);
		    }
		    else if(index == (totalNumberOfColors+2)){
			g.drawString(("" + ("UHC")), xStringPos1, yStringPos1);
		    }
		    else if(index == (totalNumberOfColors+3)){
			g.drawString(("" + ("MPC")), xStringPos1, yStringPos1);
		    }
		    else if(index < (trial.getColorChooser().getNumberOfColors())){
			g.drawString(("" + (index + 1)), xStringPos1, yStringPos1);
		    }
		    else{
			g.drawString(("G" + (index - (trial.getColorChooser().getNumberOfColors()) + 1)), xStringPos1, yStringPos1);
		    }
		    
		    g.setColor(inColor);
		    g.fillRect((5 + maxXNumFontSize + 5), 0,50, ySize);
		    
		    //Just a sanity check.
		    if((xSize - 50) > 0){
			g.setColor(isSelected ? list.getSelectionBackground() : list.getBackground());
			g.fillRect((5 + maxXNumFontSize + 5 + 50), 0,(xSize - 50), ySize);
		    }
		    
		    int xStringPos2 = 50 + (((xSize - 50) - thisXFontSize) / 2);
		    int yStringPos2 = (ySize - 5);
		    
		    g.setColor(isSelected ? list.getSelectionForeground() : list.getForeground());
		    g.drawString(tmpString2, xStringPos2, yStringPos2);
		}
		
		
		public Dimension getPreferredSize(){
		    int xSize = 0;
		    int ySize = 0;
		    int maxXNumFontSize = 0;
		    int maxXFontSize = 0;
		    int maxYFontSize = 0;
		    int barHeight = 12;
          
		    //Create font.
		    Font font = new Font(trial.getPreferences().getParaProfFont(), Font.PLAIN, barHeight);
		    Graphics g = getGraphics();
		    FontMetrics fmFont = g.getFontMetrics(font);
		    
		    String tmpString = new String("00" + (trial.getColorChooser().getNumberOfColors()));
		    maxXNumFontSize = fmFont.stringWidth(tmpString);
		    
		    maxXFontSize = fmFont.stringWidth("0000,0000,0000");
		    maxYFontSize = fmFont.getAscent();
		    
		    xSize = (maxXNumFontSize + 10 + 50 + maxXFontSize + 20);
		    ySize = (10 + maxYFontSize);
		    
		    return new Dimension(xSize,ySize);
		}
	    };
    }
    
    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    //####################################
    //End - Instance data.
    //####################################
}
