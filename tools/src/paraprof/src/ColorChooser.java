/* 
  ParaProf.java

  Title:      Racy
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.colorchooser.*;
import javax.swing.event.*;


public class ColorChooser implements WindowListener
{
  public ColorChooser(Trial inTrial, SavedPreferences inSavedPreferences)
  {
    
    try{
      trial = inTrial;
      
      if(inSavedPreferences != null)
      {
        globalColors = inSavedPreferences.getGlobalColors();
        mappingGroupColors = inSavedPreferences.getMappingGroupColors();
        highlightColor = inSavedPreferences.getHighlightColor();
        groupHighlightColor = inSavedPreferences.getGroupHighlightColor();
        uEHC = inSavedPreferences.getUEHC();
        miscMappingsColor = inSavedPreferences.getMiscMappingsColor();
      }
      else
      {
        //Set the default colours.
        this.setDefaultColors();
        this.setDefaultMappingGroupColors();
      }
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC01");
    }
  }
  
  public void showColorChooser()
  {
    try{
      if(!clrChooserFrameShowing)
      {
        //Bring up the color chooser frame.
        clrChooserFrame = new ColorChooserFrame(trial, this);
        clrChooserFrame.addWindowListener(this);
        clrChooserFrame.show();
        clrChooserFrameShowing = true;
      }
      else
      {
      //Just bring it to the foreground.
      clrChooserFrame.show();
      }
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC02");
    }
  }
  
  public void setSavedColors()
  {
    try{
      ParaProf.savedPreferences.setGlobalColors(globalColors);
      ParaProf.savedPreferences.setMappingGroupColors(mappingGroupColors);
      ParaProf.savedPreferences.setHighlightColor(highlightColor);
      ParaProf.savedPreferences.setGroupHighlightColor(groupHighlightColor);
      ParaProf.savedPreferences.setMiscMappingsColor(miscMappingsColor);
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC03");
    }
  }
  
  public int getNumberOfColors()
  {
    int tmpInt = -1;
    
    try{
      tmpInt = globalColors.size();
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC04");
    }
    
    return tmpInt;
  }
  
  public int getNumberOfMappingGroupColors()
  {
    int tmpInt = -1;
    
    try{
      tmpInt = mappingGroupColors.size();
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC05");
    }
    
    return tmpInt;
  }
  
  public void setColorInLocation(Color inColor, int inLocation)
  {
    try{
      globalColors.setElementAt(inColor, inLocation);
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC06");
    }
  }
  
  public void setMappingGroupColorInLocation(Color inColor, int inLocation)
  {
    try{
      mappingGroupColors.setElementAt(inColor, inLocation);
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC07");
    }
  }
  
  public Color getColorInLocation(int inLocation)
  {
    Color tmpColor = null;
    
    try{
      tmpColor = (Color) globalColors.elementAt(inLocation);
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC08");
    }
    
    return tmpColor;
  }
  
  public Color getMappingGroupColorInLocation(int inLocation)
  {
    Color tmpColor = null;
    
    try{
      tmpColor = (Color) mappingGroupColors.elementAt(inLocation);
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC09");
    }
    
    return tmpColor;
  }
  
  public void addColor(Color inColor)
  {
    try{
      globalColors.add(inColor);
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC10");
    }
  }
  
  public void addMappingGroupColor(Color inColor)
  {
    try{
      mappingGroupColors.add(inColor);
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC11");
    }
  }
  
  public Vector getAllColors()
  {
    return globalColors;
  }
  
  public Vector getAllMappingGroupColors()
  {
    return mappingGroupColors;
  }
  
  //***
  //Highlight color functions.
  //***
  public void setHighlightColor(Color inColor)
  {
    highlightColor = inColor;
  }
  
  public Color getHighlightColor()
  {
    return highlightColor;
  }
  
  public void setHighlightColorMappingID(int inInt)
  {
    highlightColorMappingID = inInt;
    
    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
  }
  
  public int getHighlightColorMappingID()
  {
    return highlightColorMappingID;
  }
  //***
  //End - Highlight color functions.
  //***
  
  
  //***
  //Group highlight color functions.
  //***
  public void setGroupHighlightColor(Color inColor)
  {
    groupHighlightColor = inColor;
  }
  
  public Color getGroupHighlightColor()
  {
    return groupHighlightColor;
  }
  
  public void setGroupHighlightColorMappingID(int inInt)
  {
    try{
      groupHighlightColorMappingID = inInt;
      trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC12");
    }
  }
  
  public int getGHCMID()
  {
    return groupHighlightColorMappingID;
  }
  //***
  //End - Group highlight color functions.
  //***
  
  //***
  //User Event Highlight Color functions (UEHC functions).
  //***
  public void setUEHC(Color inColor)
  {
    uEHC = inColor;
  }
  
  public Color getUEHC()
  {
    return uEHC;
  }
  
  public void setUEHCMappingID(int inInt)
  {
    try{
      uEHCMappingID = inInt;
      trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC13");
    }
  }
  
  public int getUEHCMappingID()
  {
    return uEHCMappingID;
  }
  //***
  //End - User Event Highlight Color functions (UEHC functions).
  //***
  
  
  //***
  //Misc. color functions.
  //***
  public void setMiscMappingsColor(Color inColor)
  {
    miscMappingsColor = inColor;
  }
  
  public Color getMiscMappingsColor()
  {
    return miscMappingsColor;
  }
  //***
  //End - Misc. color functions.
  //***
  
  //A function which sets the globalColors vector to be the default set.
  public void setDefaultColors()
  {
    try{
      //Clear the globalColors vector.
      globalColors.clear();
      
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
      ParaProf.systemError(e, null, "CC14");
    }
  }
  
  //A function which sets the globalColors vector to be the default set.
  public void setDefaultMappingGroupColors()
  {
    try{
      //Clear the globalColors vector.
      mappingGroupColors.clear();
      
      //Add the default colours.
      addMappingGroupColor(new Color(102,0,102));
      addMappingGroupColor(new Color(51,51,0));
      addMappingGroupColor(new Color(204,0,51));
      addMappingGroupColor(new Color(0,102,102));
      addMappingGroupColor(new Color(255,255,102));
      addMappingGroupColor(new Color(0,0,102));
      addMappingGroupColor(new Color(153,153,255));
      addMappingGroupColor(new Color(255,51,0));
      addMappingGroupColor(new Color(255,153,0));
      addMappingGroupColor(new Color(255,102,102));
      addMappingGroupColor(new Color(51,0,51));
      addMappingGroupColor(new Color(255,255,102));
    }
    catch(Exception e){
      ParaProf.systemError(e, null, "CC15");
    }
  }
  
  //Window Listener code.
  public void windowClosed(WindowEvent winevt){}
  public void windowIconified(WindowEvent winevt){}
  public void windowOpened(WindowEvent winevt){}
  public void windowClosing(WindowEvent winevt)
  {
    if(winevt.getSource() == clrChooserFrame)
    {
      clrChooserFrameShowing = false;
    }
  }
  public void windowDeiconified(WindowEvent winevt){}
  public void windowActivated(WindowEvent winevt){}
  public void windowDeactivated(WindowEvent winevt){}


  //Instance Data.
  private Trial trial = null;
  Vector globalColors = new Vector();
  Vector mappingGroupColors = new Vector();
  
  private Color highlightColor = Color.red;
  int highlightColorMappingID = -1;
  private Color groupHighlightColor = new Color(0,255,255);
  int groupHighlightColorMappingID = -1;
  private Color uEHC = new Color(255,255, 0);
  private int uEHCMappingID = -1;
  private Color miscMappingsColor = Color.black;
  private boolean clrChooserFrameShowing = false; //For determining whether the clrChooserFrame is showing.
  private ColorChooserFrame clrChooserFrame;
}
  
    
class ColorChooserFrame extends JFrame implements ActionListener
{ 
  public ColorChooserFrame(Trial inTrial, ColorChooser inColorChooser)
  {
    
    try{
      //Window Stuff.
      setLocation(new Point(100, 100));
      setSize(new Dimension(850, 450));
      
      trial = inTrial;
      colorChooserRef = inColorChooser;
      numberOfColors = trial.getColorChooser().getNumberOfColors();
      
      
      
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
      JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
      exitItem.addActionListener(this);
      fileMenu.add(exitItem);
      //******************************
      //End - File menu.
      //******************************
      
      //******************************
      //Help menu.
      //******************************
      /*JMenu helpMenu = new JMenu("Help");
      
      //Add a menu item.
      JMenuItem aboutItem = new JMenuItem("About Racy");
      helpMenu.add(aboutItem);
      
      //Add a menu item.
      JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
      showHelpWindowItem.addActionListener(this);
      helpMenu.add(showHelpWindowItem);*/
      //******************************
      //End - Help menu.
      //******************************
      
      
      //Now, add all the menus to the main menu.
      mainMenu.add(fileMenu);
      //mainMenu.add(helpMenu);
      
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
      Border raisedBev = BorderFactory.createRaisedBevelBorder();
      Border empty = BorderFactory.createEmptyBorder();
      
      
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
      sp.setBorder(raisedBev);
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
      addMappingGroupColorButton = new JButton("Add Mapping Gr. Color");
      addMappingGroupColorButton.addActionListener(this);
      addCompItem(addMappingGroupColorButton, gbc, 1, 2, 1, 1);
      
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
      
      //Now populate the colour list.
      populateColorList();
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "CCF01");
    }
  } 
  
  
  //ActionListener code.
  public void actionPerformed(ActionEvent evt)
  {
    Object EventSrc = evt.getSource();
    String arg = evt.getActionCommand();
    
    if(EventSrc instanceof JMenuItem)
    {
      if(arg.equals("Exit ParaProf!"))
      {
        setVisible(false);
        dispose();
        System.exit(0);
      }
      else if(arg.equals("Close This Window"))
      {
        setVisible(false);
      }
    }
    else if(EventSrc instanceof JButton)
    {
      if(arg.equals("Add Color"))
      {
        Color tmpColor = clrModel.getSelectedColor();
        
        //listModel.addElement(tmpColor);
        
        (colorChooserRef.getAllColors()).add(tmpColor);
        
        listModel.clear();
        populateColorList();
        
        //Update the GlobalMapping.
        GlobalMapping gMRef = trial.getGlobalMapping();
        gMRef.updateGenericColors(0);
        
        //Update the listeners.
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
      }
      else if(arg.equals("Add Mapping Gr. Color"))
      {
        Color tmpColor = clrModel.getSelectedColor();
        
        (colorChooserRef.getAllMappingGroupColors()).add(tmpColor);
        
        listModel.clear();
        populateColorList();
        
        //Update the GlobalMapping.
        GlobalMapping gMRef = trial.getGlobalMapping();
        gMRef.updateGenericColors(1);
        
        //Update the listeners.
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
      }
      else if(arg.equals("Delete Selected Color"))
      {
        //Get the currently selected items and cycle through them.
        int [] values = colorList.getSelectedIndices();
        for(int i = 0; i < values.length; i++)
        {
          if((values[i]) < trial.getColorChooser().getNumberOfColors())
          {
            System.out.println("The value being deleted is: " + values[i]);
            listModel.removeElementAt(values[i]);
            (colorChooserRef.getAllColors()).removeElementAt(values[i]);
            
            //Update the GlobalMapping.
            GlobalMapping gMRef = trial.getGlobalMapping();
            gMRef.updateGenericColors(0);
          }
          else if((values[i]) < (trial.getColorChooser().getNumberOfColors()) + (trial.getColorChooser().getNumberOfMappingGroupColors()))
          {
            System.out.println("The value being deleted is: " + values[i]);
            listModel.removeElementAt(values[i]);
            (colorChooserRef.getAllMappingGroupColors()).removeElementAt(values[i] - (trial.getColorChooser().getNumberOfColors()));
            
            //Update the GlobalMapping.
            GlobalMapping gMRef = trial.getGlobalMapping();
            gMRef.updateGenericColors(1);
          }
        }
        
        //Update the listeners.
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
      }
      else if(arg.equals("Update Selected Color"))
      {
        Color tmpColor = clrModel.getSelectedColor();
        
        //Get the currently selected items and cycle through them.
        int [] values = colorList.getSelectedIndices();
        for(int i = 0; i < values.length; i++)
        {
          listModel.setElementAt(tmpColor, values[i]);
          
          int totalNumberOfColors = (trial.getColorChooser().getNumberOfColors()) + (trial.getColorChooser().getNumberOfMappingGroupColors());
          
          if((values[i]) == (totalNumberOfColors)){
            trial.getColorChooser().setHighlightColor(tmpColor);
          }
          else if((values[i]) == (totalNumberOfColors+1)){
            trial.getColorChooser().setGroupHighlightColor(tmpColor);
          }
          else if((values[i]) == (totalNumberOfColors+2)){
            trial.getColorChooser().setUEHC(tmpColor);
          }
          else if((values[i]) == (totalNumberOfColors+3)){
            trial.getColorChooser().setMiscMappingsColor(tmpColor);
          }
          else if((values[i]) < trial.getColorChooser().getNumberOfColors())
          {
            colorChooserRef.setColorInLocation(tmpColor, values[i]);
            
            //Update the GlobalMapping.
            GlobalMapping gMRef = trial.getGlobalMapping();
            gMRef.updateGenericColors(0);
          }
          else
          {
            colorChooserRef.setMappingGroupColorInLocation(tmpColor, (values[i] - trial.getColorChooser().getNumberOfColors()));
            
            //Update the GlobalMapping.
            GlobalMapping gMRef = trial.getGlobalMapping();
            gMRef.updateGenericColors(1);
          }
        }
        
        //Update the listeners.
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
      }
      else if(arg.equals("Restore Defaults"))
      {
        colorChooserRef.setDefaultColors();
        colorChooserRef.setDefaultMappingGroupColors();
        colorChooserRef.setHighlightColor(Color.red);
        colorChooserRef.setGroupHighlightColor(new Color(0,255,255));
        colorChooserRef.setUEHC(new Color(255,255,0));
        colorChooserRef.setMiscMappingsColor(Color.black);
        listModel.clear();
        populateColorList();
        
        //Update the GlobalMapping.
        GlobalMapping gMRef = trial.getGlobalMapping();
        gMRef.updateGenericColors(0);
        gMRef.updateGenericColors(1);
        
        //Update the listeners.
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
      }
    }
  }
  
  private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h)
  {
    gbc.gridx = x;
    gbc.gridy = y;
    gbc.gridwidth = w;
    gbc.gridheight = h;
    
    getContentPane().add(c, gbc);
  }
  
  void populateColorList()
  {
    Color tmpColor;
    
    for(Enumeration e = (colorChooserRef.getAllColors()).elements(); e.hasMoreElements() ;)
    {
      tmpColor = (Color) e.nextElement();
      listModel.addElement(tmpColor);
    }
    
    for(Enumeration e = (colorChooserRef.getAllMappingGroupColors()).elements(); e.hasMoreElements() ;)
    {
      tmpColor = (Color) e.nextElement();
      listModel.addElement(tmpColor);
    }
    
    tmpColor = trial.getColorChooser().getHighlightColor();
    listModel.addElement(tmpColor);
    
    tmpColor = trial.getColorChooser().getGroupHighlightColor();
    listModel.addElement(tmpColor);
    
    tmpColor = trial.getColorChooser().getUEHC();
    listModel.addElement(tmpColor);
    
    tmpColor = trial.getColorChooser().getMiscMappingsColor();
    listModel.addElement(tmpColor);
  }
  
  //******************************
  //Instance data!
  //******************************
  private Trial trial = null;
  ColorChooser colorChooserRef;
  private ColorSelectionModel clrModel;
  JColorChooser clrChooser;
  DefaultListModel listModel;
  JList colorList;
  JButton addColorButton;
  JButton addMappingGroupColorButton;
  JButton deleteColorButton;
  JButton updateColorButton;
  JButton restoreDefaultsButton;
  
  int numberOfColors = -1;
  //******************************
  //End - Instance data!
  //******************************
  
}

class CustomCellRenderer implements ListCellRenderer
{
  CustomCellRenderer(Trial inTrial){
    trial = inTrial;
  }
  
  public Component getListCellRendererComponent(final JList list, final Object value,
                     final int index, final boolean isSelected,
                     final boolean cellHasFocus)
  {
    return new JPanel()
      {
        
        public void paintComponent(Graphics g)
        {
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
          else
          {
            g.drawString(("G" + (index - (trial.getColorChooser().getNumberOfColors()) + 1)), xStringPos1, yStringPos1);
          }
          
          g.setColor(inColor);
          g.fillRect((5 + maxXNumFontSize + 5), 0,50, ySize);
          
          //Just a sanity check.
          if((xSize - 50) > 0)
          {
            g.setColor(isSelected ? list.getSelectionBackground() : list.getBackground());
            g.fillRect((5 + maxXNumFontSize + 5 + 50), 0,(xSize - 50), ySize);
          }
          
          int xStringPos2 = 50 + (((xSize - 50) - thisXFontSize) / 2);
          int yStringPos2 = (ySize - 5);
          
          g.setColor(isSelected ? list.getSelectionForeground() : list.getForeground());
          g.drawString(tmpString2, xStringPos2, yStringPos2);
        }
        
        
        public Dimension getPreferredSize()
        {
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
  
  //Instance Data.
  private Trial trial = null;
}
