/*  
  LocalPrefWindow.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package ParaProf;

import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

public class LocalPrefWindow extends JFrame implements ActionListener, Observer
{ 
  //******************************
  //Instance data.
  //******************************
  private Trial trial = null;
  private ThreadDataWindow tDWindow;
    
  //References for some of the components for this frame.
  private PrefSpacingPanel pSPanel;
  
  private JCheckBox loadPprofDat;
  
  
  private JRadioButton normal;
  private JRadioButton bold;
  private JRadioButton italic;
  
  private ButtonGroup buttonGroup;
  
  private JLabel fontLabel = new JLabel("Font Selection");
  
  private JComboBox fontComboBox;
  
  private JLabel barSpacingLabel = new JLabel("Adjust Bar Spacing");
  private JLabel barHeightLabel = new JLabel("Adjust Bar Height");
  
  private JSlider barSpacingSlider = new JSlider(SwingConstants.VERTICAL, 0, 100, 0);
  private JSlider barHeightSlider = new JSlider(SwingConstants.VERTICAL, 0, 100, 0);
  
  private JButton colorButton;
  
  int fontStyle;
  
  private boolean barDetailsSet = false;
  private int barSpacing = -1;
  private int barHeight = -1;
  
  
  private int numberPrecision = 5;
  private JButton numberPrecisionButton = null;
  
  String ParaProfFont;
  
  //Whether we are doing inclusive or exclusive.
  String inExValue;
  
  //Variable to determine which sorting paradigm has been chosen.
  String sortBy;  //Possible values are:
          //mappingID
          //millDes
          //millAsc
  
  //******************************
  //End - Instance data.
  //******************************
  public LocalPrefWindow(Trial inTrial, ThreadDataWindow inTDWindow)
  { 
    
    try{
      trial = inTrial;
      tDWindow = inTDWindow;
      
      ParaProfFont = trial.getPreferences().getJRacyFont();
      barSpacing = trial.getPreferences().getBarSpacing();
      barHeight = trial.getPreferences().getBarHeight();
      fontStyle = trial.getPreferences().getFontStyle();
      
      //Add some window listener code
      addWindowListener(new java.awt.event.WindowAdapter() {
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
      while((counter < tmpInt) && (!(ParaProfFont.equals(tmpString))))
      {
        counter++;
        tmpString = (String) fontComboBox.getItemAt(counter);
      }
      
      if(counter == tmpInt)
      {
        //The default font was not available.  Indicate an error.
        System.out.println("The default font was not found!!  This is not a good thing as it is a default Java font!!");
      }
      else
      {
        fontComboBox.setSelectedIndex(counter);
      }
      
      //Set the sliders.
      barHeightSlider.setValue(barHeight);
      barSpacingSlider.setValue(barSpacing);
      
      fontComboBox.addActionListener(this);
      
      //Now initialize the panels.
      pSPanel = new PrefSpacingPanel(trial);
      
      //Window Stuff.
      setTitle("ParaProf Preferences: No Data Loaded");
      
      int windowWidth = 900;
      int windowHeight = 350;
      setSize(new java.awt.Dimension(windowWidth, windowHeight));
      
      //There is really no need to resize this window.
      setResizable(false);
      
      
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

      //******************************
      //Code to generate the menus.
      //******************************
      
      
      JMenuBar mainMenu = new JMenuBar();
      
      //******************************
      //File menu.
      //******************************
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
      JMenuItem closeItem = new JMenuItem("Close Window");
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
      Border mainloweredbev = BorderFactory.createLoweredBevelBorder();
      Border mainraisedbev = BorderFactory.createRaisedBevelBorder();
      Border mainempty = BorderFactory.createEmptyBorder();
      
      //**********
      //Panel and ScrollPane definition.
      //**********
      JScrollPane scrollPaneS = new JScrollPane(pSPanel);
      scrollPaneS.setBorder(mainloweredbev);
      scrollPaneS.setPreferredSize(new Dimension(200, 200));
      //**********
      //End - Panel and ScrollPane definition.
      //**********
      
      //**********
      //Slider Setup
      //**********
      barSpacingSlider.setPaintTicks(true);
      barSpacingSlider.setMajorTickSpacing(20);
      barSpacingSlider.setMinorTickSpacing(5);
      barSpacingSlider.setPaintLabels(true);
      barSpacingSlider.addChangeListener(pSPanel);
      
      barHeightSlider.setPaintTicks(true);
      barHeightSlider.setMajorTickSpacing(20);
      barHeightSlider.setMinorTickSpacing(5);
      barHeightSlider.setPaintLabels(true);
      barHeightSlider.addChangeListener(pSPanel);
      //**********
      //End - Slider Setup
      //**********
      
      //**********
      //RadioButton and ButtonGroup Setup
      //**********
      normal = new JRadioButton("Plain Font", ((fontStyle == Font.PLAIN) || (fontStyle == (Font.PLAIN|Font.ITALIC))));
      normal.addActionListener(this);
      bold = new JRadioButton("Bold Font", ((fontStyle == Font.BOLD) || (fontStyle == (Font.BOLD|Font.ITALIC))));
      bold.addActionListener(this);
      italic = new JRadioButton("Italic Font", ((fontStyle == (Font.PLAIN|Font.ITALIC)) || (fontStyle == (Font.BOLD|Font.ITALIC))));
      italic.addActionListener(this);
      
      buttonGroup = new ButtonGroup();
      buttonGroup.add(normal);
      buttonGroup.add(bold);
      //**********
      //End - RadioButton and ButtonGroup Setup
      //**********
      
      //**********
      //Button Setup
      //**********
      numberPrecisionButton = new JButton("Select Number Precision");
      numberPrecisionButton.addActionListener(this);
      //**********
      //End - Button Setup
      //**********
      
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
      
      gbc.fill = GridBagConstraints.NONE;
      gbc.anchor = GridBagConstraints.CENTER;
      gbc.weightx = 1;
      gbc.weighty = 1;
      addCompItem(numberPrecisionButton, gbc, 5, 0, 1, 1);
      
      gbc.fill = GridBagConstraints.BOTH;
      gbc.anchor = GridBagConstraints.CENTER;
      gbc.weightx = 1;
      gbc.weighty = 1;
      addCompItem(scrollPaneS, gbc, 0, 2, 2, 2);
      
      gbc.fill = GridBagConstraints.NONE;
      gbc.anchor = GridBagConstraints.NORTH;
      gbc.weightx = 1;
      gbc.weighty = 1;
      addCompItem(barSpacingLabel, gbc, 2, 2, 1, 1);
      
      gbc.fill = GridBagConstraints.BOTH;
      gbc.anchor = GridBagConstraints.NORTH;
      gbc.weightx = 1;
      gbc.weighty = 1;
      addCompItem(barSpacingSlider, gbc, 2, 3, 1, 1);
      
      gbc.fill = GridBagConstraints.NONE;
      gbc.anchor = GridBagConstraints.NORTH;
      gbc.weightx = 1;
      gbc.weighty = 1;
      addCompItem(barHeightLabel, gbc, 3, 2, 1, 1);
      
      gbc.fill = GridBagConstraints.BOTH;
      gbc.anchor = GridBagConstraints.NORTH;
      gbc.weightx = 1;
      gbc.weighty = 1;
      addCompItem(barHeightSlider, gbc, 3, 3, 1, 1);
      //******************************
      //End - Create and add the componants.
      //******************************
    }
    catch(Exception e){
      System.out.println(e);
      ParaProf.systemError(e, null, "LPW01");
    }
  }
  
  public void showPreferencesWindow()
  {
    //The path to data might have changed, therefore, reset the title.
    this.setTitle("ParaProf Preferences: " + ParaProf.profilePathName);
    this.show();
  }
  
  public void setSavedPreferences()
  {
    ParaProf.savedPreferences.setJRacyFont(ParaProfFont);
    ParaProf.savedPreferences.setBarSpacing(barSpacing);
    ParaProf.savedPreferences.setBarHeight(barHeight);
    ParaProf.savedPreferences.setInExValue(inExValue);
    ParaProf.savedPreferences.setSortBy(sortBy);
    ParaProf.savedPreferences.setFontStyle(fontStyle);
    ParaProf.savedPreferences.setBarDetailsSet(barDetailsSet);
    
  }
  
  public int getNumberPrecision(){
    return numberPrecision;
  }
    
  
  public boolean areBarDetailsSet()
  {
    return barDetailsSet;
  }
  
  public String getJRacyFont()
  {
    return ParaProfFont;
  }
  
  public int getFontStyle()
  {
    return fontStyle;
  }
  
  public void setBarDetails(int inBarHeight, int inBarSpacing)
  {
    barHeight = inBarHeight;
    barSpacing = inBarSpacing;
    
    barDetailsSet = true;
  }
  
  public void setSliders(int inBarHeight, int inBarSpacing)
  {
    //Set the slider values.
    barHeightSlider.setValue(inBarHeight);
    barSpacingSlider.setValue(inBarSpacing);
  }
  
  public int getBarSpacing()
  {
    return barSpacing;
  }
  
  public int getBarHeight()
  {
    return barHeight;
  }
  
  //Setting and returning the inExValue.
  public void setInExValue(String inString)
  {
    inExValue = inString;
  }
  
  public String getInExValue()
  {
    return inExValue;
  }
  
  //Setting and returning sortBy.
  public void setSortBy(String inString)
  {
    sortBy = inString;
  }
  
  public String getSortBy()
  {
    return sortBy;
  }
  
  //******************************
  //Event listener code!!
  //******************************
  
  //Observer functions.
  public void update(Observable o, Object arg)
  {
    String tmpString = (String) arg;
    
    if(tmpString.equals("colorEvent"))
    {     
      //Just need to call a repaint.
      pSPanel.repaint();
    }
  }
  
  //ActionListener code.
  public void actionPerformed(ActionEvent evt)
  {
    Object EventSrc = evt.getSource();
    String arg = evt.getActionCommand();
    
    if(EventSrc instanceof JMenuItem)
    {
      
      if(arg.equals("Edit Color Map"))
      {
        trial.getColorChooser().showColorChooser();
      }
      else if(arg.equals("Load Color Map"))
      {
        JFileChooser fileChooser = new JFileChooser();
        
        //Set the directory to the current directory.
        fileChooser.setCurrentDirectory(new File("."));
        
        //Bring up the file chooser.
        int resultValue = fileChooser.showOpenDialog(this);
        
        if(resultValue == JFileChooser.APPROVE_OPTION)
        {
          //Try and get the file name.
          File file = fileChooser.getSelectedFile();
          
          //Test to see if valid.
          if(file != null)
          { 
            System.out.println("Loading color map ...");
            loadColorMap(file);
            trial.getSystemEvents().updateRegisteredObjects("prefEvent");
            System.out.println("Done loading color map ...");
          }
          else
          {
            System.out.println("There was some sort of internal error!");
          }
        }
        
      }
      else if(arg.equals("Save Color Map"))
      {
        JFileChooser fileChooser = new JFileChooser();
        
        //Set the directory to the current directory.
        fileChooser.setCurrentDirectory(new File("."));
        fileChooser.setSelectedFile(new File("colorMap.dat"));
        
        //Display the save file chooser.
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
              //Just output the data for the moment to have a look at it.
              Vector nameColorVector = new Vector();
              GlobalMapping tmpGlobalMapping = trial.getGlobalMapping();
              
              int numOfMappings = tmpGlobalMapping.getNumberOfMappings(0);
              
              for(int i=0; i<numOfMappings; i++)
              {
                GlobalMappingElement tmpGME = (GlobalMappingElement) tmpGlobalMapping.getGlobalMappingElement(i,0);
                if((tmpGME.getMappingName()) != null)
                { 
                  ColorPair tmpCP = new ColorPair(tmpGME.getMappingName(),tmpGME.getMappingColor());
                  nameColorVector.add(tmpCP);
                }
              }
              Collections.sort(nameColorVector);
              
              
              PrintWriter out = new PrintWriter(new FileWriter(file));
              
              System.out.println("Saving color map ...");
              if(ParaProf.debugIsOn)
              {
                System.out.println("**********************");
                System.out.println("Color values loaded were:");
              }
              for(Enumeration e1 = nameColorVector.elements(); e1.hasMoreElements() ;)
              {
                ColorPair tmpCP = (ColorPair) e1.nextElement();
                Color tmpColor = tmpCP.getMappingColor();
                if(ParaProf.debugIsOn)
                {
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
              if(ParaProf.debugIsOn)
              {
                System.out.println("**********************");
              }
              System.out.println("Done saving color map!");
              out.close();
            }
            catch(Exception e)
            {
              //Display an error
              JOptionPane.showMessageDialog(this, "An error occured whilst trying to save the color map.", "Error!"
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
      else if(arg.equals("Exit ParaProf!"))
      {
        setVisible(false);
        dispose();
        System.exit(0);
      }
      else if(arg.equals("Close Window"))
      {
        setVisible(false);
        //trial.getSystemEvents().updateRegisteredObjects("prefEvent");
      }
    }
    else if(EventSrc instanceof JRadioButton)
    {
      if(arg.equals("Plain Font"))
      {
        if(italic.isSelected())
          fontStyle = Font.PLAIN|Font.ITALIC;
        else
          fontStyle = Font.PLAIN;
        
        pSPanel.repaint();
      }
      else if(arg.equals("Bold Font"))
      {
        if(italic.isSelected())
          fontStyle = Font.BOLD|Font.ITALIC;
        else
          fontStyle = Font.BOLD;
        
        pSPanel.repaint();
      }
      else if(arg.equals("Italic Font"))
      {
        if(italic.isSelected())
        {
          if(normal.isSelected())
            fontStyle = Font.PLAIN|Font.ITALIC;
          else
            fontStyle = Font.BOLD|Font.ITALIC;
        }
        else
        {
          if(normal.isSelected())
            fontStyle = Font.PLAIN;
          else
            fontStyle = Font.BOLD;
        }
        
        pSPanel.repaint();
      }
    }
    else if(EventSrc instanceof JButton){
      if(arg.equals("Select Number Precision")){
        String tmpString = JOptionPane.showInputDialog(this, "Precision Input", "Please enter an integer.", JOptionPane.QUESTION_MESSAGE);
        if(tmpString != null){
          numberPrecision = Integer.parseInt(tmpString);
          tDWindow.refreshPanel();
        }
      }
    }
    else if(EventSrc == fontComboBox)
    {
      ParaProfFont = (String) fontComboBox.getSelectedItem();
      pSPanel.repaint();
    }
  }
  
  public void updateBarDetails()
  {
    barHeight = barHeightSlider.getValue();
    barSpacing = barSpacingSlider.getValue();
  }
    
  private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h)
  {
    gbc.gridx = x;
    gbc.gridy = y;
    gbc.gridwidth = w;
    gbc.gridheight = h;
    
    getContentPane().add(c, gbc);
  }
  
  //******************************
  //End - Event listener code!!
  //******************************
  
  public void loadColorMap(File inFile)
  {
    try
    {
      //First, get the file stuff.
      BufferedReader br = new BufferedReader(new FileReader(inFile));
      
      Vector nameColorVector = new Vector();
      String tmpString;
      int red = 0;
      int green = 0;
      int blue = 0;
      
      GlobalMapping tmpGlobalMapping = trial.getGlobalMapping(); 
      
      
      //Read in the file line by line!
      while((tmpString = br.readLine()) != null)
      { 
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
        
        StringTokenizer getMappingColorTokenizer = new StringTokenizer(tmpString, ",");
        
        tmpString = getMappingColorTokenizer.nextToken();
        red = Integer.parseInt(tmpString);
        
        tmpString = getMappingColorTokenizer.nextToken();
        green = Integer.parseInt(tmpString);
        
        tmpString = getMappingColorTokenizer.nextToken();
        blue = Integer.parseInt(tmpString);
        
        Color tmpColor = new Color(red,green,blue);
        tmpCP.setMappingColor(tmpColor);
        
        nameColorVector.add(tmpCP);
      }
      
      if(ParaProf.debugIsOn)
      {
        System.out.println("**********************");
        System.out.println("Color values loaded were:");
      }
      for(Enumeration e1 = nameColorVector.elements(); e1.hasMoreElements() ;)
      {
        ColorPair tmpCP = (ColorPair) e1.nextElement();
        int mappingID = tmpGlobalMapping.getMappingId(tmpCP.getMappingName(), 0);
        if(mappingID != -1)
        {
          GlobalMappingElement tmpGME = tmpGlobalMapping.getGlobalMappingElement(mappingID, 0);
          
          Color tmpColor = tmpCP.getMappingColor();
          tmpGME.setSpecificColor(tmpColor);
          tmpGME.setColorFlag(true);
          if(ParaProf.debugIsOn)
          {
            System.out.println("MAPPING_NAME=\"" + (tmpCP.getMappingName()) + "\"" +
                  " RGB=\"" +
                  tmpColor.getRed() +
                  "," + tmpColor.getGreen() +
                  "," + tmpColor.getBlue() + "\"");
          }
        }
      }
      if(ParaProf.debugIsOn)
      {
        System.out.println("**********************");
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "P01");
    }
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
  }

}
