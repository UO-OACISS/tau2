/*
  HelpWindow.java
  
  
  Title:      ParaProf
  Author:     Robert Ansell-Bell
  Description:  This class provides detailed help information for the user.
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.border.*;
import javax.swing.event.*;
import edu.uoregon.tau.dms.dss.*;

public class HelpWindow extends JFrame implements ActionListener, Observer{

  //*****
  //Instance data.
  //*****
  
  //General.
  int windowWidth = 750;
  int windowHeight = 500;
  
  //Text area stuff.
  JTextArea helpJTextArea;
  Document helpJTextAreaDocument;
  
  //*****
  //End - Instance data.
  //*****
  
  public HelpWindow(boolean debug){
      this.debug = debug;

    //Set the preferend initial size for this window.
    setSize(new java.awt.Dimension(windowWidth, windowHeight));
    setTitle("ParaProf Help Window");
    
    //******************************
    //Code to generate the menus.
    //******************************
    JMenuBar mainMenu = new JMenuBar();
    
    //******************************
    //File menu.
    //******************************
    JMenu fileMenu = new JMenu("File");
    
    //Add a menu item.
    JMenuItem generalHelpItem = new JMenuItem("Display General Help");
    generalHelpItem.addActionListener(this);
    fileMenu.add(generalHelpItem);
    
    //Add a menu item.
    JMenuItem closeItem = new JMenuItem("Close ParaProf Help Window");
    closeItem.addActionListener(this);
    fileMenu.add(closeItem);
    
    
    //Add a menu item.
    JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
    exitItem.addActionListener(this);
    fileMenu.add(exitItem);
    //******************************
    //End - File menu.
    //******************************
    
    //Now, add all the menus to the main menu.
    mainMenu.add(fileMenu);
    
    setJMenuBar(mainMenu);
    //******************************
    //End - Code to generate the menus.
    //******************************
    
    //Create the text area and get its document.
    helpJTextArea = new JTextArea();
    helpJTextArea.setLineWrap(true);
    helpJTextArea.setWrapStyleWord(true);
    helpJTextArea.setSize(new java.awt.Dimension(windowWidth, windowHeight));
    helpJTextAreaDocument = helpJTextArea.getDocument();
    
    //Setting up the layout system for the main window.
    Container contentPane = getContentPane();
    GridBagLayout gbl = new GridBagLayout();
    contentPane.setLayout(gbl);
    GridBagConstraints gbc = new GridBagConstraints();
    gbc.insets = new Insets(5, 5, 5, 5);
    
    //Create a borders.
    Border mainloweredbev = BorderFactory.createLoweredBevelBorder();
    
    //The scroll panes into which the list shall be placed.
    JScrollPane scrollPane = new JScrollPane(helpJTextArea);
    scrollPane.setBorder(mainloweredbev);
    scrollPane.setPreferredSize(new Dimension(windowWidth, windowHeight));
    
    //Add the componants.
    gbc.fill = GridBagConstraints.BOTH;
    gbc.anchor = GridBagConstraints.CENTER;
    gbc.weightx = 1;
    gbc.weighty = 1;
    addCompItem(scrollPane, gbc, 0, 0, 2, 1);
    
    writeText("Welcome to ParaProf!");
    writeText("");
    writeText("For general help, please select display general help from the file menu.");    
    
  }
  
  public void clearText()
  { 
    try
    {
      helpJTextAreaDocument.remove(0, helpJTextAreaDocument.getLength());
    }
    catch(Exception e)
    {
      System.out.println("There was a problem with the help window!");
      System.out.println(e.toString());
    }
  }
  
  public void writeText(String inText)
  {
    helpJTextArea.append(inText);
    helpJTextArea.append("\n");
  }
  
  //Observer functions.
  public void update(Observable o, Object arg)
  {
    String tmpString = (String) arg;
    if(tmpString.equals("subWindowCloseEvent"))
    {
      setVisible(false);
    }
  }
  
  //Helper function for adding componants.
  private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h)
  {
    gbc.gridx = x;
    gbc.gridy = y;
    gbc.gridwidth = w;
    gbc.gridheight = h;
    
    getContentPane().add(c, gbc);
  }
  
  //ActionListener code.
  public void actionPerformed(ActionEvent evt)
  {
    try{
      Object EventSrc = evt.getSource();
      
      if(EventSrc instanceof JMenuItem)
      {
        String arg = evt.getActionCommand();
        
        if(arg.equals("Display General Help"))
        {
          clearText();
          
          writeText("Welcome to ParaProf!");
          writeText("");
          writeText("More detailed information can be found in the ParaProfHelp.txt that is distributed with TAU."
          + " What follows is a brief summary of the new features.");
          writeText("");
          writeText("1) Added new group tracking in ParaProf:");
          writeText("pprof has been updated to track group names, and ParaProf has also been updated to take"
          + " advantage of this.  At present, a new group ledger window has been added to display groups."
          + " Clicking on one of the groups in this window will cause all members of the group to be"
          + " highlighted in all open windows, except for the total stat windows."
          + " This is only the first stage of group capabilities, but it has proved more time consuming"
          + " than expected to add all the features.  Much more to follow!");
          writeText("");
          writeText("2) Add the ability to save color maps between sessions:");
          writeText("When you save a color map, a text file is created with the current map." 
          + " Loading a map causes the system to try and match the mapping names it know about"
          + " with names in the selected color map file. If it finds a match, it updates the color."
          + " This means that you do not need to have exactly the same mappings in the system as when"
          + " the map was saved.");
          writeText("");
          writeText("3) Sliders have been added to all windows that have bars.");
          writeText("To display the sliders, select the display sliders option from the options menu"
          + " of any window containing bars.  This will display the sliders for that window.  The slider"
          + " options are as follows: A drop down list selecting a portion of the original length, and a bar"
          + " giving a multiple of this length.  This has proved useful in tailoring the exact lengths of"
          + " bars needed.");
          writeText("");
          writeText("4) Added more color selections for mapping group colors.");
          writeText("");
          writeText("5) Added full masking of groups.");
          writeText("");
          writeText("6) A number of other minor changes and bug fixes.  As always, this is an ongoing project" 
          + " as such, ANY suggestions are welcome.  There will also be a clean up of the code to provide" 
          + " more effective searching as some things have been implimented just to get it working.");
        }
        else if(arg.equals("Close ParaProf Help Window"))
        {
          setVisible(false);
        }
        else if(arg.equals("Exit ParaProf!"))
        {
          setVisible(false);
          dispose();
          System.exit(0);
        } 
      }
    }
    catch(Exception e)
    {
      UtilFncs.systemError(e, null, "HW01");
    }
  }

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //Instance data.
    //####################################
    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################
}
