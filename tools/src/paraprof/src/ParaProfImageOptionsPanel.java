/*  
    ParaProfImageOptionsPanel.java

    Title:      ParaProf
    Author:     Robert Bell
    Description:  
*/

package paraprof;

import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

public class  ParaProfImageOptionsPanel extends JPanel{ 
    public ParaProfImageOptionsPanel(Component component){
	try{
	    this.component = component;
	    
	    //####################################
	    //Window Stuff.
	    //####################################
	    int windowWidth = 200;
	    int windowHeight = 500;
	    setSize(new java.awt.Dimension(windowWidth, windowHeight));
	    //####################################
	    //End - Window Stuff.
	    //####################################
	    
	    //####################################
	    //Create and add the components.
	    //####################################
	    //Setting up the layout system for the main window.
	    GridBagLayout gbl = new GridBagLayout();
	    this.setLayout(gbl);
	    GridBagConstraints gbc = new GridBagConstraints();
	    gbc.insets = new Insets(5, 5, 5, 5);
	    
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    addCompItem(fullScreen, gbc, 0, 0, 1, 1);
	    
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    addCompItem(prependHeader, gbc, 0, 1, 1, 1);
	    
	    gbc.fill = GridBagConstraints.NONE;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 0;
	    gbc.weighty = 0;
	    addCompItem(imageQualityLabel, gbc, 0, 2, 1, 1);
	    
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.WEST;
	    gbc.weightx = 100;
	    gbc.weighty = 0;
	    addCompItem(imageQuality, gbc, 1, 2, 1, 1);
	    //####################################
	    //End - Create and add the components.
	    //####################################
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "PPIOP01");
	}
    }

    public boolean isFullScreen(){
	return fullScreen.isSelected();}

    public boolean isPrependHeader(){
	return prependHeader.isSelected();}

    public float getImageQuality(){
	return Float.valueOf((String) imageQuality.getSelectedItem()).floatValue();}

    //####################################
    //Private Section.
    //####################################
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	try{
	    gbc.gridx = x;
	    gbc.gridy = y;
	    gbc.gridwidth = w;
	    gbc.gridheight = h;
	    
	    this.add(c, gbc);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "PPIOP02");
	}
    }
    
    //####################################
    //End - Private Section.
    //####################################
    
    //####################################
    //Instance data.
    //####################################
    Component component = null;
    JCheckBox fullScreen = new JCheckBox("Full Screen");
    JCheckBox prependHeader = new JCheckBox("Prepend Header");
    JLabel imageQualityLabel = new JLabel("Image Quality");
    String imageQualityStrings[] = {"1.0", "0.75", "0.5", "0.25"};
    JComboBox imageQuality = new JComboBox(imageQualityStrings);
    //####################################
    //End - Instance data.
    //#################################### 
}
