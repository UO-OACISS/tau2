/*  
    PPMLWindow.java

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

public class  PPMLWindow extends JFrame implements ActionListener{ 
    
    public PPMLWindow(ParaProfManager paraProfManager){
	this.paraProfManager = paraProfManager;

	//####################################
	//Window Stuff.
	//####################################
	int windowWidth = 400;
	int windowHeight = 200;

	//Grab paraProfManager position and size.
	Point parentPosition = paraProfManager.getLocationOnScreen();
	Dimension parentSize = paraProfManager.getSize();
	int parentWidth = parentSize.width;
	int parentHeight = parentSize.height;
	
	//Set the window to come up in the center of the screen.
	int xPosition = (parentWidth - windowWidth) / 2;
	int yPosition = (parentHeight - windowHeight) / 2;

	xPosition = (int) parentPosition.getX() + xPosition;
	yPosition = (int) parentPosition.getY() + yPosition;

	this.setLocation(xPosition, yPosition);
	setSize(new java.awt.Dimension(windowWidth, windowHeight));
	setTitle("Database Configuration");
	//####################################
	//End -Window Stuff.
	//####################################
	
	//Add some window listener code
	addWindowListener(new java.awt.event.WindowAdapter() {
		public void windowClosing(java.awt.event.WindowEvent evt) {
		    thisWindowClosing(evt);
		}
	    });
	
	//####################################
	//Create and add the components.
	//####################################
	//Setting up the layout system for the main window.
	Container contentPane = getContentPane();
	GridBagLayout gbl = new GridBagLayout();
	contentPane.setLayout(gbl);
	GridBagConstraints gbc = new GridBagConstraints();
	gbc.insets = new Insets(5, 5, 5, 5);

 	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.WEST;
	gbc.weightx = 0;
	gbc.weighty = 0;
	addCompItem(new JLabel("Argument 1:"), gbc, 0, 0, 1, 1);

	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.WEST;
	gbc.weightx = 100;
	gbc.weighty = 0;
	addCompItem(arg1Field, gbc, 1, 0, 2, 1);

	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.WEST;
	gbc.weightx = 0;
	gbc.weighty = 0;
	addCompItem(new JLabel("Argument 2:"), gbc, 0, 1, 1, 1);
	    
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.WEST;
	gbc.weightx = 100;
	gbc.weighty = 0;
	addCompItem(arg2Field, gbc, 1, 1, 2, 1);

	JButton jButton = new JButton("Cancel");
	jButton.addActionListener(this);
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.EAST;
	gbc.weightx = 0;
	gbc.weighty = 0;
	addCompItem(jButton, gbc, 0, 2, 1, 1);

	jButton.addActionListener(this);
	gbc.fill = GridBagConstraints.BOTH;
	gbc.anchor = GridBagConstraints.CENTER;
	gbc.weightx = 100;
	gbc.weighty = 0;
	addCompItem(operation, gbc, 1, 2, 1, 1);

	jButton = new JButton("Apply operation");
	jButton.addActionListener(this);
	gbc.fill = GridBagConstraints.NONE;
	gbc.anchor = GridBagConstraints.EAST;
	gbc.weightx = 0;
	gbc.weighty = 0;
	addCompItem(jButton, gbc, 2, 2, 1, 1);
	//####################################
	//End - Create and add the components.
	//####################################
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
	    String arg = evt.getActionCommand();
	    if(arg.equals("Cancel")){
		closeThisWindow();}
	    else if(arg.equals("Apply operation")){
		Metric metric = PPML.applyOperation(arg1Field.getText().trim(),
						    arg2Field.getText().trim(),
						    (String) operation.getSelectedItem());
		paraProfManager.insertMetric(metric);
		closeThisWindow();
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "DBC02");
	}
    }
    //######
    //End - ActionListener.
    //######

    //####################################
    //End - Interface code.
    //####################################

    //####################################
    //Pivate Section.
    //####################################
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	try{
	    gbc.gridx = x;
	    gbc.gridy = y;
	    gbc.gridwidth = w;
	    gbc.gridheight = h;
	    
	    getContentPane().add(c, gbc);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "DBC03");
	}
    }
    
    //Close the window when the close box is clicked
    private void thisWindowClosing(java.awt.event.WindowEvent e){
	closeThisWindow();}
    
    void closeThisWindow(){ 
	this.setVisible(false);
	dispose();
    }
    //####################################
    //End - Pivate Section.
    //####################################
    
    //####################################
    //Instance data.
    //####################################
    ParaProfManager paraProfManager = null;
    JTextField arg1Field = new JTextField("Argument 1 (x:x:x:x)", 15);
    JTextField arg2Field = new JTextField("Argument 2 (x:x:x:x)", 15);
    String operationStrings[] = {"Add", "Subtract", "Multiply", "Divide"};
    JComboBox operation = new JComboBox(operationStrings);
    //####################################
    //End - Instance data.
    //#################################### 
}
