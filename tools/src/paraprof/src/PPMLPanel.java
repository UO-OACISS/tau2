/*  
    PPMLPanel.java

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

public class PPMLPanel extends JPanel implements ActionListener{ 
    
    public PPMLPanel(ParaProfManager paraProfManager){
	this.paraProfManager = paraProfManager;

	//####################################
	//Create and add the components.
	//####################################
	//Setting up the layout system for the main window.
	GridBagLayout gbl = new GridBagLayout();
	this.setLayout(gbl);
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
	    if(arg.equals("Apply operation")){
		Metric metric = PPML.applyOperation(arg1Field.getText().trim(),
						    arg2Field.getText().trim(),
						    (String) operation.getSelectedItem());
		paraProfManager.insertMetric(metric);
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "DBC02");
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
	    
	    this.add(c, gbc);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "DBC03");
	}
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
