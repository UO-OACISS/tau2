/*
	HelpWindow.java
	
	
	Title:			jRacy
	Author:			Robert Ansell-Bell
	Description:	This class provides detailed help information for the user.
*/

package jRacy;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.border.*;
import javax.swing.event.*;

public class HelpWindow extends JFrame implements Observer
{

	//*****
	//Instance data.
	//*****
	
	//General.
	int windowWidth = 500;
	int windowHeight = 500;
	
	//Text area stuff.
	JTextArea helpJTextArea;
	Document helpJTextAreaDocument;
	
	//*****
	//End - Instance data.
	//*****
	
	public HelpWindow()
	{
		//Set the preferend initial size for this window.
		setSize(new java.awt.Dimension(windowWidth, windowHeight));
		setTitle("Racy Help Window");
		
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
}