/*
 * Created on Nov 7, 2003
 *
 */
package edu.uoregon.tau.viewer.GUI;


import java.awt.event.*;
import java.awt.*;
import javax.swing.*;

/**
 * @author lili
 *
 */
public class PlotDrawingWindow extends JFrame {

	
	public PlotDrawingWindow(String arg0) {
	
		super(arg0);
			
		// int winWidth = 600;
		// int winHeight = 550;	
		int winWidth = 310;
		int winHeight = 300;	
		setSize(new java.awt.Dimension(winWidth, winHeight));
			
		//	Add window listener 
		addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent evt) {
		 		setVisible(false);
				dispose();
			  }
		  });
      
		//get the screen size.
		Toolkit tk = Toolkit.getDefaultToolkit();
		Dimension screenSize = tk.getScreenSize();
		int screenHeight = screenSize.height;
		int screenWidth = screenSize.width;
      
		//Set the window location.
		int xPosition = (screenWidth - winWidth) / 2;
		int yPosition = (screenHeight - winHeight) / 2;      
		setLocation(xPosition, yPosition);			    			    
	}	
	

	
}
