/* 
  
PrefSpacingPanel.java
  
Title:      ParaProf
Author:     Robert Bell
Description:  
*/

package edu.uoregon.tau.paraprof;

import java.awt.*;
import javax.swing.*;
import javax.swing.event.*;


public class PrefSpacingPanel extends JPanel implements ChangeListener{
      
    public PrefSpacingPanel(ParaProfTrial trial){ 
	this.trial = trial;
 	setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	setPreferredSize(new java.awt.Dimension(xPanelSize, yPanelSize));
 	//Set the default tool tip for this panel.
	setBackground(Color.white);
 	this.repaint();
     }
  
    public void paintComponent(Graphics g){
 	super.paintComponent(g);

	Graphics2D g2D = (Graphics2D)g;

	//To make sure the bar details are set, this
	//method must be called.
	trial.getPreferences().setBarDetails(g2D);

	//Set local spacing and bar heights.
	barSpacing = trial.getPreferences().getBarSpacing();
	barHeight = trial.getPreferences().getBarHeight();
    
	//Set up the yCoord.
	yCoord = 25 + barHeight;
    
	//Create font.
	Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(), barHeight);
	g2D.setFont(font);
	FontMetrics fmFont = g2D.getFontMetrics(font);
    
	//calculate the maximum string width.
	int maxStringWidth = 0;
	for(int k=0; k<3; k++){
	    String s1 = "n,c,t    0,0," + k;
	    int tmpInt = fmFont.stringWidth(s1);
	    
	    if(maxStringWidth < tmpInt)
		maxStringWidth = tmpInt;
	}
	
	//Now set the start location of the bars.
	barXStart = maxStringWidth + 25;
	barXCoord = barXStart;
	
	
	for(int i=0; i<3; i++){
	    String s1 = "n,c,t    0,0," + i;
	    int tmpStringWidth = fmFont.stringWidth(s1);
	    g2D.drawString(s1, (barXStart - tmpStringWidth - 5), yCoord);
	    
	    //After the above check, do the usual drawing stuff.
	    for(int j=0; j<3; j++){
		tmpColor = trial.getColorChooser().getColor(j);
		g2D.setColor(tmpColor);
		g2D.fillRect(barXCoord, (yCoord - barHeight), 40, barHeight);
		barXCoord = barXCoord + 30;
	    }
	    
	    barXCoord = barXStart;
	    g2D.setColor(Color.black);
	    yCoord = yCoord + (barSpacing);
	}
    }
    
    public Dimension getPreferredSize(){
	return new Dimension((xPanelSize + 10), (yPanelSize + 10));}
  
    //####################################
    //Interface code.
    //####################################
    
    //######
    //ChangeListener.
    //######
    public void stateChanged(ChangeEvent event){
	trial.getPreferences().updateFontSize();
	this.repaint();
    }
    //######
    //End - ChangeListener.
    //######

    //####################################
    //End - Interface code.
    //####################################

    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
  
    int xPanelSize = 600;
    int yPanelSize = 200;
  
    int barXStart = -1;
    int barXCoord = -1;
    int yCoord = -1;
  
    int barSpacing = -1;
    int barHeight = -1;
  
    Color tmpColor;
    //####################################
    //End - Instance data.
    //####################################
}
