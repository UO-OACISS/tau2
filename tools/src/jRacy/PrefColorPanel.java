/* 
	
	PrefColorPanel.java
	
	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;


public class PrefColorPanel extends JPanel implements ChangeListener
{
	//******************************
	//******************************
	//Instance data.
	//******************************
	
	private ExperimentRun expRun = null;
	
	int xPanelSize = 600;
	int yPanelSize = 200;
	
	int barXStart = -1;
	int barXCoord = -1;
	int yCoord = -1;
	
	int barSpacing = -1;
	int barHeight = -1;
	
	Color tmpColor;
	
	public PrefColorPanel(ExperimentRun inExpRun)
	{
		
		expRun = inExpRun; 
		
		setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
		setPreferredSize(new java.awt.Dimension(xPanelSize, yPanelSize));
		
		//Set the default tool tip for this panel.
		setBackground(Color.white);
		
		
		this.repaint();
	}
	
	public void paintComponent(Graphics g)
	{
	
		super.paintComponent(g);
		
		//Do the standard font and spacing stuff.
		if(!(expRun.getPreferences().areBarDetailsSet()))
		{
			
			//Create font.
			Font font = new Font(expRun.getPreferences().getJRacyFont(), Font.PLAIN, 12);
			g.setFont(font);
			FontMetrics fmFont = g.getFontMetrics(font);
			
			//Set up the bar details.
			
			//Compute the font metrics.
			int maxFontAscent = fmFont.getAscent();
			
			expRun.getPreferences().setBarDetails(maxFontAscent, maxFontAscent);
			
			expRun.getPreferences().setSliders(maxFontAscent, maxFontAscent);
		}
		
		//Set local spacing and bar heights.
		barSpacing = expRun.getPreferences().getBarSpacing();
		barHeight = expRun.getPreferences().getBarHeight();
		
		//Set up the yCoord.
		yCoord = 25 + barHeight;
		
		//Create font.
		Font font = new Font(expRun.getPreferences().getJRacyFont(), Font.PLAIN, barHeight);
		g.setFont(font);
		FontMetrics fmFont = g.getFontMetrics(font);
		
		//calculate the maximum string width.
		int stringWidth = 0;
		
		String s1 = "n,c,t    0,0,0";
		stringWidth = fmFont.stringWidth(s1);
		
		//Now set the start location of the bars.
		barXStart = stringWidth + 25;
		barXCoord = barXStart;
		
		g.drawString(s1, (barXStart - stringWidth - 5), yCoord);
		
		//After the above check, do the usual drawing stuff.
		for(int j=0; j<(expRun.getColorChooser().getNumberOfColors()); j++)
		{
			tmpColor = expRun.getColorChooser().getColorInLocation(j);
			g.setColor(tmpColor);
			g.fillRect(barXCoord, (yCoord - barHeight), 40, barHeight);
			barXCoord = barXCoord + 30;
		}
		
		barXCoord = barXStart;
		g.setColor(Color.black);
		yCoord = yCoord + (barSpacing);
		
		//Now draw the highlight mapping colour and the misc. mapping colour.
		int tmpInt = barXStart - stringWidth - 5;
		g.drawString("Highlight colour:", tmpInt, yCoord);
		stringWidth = fmFont.stringWidth("Highlight colour:");
		tmpInt = tmpInt + 5 + stringWidth;
		g.setColor(expRun.getColorChooser().getHighlightColor());
		g.fillRect(tmpInt, (yCoord - barHeight), 100, barHeight);
		g.setColor(Color.black);
		stringWidth = fmFont.stringWidth("Misc. colour:");
		tmpInt = tmpInt + 30 + 100;
		g.drawString("Misc. colour:", tmpInt, yCoord);
		tmpInt = tmpInt + 5 + stringWidth;
		g.setColor(expRun.getColorChooser().getMiscMappingsColor());
		g.fillRect(tmpInt, (yCoord - barHeight), 100, barHeight);
		
		
	}
	
	public Dimension getPreferredSize()
	{
		return new Dimension((xPanelSize + 10), (yPanelSize + 10));
	}
	
	public void stateChanged(ChangeEvent event)
	{
		this.repaint();
	}
}




