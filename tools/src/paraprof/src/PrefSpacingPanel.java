/* 
  
  PrefSpacingPanel.java
  
  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;


public class PrefSpacingPanel extends JPanel implements ChangeListener
{
  //******************************
  //Instance data.
  //******************************
  
  private Trial trial = null;
  
  
  int xPanelSize = 600;
  int yPanelSize = 200;
  
  int barXStart = -1;
  int barXCoord = -1;
  int yCoord = -1;
  
  int barSpacing = -1;
  int barHeight = -1;
  
  Color tmpColor;
  
  public PrefSpacingPanel(Trial inTrial)
  { 
    trial = inTrial;
    
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
    if(!(trial.getPreferences().areBarDetailsSet()))
    {
      
      //Create font.
      Font font = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), 12);
      g.setFont(font);
      FontMetrics fmFont = g.getFontMetrics(font);
      
      //Set up the bar details.
      
      //Compute the font metrics.
      int maxFontAscent = fmFont.getAscent();
      int maxFontDescent = fmFont.getMaxDescent();
      
      int tmpInt = maxFontAscent + maxFontDescent;
      
      trial.getPreferences().setBarDetails(maxFontAscent, (tmpInt + 5));
      
      trial.getPreferences().setSliders(maxFontAscent, (tmpInt + 5));
    }
    
    //Set local spacing and bar heights.
    barSpacing = trial.getPreferences().getBarSpacing();
    barHeight = trial.getPreferences().getBarHeight();
    
    //Set up the yCoord.
    yCoord = 25 + barHeight;
    
    //Create font.
    Font font = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), barHeight);
    g.setFont(font);
    FontMetrics fmFont = g.getFontMetrics(font);
    
    //calculate the maximum string width.
    int maxStringWidth = 0;
    for(int k=0; k<3; k++)
    {
      String s1 = "n,c,t    0,0," + k;
      int tmpInt = fmFont.stringWidth(s1);
      
      if(maxStringWidth < tmpInt)
        maxStringWidth = tmpInt;
    }
    
    //Now set the start location of the bars.
    barXStart = maxStringWidth + 25;
    barXCoord = barXStart;
    
    
    for(int i=0; i<3; i++)
    {
      
      String s1 = "n,c,t    0,0," + i;
      int tmpStringWidth = fmFont.stringWidth(s1);
      g.drawString(s1, (barXStart - tmpStringWidth - 5), yCoord);
      
      //After the above check, do the usual drawing stuff.
      for(int j=0; j<3; j++)
      {
        tmpColor = trial.getColorChooser().getColorInLocation(j);
        g.setColor(tmpColor);
        g.fillRect(barXCoord, (yCoord - barHeight), 40, barHeight);
        barXCoord = barXCoord + 30;
      }
      
      barXCoord = barXStart;
      g.setColor(Color.black);
      yCoord = yCoord + (barSpacing);
    }
  }
  
  public Dimension getPreferredSize()
  {
    return new Dimension((xPanelSize + 10), (yPanelSize + 10));
  }
  
  public void stateChanged(ChangeEvent event)
  {
    trial.getPreferences().updateBarDetails();
    this.repaint();
  }
}
