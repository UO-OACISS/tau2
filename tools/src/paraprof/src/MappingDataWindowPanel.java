/* 
  
  MappingDataWindowPanel.java
  
  Title:      ParaProf
  Author:     Robert Bell
  Description:
  
  Things to do:
  
  1) Add clipping support to this window. 
*/

package ParaProf;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import java.awt.geom.*;


public class MappingDataWindowPanel extends JPanel implements ActionListener, MouseListener
{
  int xPanelSize = 550;
  int yPanelSize = 550;
  
  public MappingDataWindowPanel()
  {
    try{
      setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
      
      //Schedule a repaint of this panel.
      this.repaint();
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MDWP01");
    }
  
  }
  
  
  public MappingDataWindowPanel(Trial inTrial, int inMappingID, MappingDataWindow inMDWindow)
  {
    try{
      setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
      setBackground(Color.white);
      
      //Add this object as a mouse listener.
      addMouseListener(this);
      
      trial = inTrial;
      
      
      //Grab the appropriate global mapping element.
      GlobalMapping tmpGM = trial.getGlobalMapping();
      GlobalMappingElement tmpGME = tmpGM.getGlobalMappingElement(inMappingID, 0);
      
      mappingName = tmpGME.getMappingName();
      mDWindow = inMDWindow;
      
      //Add items to the popu menu.
      JMenuItem changeColorItem = new JMenuItem("Change Function Color");
      changeColorItem.addActionListener(this);
      popup.add(changeColorItem);
      
      JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
      maskMappingItem.addActionListener(this);
      popup.add(maskMappingItem);
      
      //Schedule a repaint of this panel.
      this.repaint();
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MDWP02");
    }
  
  }
  

  public void paintComponent(Graphics g)
  {
    try
    {
      super.paintComponent(g);
      
      Graphics2D g2 = (Graphics2D) g;
      
      AffineTransform currentTransform = g2.getTransform();
      
      System.out.println("The x transform is: " + mDWindow.getHorizontalTranslate());
      
      currentTransform.translate(-(mDWindow.getHorizontalTranslate()), 0.0);
      g2.setTransform(currentTransform);
      
      Rectangle clipRect = g.getClipBounds();
      
      int yBeg = (int) clipRect.getY();
      int yEnd = (int) (yBeg + clipRect.getHeight());
      yEnd = yEnd + barSpacing;
      
      System.out.println("Clipping coords are: " + yBeg + ":" + yEnd);
      
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
      
      //Reset the font to the correct height.
      Font nameMetricFont = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), 12);
      FontMetrics fmFont = g.getFontMetrics(nameMetricFont);
      //Compute the font metrics for this font.
      int nameMetricHeight = fmFont.getAscent() + fmFont.getMaxDescent();
      
      //Create font.
      Font font = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), barHeight);
      g.setFont(font);
      fmFont = g.getFontMetrics(font);

      int defaultNumberPrecision = ParaProf.defaultNumberPrecision;
      double tmpSum;
      double tmpDataValue;
      Color tmpColor;
      String tmpString;
      int stringWidth;
      int stringStart;
      
      //Convenient counters.
      int colorCounter = 0;
    
      int yCoord = (2*nameMetricHeight)+14;
      int tmpXWidthCalc = 0;
      
      //An XCoord used in drawing the bars.
      int barXCoord = defaultBarLength + 60;
      
      //Grab the appropriate global mapping element.
      GlobalMapping tmpGM = trial.getGlobalMapping();
      GlobalMappingElement tmpGME = tmpGM.getGlobalMappingElement(mappingName, 0);
      
      mappingID = tmpGME.getGlobalID();
      
      //Set the max values for this mapping.
      maxInclusiveValue = tmpGME.getMaxInclusiveValue(trial.getCurValLoc());
      maxExclusiveValue = tmpGME.getMaxExclusiveValue(trial.getCurValLoc());
      maxInclusivePercentValue = tmpGME.getMaxInclusivePercentValue(trial.getCurValLoc());
      maxExclusivePercentValue = tmpGME.getMaxExclusivePercentValue(trial.getCurValLoc());
      maxNumberOfCalls = tmpGME.getMaxNumberOfCalls();
      maxNumberOfSubroutines = tmpGME.getMaxNumberOfSubRoutines();
      maxUserSecPerCall = tmpGME.getMaxUserSecPerCall(trial.getCurValLoc());
      
      //**********
      //Draw the counter name if required.
      //counterName = trial.getCounterName();
      //if(counterName != null){
      //  g.drawString("COUNTER NAME: " + counterName, 5, yBeg + 16);
      //  yCoord = yCoord + (16);
      //}
      //End - Draw the counter name if required.
      //**********
      
      //**********
      //Draw the mapping name.
      //g.drawString("FUNCTION NAME: " + mappingName, 5, yBeg + 32);
      //Calculate its width.
      //tmpXWidthCalc = fmFont.stringWidth(mappingName);
      //yCoord = yCoord + (barSpacing);
      //End - Draw the mapping name.
      //**********
      
      //Get some string lengths.
      if((mDWindow.isInclusive())){
          if(mDWindow.isPercent()){
            //Need to figure out how long the percentage string will be.
            stringWidth = fmFont.stringWidth(UtilFncs.getTestString(maxInclusivePercentValue, defaultNumberPrecision) + "%");
            barXCoord = barXCoord + stringWidth;
          }
          else{
            //Check to see what the units are.
            if((mDWindow.units()).equals("Seconds")){
              stringWidth = fmFont.stringWidth(UtilFncs.getTestString((maxInclusiveValue/1000000), defaultNumberPrecision));
              barXCoord = barXCoord + stringWidth;
            }
            else if((mDWindow.units()).equals("Milliseconds")){
              stringWidth = fmFont.stringWidth(UtilFncs.getTestString((maxInclusiveValue/1000), defaultNumberPrecision));
              barXCoord = barXCoord + stringWidth;
            }
            else{
              stringWidth = fmFont.stringWidth(UtilFncs.getTestString(maxInclusiveValue, defaultNumberPrecision));
              barXCoord = barXCoord + stringWidth;
            }
          }
        }
        else{
          if(mDWindow.isPercent()){
            //Need to figure out how long the percentage string will be.
            stringWidth = fmFont.stringWidth(UtilFncs.getTestString(maxExclusivePercentValue, defaultNumberPrecision) + "%");
            barXCoord = barXCoord + stringWidth;
          }
          else{
            //Check to see what the units are.
            if((mDWindow.units()).equals("Seconds")){
              stringWidth = fmFont.stringWidth(UtilFncs.getTestString((maxExclusiveValue/1000000), defaultNumberPrecision));
              barXCoord = barXCoord + stringWidth;
            }
            else if((mDWindow.units()).equals("Milliseconds")){
              stringWidth = fmFont.stringWidth(UtilFncs.getTestString((maxExclusiveValue/1000), defaultNumberPrecision));
              barXCoord = barXCoord + stringWidth;
            }
            else{
              stringWidth = fmFont.stringWidth(UtilFncs.getTestString(maxExclusiveValue, defaultNumberPrecision));
              barXCoord = barXCoord + stringWidth;
            }
          }
        }
      
      //******************************
      //Do the mean bar.
      //******************************
      
      //Build the node,context,thread string.
      String s1 = "mean";
      
      if((yCoord >= yBeg) && (yCoord <= yEnd)){
      
      //Test for the different menu options for this window.
      if((mDWindow.getMetric()).equals("Inclusive"))
      { 
        if(mDWindow.isPercent())
        {   
          yCoord = yCoord + (barSpacing);
          
          tmpDataValue = tmpGME.getMeanInclusivePercentValue(trial.getCurValLoc());
          
          int xLength;
          double tmpDouble;
          tmpDouble = (tmpDataValue / maxInclusivePercentValue);
          xLength = (int) (tmpDouble * defaultBarLength);
          if(xLength == 0)
            xLength = 1;
          
          //Now set the color values for drawing!
          //Get the appropriate color.
          tmpColor = tmpGME.getMappingColor();
          g.setColor(tmpColor);
          
          if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
          {
            g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
            
            if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
            {
              g.setColor(trial.getColorChooser().getHighlightColor());
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
              g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            }
            else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
            {
              g.setColor(trial.getColorChooser().getGroupHighlightColor());
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
              g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            }
            else
            {
              g.setColor(Color.black);
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            }
          }
          else
          {
            if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
              g.setColor(trial.getColorChooser().getHighlightColor());
            else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
              g.setColor(trial.getColorChooser().getGroupHighlightColor());
            else
            {
              tmpColor = tmpGME.getMappingColor();
              g.setColor(tmpColor);
            }
            
            g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
          }
          
          //Now print the percentage to the left of the bar.
          g.setColor(Color.black);
          //Need to figure out how long the percentage string will be.
          tmpString = (UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)) + "%";
          stringWidth = fmFont.stringWidth(tmpString);
          //Now draw the percent value to the left of the bar.
          stringStart = barXCoord - xLength - stringWidth - 5;
          g.drawString(tmpString, stringStart, yCoord);
          
          //Now print the node,context,thread to the right of the bar.
          tmpString = s1;
          g.drawString(tmpString, (barXCoord + 5), yCoord);
          
          //Figure out how wide that string was for x coord reasons.
          stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
          if(tmpXWidthCalc < stringWidth)
          {
            tmpXWidthCalc = stringWidth + 15;
          }
        }
        else
        {// End - mDWindow.isPercent()
        
          //For consistancy in drawing, the y coord is updated at the beggining of the loop.
          yCoord = yCoord + (barSpacing);
          
          //Set tmpDataValue to the correct value.
          tmpDataValue = tmpGME.getMeanInclusiveValue(trial.getCurValLoc());
          
          //Figure out how long the bar should be.
          int xLength;
          double tmpDouble;
          tmpDouble = (tmpDataValue / maxInclusiveValue);
          xLength = (int) (tmpDouble * defaultBarLength);
          if(xLength == 0)
            xLength = 1;
          
          //Now set the color values for drawing!
          //Get the appropriate color.
          tmpColor = tmpGME.getMappingColor();
          g.setColor(tmpColor);
          
          if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
          {
            g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
            
            if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
            {
              g.setColor(trial.getColorChooser().getHighlightColor());
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
              g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            }
            else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
            {
              g.setColor(trial.getColorChooser().getGroupHighlightColor());
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
              g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            }
            else
            {
              g.setColor(Color.black);
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            }
          }
          else
          {
            if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
              g.setColor(trial.getColorChooser().getHighlightColor());
            else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
              g.setColor(trial.getColorChooser().getGroupHighlightColor());
            else
            {
              tmpColor = tmpGME.getMappingColor();
              g.setColor(tmpColor);
            }
            
            g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
          }
          
          //Now print the percentage to the left of the bar.
          g.setColor(Color.black);
          
          //Check to see what the units are.
          if((mDWindow.units()).equals("Seconds"))
          {
            tmpString = new String(Double.toString(
                  UtilFncs.adjustDoublePresision((tmpDataValue / 1000000), defaultNumberPrecision)));
            stringWidth = fmFont.stringWidth(tmpString);
            stringStart = barXCoord - xLength - stringWidth - 5;
            g.drawString(tmpString, stringStart, yCoord);
          }
          else if((mDWindow.units()).equals("Milliseconds"))
          {
            tmpString = new String(Double.toString(
                  UtilFncs.adjustDoublePresision((tmpDataValue / 1000), defaultNumberPrecision)));
            stringWidth = fmFont.stringWidth(tmpString);
            stringStart = barXCoord - xLength - stringWidth - 5;
            g.drawString(tmpString, stringStart, yCoord);
          }
          else
          {
            tmpString = new String(Double.toString(
                  UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)));
            stringWidth = fmFont.stringWidth(tmpString);
            stringStart = barXCoord - xLength - stringWidth - 5;
            g.drawString(tmpString, stringStart, yCoord);
          }       
          
          //Now print the node,context,thread to the right of the bar.
          tmpString = s1;
          g.drawString(tmpString, (barXCoord + 5), yCoord);
          
          //Figure out how wide that string was for x coord reasons.
          stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
          if(tmpXWidthCalc < stringWidth)
          {
            tmpXWidthCalc = stringWidth + 15;
          }
        }
      }
      else if((mDWindow.getMetric()).equals("Exclusive"))
      { 
        if(mDWindow.isPercent())
        {
          //For consistancy in drawing, the y coord is updated at the beggining of the loop.
          yCoord = yCoord + (barSpacing);
          
          tmpDataValue = tmpGME.getMeanExclusivePercentValue(trial.getCurValLoc());
          
          int xLength;
          double tmpDouble;
          tmpDouble = (tmpDataValue / maxExclusivePercentValue);
          xLength = (int) (tmpDouble * defaultBarLength);
          if(xLength == 0)
            xLength = 1;
          
          //Now set the color values for drawing!
          //Get the appropriate color.
          tmpColor = tmpGME.getMappingColor();
          g.setColor(tmpColor);
          
          if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
          {
            g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
            
            if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
            {
              g.setColor(trial.getColorChooser().getHighlightColor());
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
              g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            }
            else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
            {
              g.setColor(trial.getColorChooser().getGroupHighlightColor());
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
              g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            }
            else
            {
              g.setColor(Color.black);
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            }
          }
          else
          {
            if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
              g.setColor(trial.getColorChooser().getHighlightColor());
            else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
              g.setColor(trial.getColorChooser().getGroupHighlightColor());
            else
            {
              tmpColor = tmpGME.getMappingColor();
              g.setColor(tmpColor);
            }
            
            g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
          }
          
          //Now print the percentage to the left of the bar.
          g.setColor(Color.black);
          //Need to figure out how long the percentage string will be.
          tmpString = (UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)) + "%";
          stringWidth = fmFont.stringWidth(tmpString);
          stringStart = barXCoord - xLength - stringWidth - 5;
          //Now draw the percent value to the left of the bar.
          g.drawString(tmpString, stringStart, yCoord);
          
          //Now print the node,context,thread to the right of the bar.
          tmpString = s1;
          g.drawString(tmpString, (barXCoord + 5), yCoord);
          
          //Figure out how wide that string was for x coord reasons.
          stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
          if(tmpXWidthCalc < stringWidth)
          {
            tmpXWidthCalc = stringWidth + 15;
          }
          
        }
        else
        {
          yCoord = yCoord + (barSpacing);
          
          //Set tmpDataValue to the correct value.
          tmpDataValue = tmpGME.getMeanExclusiveValue(trial.getCurValLoc());
          
          //Figure out how long the bar should be.
          int xLength;
          double tmpDouble;
          tmpDouble = (tmpDataValue / maxExclusiveValue);
          xLength = (int) (tmpDouble * defaultBarLength);
          if(xLength == 0)
            xLength = 1;
          
          //Now set the color values for drawing!
          //Get the appropriate color.
          tmpColor = tmpGME.getMappingColor();
          g.setColor(tmpColor);
          
          if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
          {
            g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
            
            if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
            {
              g.setColor(trial.getColorChooser().getHighlightColor());
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
              g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            }
            else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
            {
              g.setColor(trial.getColorChooser().getGroupHighlightColor());
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
              g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            }
            else
            {
              g.setColor(Color.black);
              g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            }
          }
          else
          {
            if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
              g.setColor(trial.getColorChooser().getHighlightColor());
            else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
              g.setColor(trial.getColorChooser().getGroupHighlightColor());
            else
            {
              tmpColor = tmpGME.getMappingColor();
              g.setColor(tmpColor);
            }
            
            g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
          }
          
          //Now print the percentage to the left of the bar.
          g.setColor(Color.black);
          
          //Check to see what the units are.
          if((mDWindow.units()).equals("Seconds")){
            tmpString = new String(Double.toString(
                  UtilFncs.adjustDoublePresision((tmpDataValue / 1000000), defaultNumberPrecision)));
            stringWidth = fmFont.stringWidth(tmpString);
            stringStart = barXCoord - xLength - stringWidth - 5;
            g.drawString(tmpString, stringStart, yCoord);
          }
          else if((mDWindow.units()).equals("Milliseconds")){
            tmpString = new String(Double.toString(
                  UtilFncs.adjustDoublePresision((tmpDataValue / 1000), defaultNumberPrecision)));
            stringWidth = fmFont.stringWidth(tmpString);
            stringStart = barXCoord - xLength - stringWidth - 5;
            g.drawString(tmpString, stringStart, yCoord);
          }
          else{
            tmpString = new String(Double.toString(
                  UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)));
            stringWidth = fmFont.stringWidth(tmpString);
            stringStart = barXCoord - xLength - stringWidth - 5;
            g.drawString(tmpString, stringStart, yCoord);
          }       
          
          //Now print the node,context,thread to the right of the bar.
          tmpString = s1;
          g.drawString(tmpString, (barXCoord + 5), yCoord);
          
          //Figure out how wide that string was for x coord reasons.
          stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
          if(tmpXWidthCalc < stringWidth)
          {
            tmpXWidthCalc = stringWidth + 15;
          }
        }
      }
      else if((mDWindow.getMetric()).equals("Number of Calls"))
      { 
        
        yCoord = yCoord + (barSpacing);
        
        //Set tmpDataValue to the correct value.
        tmpDataValue = tmpGME.getMeanNumberOfCalls();
        
        //Figure out how long the bar should be.
        int xLength;
        double tmpDouble;
        tmpDouble = (tmpDataValue / maxNumberOfCalls);
        xLength = (int) (tmpDouble * defaultBarLength);
        if(xLength == 0)
          xLength = 1;
        
        //Now set the color values for drawing!
        //Get the appropriate color.
        tmpColor = tmpGME.getMappingColor();
        g.setColor(tmpColor);
        
        if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
        {
          g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
          
          if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
          {
            g.setColor(trial.getColorChooser().getHighlightColor());
            g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
          }
          else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
          {
            g.setColor(trial.getColorChooser().getGroupHighlightColor());
            g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
          }
          else
          {
            g.setColor(Color.black);
            g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
          }
        }
        else
        {
          if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
            g.setColor(trial.getColorChooser().getHighlightColor());
          else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
            g.setColor(trial.getColorChooser().getGroupHighlightColor());
          else
          {
            tmpColor = tmpGME.getMappingColor();
            g.setColor(tmpColor);
          }
          
          g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
        }
        
        //Now print the percentage to the left of the bar.
        g.setColor(Color.black);
        
        tmpString = new String(Double.toString(tmpDataValue));
        stringWidth = fmFont.stringWidth(tmpString);
        stringStart = barXCoord - xLength - stringWidth - 5;
        g.drawString((Double.toString(tmpDataValue)), stringStart, yCoord);
        
        //Now print the node,context,thread to the right of the bar.
        tmpString = s1;
        g.drawString(tmpString, (barXCoord + 5), yCoord);
        
        //Figure out how wide that string was for x coord reasons.
        stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
        if(tmpXWidthCalc < stringWidth)
        {
          tmpXWidthCalc = stringWidth + 15;
        }
      }
      else if((mDWindow.getMetric()).equals("Number of Subroutines"))
      { 
        
        yCoord = yCoord + (barSpacing);
        
        //Set tmpDataValue to the correct value.
        tmpDataValue = tmpGME.getMeanNumberOfSubRoutines();
        
        //Figure out how long the bar should be.
        int xLength;
        double tmpDouble;
        tmpDouble = (tmpDataValue / maxNumberOfSubroutines);
        xLength = (int) (tmpDouble * defaultBarLength);
        if(xLength == 0)
          xLength = 1;
        
        //Now set the color values for drawing!
        //Get the appropriate color.
        tmpColor = tmpGME.getMappingColor();
        g.setColor(tmpColor);
        
        if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
        {
          g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
          
          if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
          {
            g.setColor(trial.getColorChooser().getHighlightColor());
            g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
          }
          else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
          {
            g.setColor(trial.getColorChooser().getGroupHighlightColor());
            g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
          }
          else
          {
            g.setColor(Color.black);
            g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
          }
        }
        else
        {
          if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
            g.setColor(trial.getColorChooser().getHighlightColor());
          else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
            g.setColor(trial.getColorChooser().getGroupHighlightColor());
          else
          {
            tmpColor = tmpGME.getMappingColor();
            g.setColor(tmpColor);
          }
          
          g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
        }
        
        //Now print the percentage to the left of the bar.
        g.setColor(Color.black);
        
        tmpString = new String(Double.toString(tmpDataValue));
        stringWidth = fmFont.stringWidth(tmpString);
        stringStart = barXCoord - xLength - stringWidth - 5;
        g.drawString((Double.toString(tmpDataValue)), stringStart, yCoord);
        
        //Now print the node,context,thread to the right of the bar.
        tmpString = s1;
        g.drawString(tmpString, (barXCoord + 5), yCoord);
        
        //Figure out how wide that string was for x coord reasons.
        stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
        if(tmpXWidthCalc < stringWidth)
        {
          tmpXWidthCalc = stringWidth + 15;
        }
      }
      else if((mDWindow.getMetric()).equals("Per Call Value"))
      { 
        
        yCoord = yCoord + (barSpacing);
        
        //Set tmpDataValue to the correct value.
        tmpDataValue = tmpGME.getMeanUserSecPerCall(trial.getCurValLoc());
        
        //Figure out how long the bar should be.
        int xLength;
        double tmpDouble;
        tmpDouble = (tmpDataValue / maxUserSecPerCall);
        xLength = (int) (tmpDouble * defaultBarLength);
        if(xLength == 0)
          xLength = 1;
        
        //Now set the color values for drawing!
        //Get the appropriate color.
        tmpColor = tmpGME.getMappingColor();
        g.setColor(tmpColor);
        
        if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
        {
          g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
          
          if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
          {
            g.setColor(trial.getColorChooser().getHighlightColor());
            g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
          }
          else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
          {
            g.setColor(trial.getColorChooser().getGroupHighlightColor());
            g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
          }
          else
          {
            g.setColor(Color.black);
            g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
          }
        }
        else
        {
          if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
            g.setColor(trial.getColorChooser().getHighlightColor());
          else if((tmpGME.isGroupMember(trial.getColorChooser().getGHCMID())))
            g.setColor(trial.getColorChooser().getGroupHighlightColor());
          else
          {
            tmpColor = tmpGME.getMappingColor();
            g.setColor(tmpColor);
          }
          
          g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
        }
        
        //Now print the percentage to the left of the bar.
        g.setColor(Color.black);
        
        tmpString = new String(Double.toString(
                  UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)));
        stringWidth = fmFont.stringWidth(tmpString);
        stringStart = barXCoord - xLength - stringWidth - 5;
        g.drawString(tmpString, stringStart, yCoord);
        
        //Now print the node,context,thread to the right of the bar.
        tmpString = s1;
        g.drawString(tmpString, (barXCoord + 5), yCoord);
        
        //Figure out how wide that string was for x coord reasons.
        stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
        if(tmpXWidthCalc < stringWidth)
        {
          tmpXWidthCalc = stringWidth + 15;
        }
      }
      
      //******************************
      //End - Do the mean bar.
      //******************************
      }
      
      //******************************
      //Now the rest.
      //******************************
      
      serverNumber = 0;
        
      for(Enumeration e1 = (mDWindow.getStaticMainWindowSystemData()).elements(); e1.hasMoreElements() ;)
      {
        //Get the name of the server.
        tmpSMWServer = (SMWServer) e1.nextElement();
        
        //Setting the context counter to zero ... this is really required.
        contextNumber = 0;
        tmpContextList = tmpSMWServer.getContextList();
        for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
        {
          //Get the next context.
          tmpSMWContext = (SMWContext) e2.nextElement();
          
          //Now draw the thread stuff for this context.
          tmpThreadList = tmpSMWContext.getThreadList();
          
          //Setting the context counter to zero ... this is really required as well. :-)
          threadNumber = 0;
          for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
          {
              //Build the node,context,thread string.
            s1 = "n,c,t   " + serverNumber + "," + contextNumber + "," + threadNumber;
            
            tmpSMWThread = (SMWThread) e3.nextElement();
            tmpThreadDataElementList = tmpSMWThread.getThreadDataList();
            
            //Test for the different menu options for this window.
            if((mDWindow.getMetric()).equals("Inclusive")){
              if(mDWindow.isPercent()){
                for(Enumeration e4 = tmpThreadDataElementList.elements(); e4.hasMoreElements() ;){
                  tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
                  
                  if((tmpSMWThreadDataElement.getMappingID()) == mappingID){
                    //For consistancy in drawing, the y coord is updated at the beggining of the loop.
                    yCoord = yCoord + (barSpacing);
                    
                    //Now select whether to draw this thread based on clip rectangle.
                    if((yCoord >= yBeg) && (yCoord <= yEnd)){
                      tmpDataValue = tmpSMWThreadDataElement.getInclusivePercentValue();
                      int xLength;
                      double tmpDouble;
                      tmpDouble = (tmpDataValue / maxInclusivePercentValue);
                      xLength = (int) (tmpDouble * defaultBarLength);
                      if(xLength == 0)
                        xLength = 1;
                      
                      //Now set the color values for drawing!
                      //Get the appropriate color.
                      tmpColor = tmpSMWThreadDataElement.getMappingColor();
                      g.setColor(tmpColor);
                      
                      if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
                      {
                        g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
                        
                        if((tmpSMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
                        {
                          g.setColor(trial.getColorChooser().getHighlightColor());
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                          g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        }
                        else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                        {
                          g.setColor(trial.getColorChooser().getGroupHighlightColor());
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                          g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        }
                        else
                        {
                          g.setColor(Color.black);
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        }
                      }
                      else
                      {
                        if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
                          g.setColor(trial.getColorChooser().getHighlightColor());
                        else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                          g.setColor(trial.getColorChooser().getGroupHighlightColor());
                        else
                        {
                          tmpColor = tmpSMWThreadDataElement.getMappingColor();
                          g.setColor(tmpColor);
                        }
                        
                        g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
                      }
                      
                      //Now print the percentage to the left of the bar.
                      g.setColor(Color.black);
                      //Need to figure out how long the percentage string will be.
                      tmpString = (UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)) + "%";
                      stringWidth = fmFont.stringWidth(tmpString);
                      //Now draw the percent value to the left of the bar.
                      stringStart = barXCoord - xLength - stringWidth - 5;
                      g.drawString(tmpString, stringStart, yCoord);
                      
                      //Now print the node,context,thread to the right of the bar.
                      tmpString = s1;
                      g.drawString(tmpString, (barXCoord + 5), yCoord);
                      
                      //Figure out how wide that string was for x coord reasons.
                      stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
                      if(tmpXWidthCalc < stringWidth)
                      {
                        tmpXWidthCalc = stringWidth + 15;
                      }
                      
                      //Update the drawing coordinates.
                      tmpSMWThreadDataElement.setMDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
                    }
                  }
                }
              }
              else
              {//@@@
                for(Enumeration e5 = tmpThreadDataElementList.elements(); e5.hasMoreElements() ;)
                {
                  tmpSMWThreadDataElement = (SMWThreadDataElement) e5.nextElement();
                  
                  if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
                  {
                    //For consistancy in drawing, the y coord is updated at the beggining of the loop.
                    yCoord = yCoord + (barSpacing);
                    //Now select whether to draw this thread based on clip rectangle.
                    if((yCoord >= yBeg) && (yCoord <= yEnd)){
                      //Set tmpDataValue to the correct value.
                      tmpDataValue = tmpSMWThreadDataElement.getInclusiveValue();
                      
                      //Figure out how long the bar should be.
                      int xLength;
                      double tmpDouble;
                      tmpDouble = (tmpDataValue / maxInclusiveValue);
                      xLength = (int) (tmpDouble * defaultBarLength);
                      if(xLength == 0)
                        xLength = 1;
                      
                      //Now set the color values for drawing!
                      //Get the appropriate color.
                      tmpColor = tmpSMWThreadDataElement.getMappingColor();
                      g.setColor(tmpColor);
                      
                      if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
                      {
                        g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
                        
                        if((tmpSMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
                        {
                          g.setColor(trial.getColorChooser().getHighlightColor());
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                          g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        }
                        else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                        {
                          g.setColor(trial.getColorChooser().getGroupHighlightColor());
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                          g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        }
                        else
                        {
                          g.setColor(Color.black);
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        }
                      }
                      else
                      {
                        if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
                          g.setColor(trial.getColorChooser().getHighlightColor());
                        else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                          g.setColor(trial.getColorChooser().getGroupHighlightColor());
                        else
                        {
                          tmpColor = tmpSMWThreadDataElement.getMappingColor();
                          g.setColor(tmpColor);
                        }
                        
                        g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
                      }
                      
                      //Now print the percentage to the left of the bar.
                      g.setColor(Color.black);
                      
                      //Check to see what the units are.
                      if((mDWindow.units()).equals("Seconds"))
                      {
                        tmpString = new String(Double.toString(
                          UtilFncs.adjustDoublePresision((tmpDataValue / 1000000), defaultNumberPrecision)));
                        stringWidth = fmFont.stringWidth(tmpString);
                        stringStart = barXCoord - xLength - stringWidth - 5;
                        g.drawString(tmpString, stringStart, yCoord);
                      }
                      else if((mDWindow.units()).equals("Milliseconds"))
                      {
                        tmpString = new String(Double.toString(
                          UtilFncs.adjustDoublePresision((tmpDataValue / 1000), defaultNumberPrecision)));
                        stringWidth = fmFont.stringWidth(tmpString);
                        stringStart = barXCoord - xLength - stringWidth - 5;
                        g.drawString(tmpString, stringStart, yCoord);
                      }
                      else
                      {
                        tmpString = new String(Double.toString(
                          UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)));
                        stringWidth = fmFont.stringWidth(tmpString);
                        stringStart = barXCoord - xLength - stringWidth - 5;
                        g.drawString(tmpString, stringStart, yCoord);
                      }       
                      
                      //Now print the node,context,thread to the right of the bar.
                      tmpString = s1;
                      g.drawString(tmpString, (barXCoord + 5), yCoord);
                      
                      //Figure out how wide that string was for x coord reasons.
                      stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
                      if(tmpXWidthCalc < stringWidth)
                      {
                        tmpXWidthCalc = stringWidth + 15;
                      }
                      
                      //Update the drawing coordinates.
                      tmpSMWThreadDataElement.setMDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
                    }
                  }
                }
              }
            }
            else if((mDWindow.getMetric()).equals("Exclusive"))
            {
              if(mDWindow.isPercent())
              {
                for(Enumeration e6 = tmpThreadDataElementList.elements(); e6.hasMoreElements() ;)
                { 
                  tmpSMWThreadDataElement = (SMWThreadDataElement) e6.nextElement();
                  
                  if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
                  {
                    //For consistancy in drawing, the y coord is updated at the beggining of the loop.
                    yCoord = yCoord + (barSpacing);
                    //Now select whether to draw this thread based on clip rectangle.
                    if((yCoord >= yBeg) && (yCoord <= yEnd)){
                      tmpDataValue = tmpSMWThreadDataElement.getExclusivePercentValue();
                      
                      int xLength;
                      double tmpDouble;
                      tmpDouble = (tmpDataValue / maxExclusivePercentValue);
                      xLength = (int) (tmpDouble * defaultBarLength);
                      if(xLength == 0)
                        xLength = 1;
                      
                      //Now set the color values for drawing!
                      //Get the appropriate color.
                      tmpColor = tmpSMWThreadDataElement.getMappingColor();
                      g.setColor(tmpColor);
                      
                      if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
                      {
                        g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
                        
                        if((tmpSMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
                        {
                          g.setColor(trial.getColorChooser().getHighlightColor());
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                          g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        }
                        else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                        {
                          g.setColor(trial.getColorChooser().getGroupHighlightColor());
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                          g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        }
                        else
                        {
                          g.setColor(Color.black);
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        }
                      }
                      else
                      {
                        if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
                          g.setColor(trial.getColorChooser().getHighlightColor());
                        else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                          g.setColor(trial.getColorChooser().getGroupHighlightColor());
                        else
                        {
                          tmpColor = tmpSMWThreadDataElement.getMappingColor();
                          g.setColor(tmpColor);
                        }
                        
                        g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
                      }
                      
                      //Now print the percentage to the left of the bar.
                      g.setColor(Color.black);
                      //Need to figure out how long the percentage string will be.
                      tmpString = (UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)) + "%";
                      stringWidth = fmFont.stringWidth(tmpString);
                      stringStart = barXCoord - xLength - stringWidth - 5;
                      //Now draw the percent value to the left of the bar.
                      g.drawString(tmpString, stringStart, yCoord);
                      
                      //Now print the node,context,thread to the right of the bar.
                      tmpString = s1;
                      g.drawString(tmpString, (barXCoord + 5), yCoord);
                      
                      //Figure out how wide that string was for x coord reasons.
                      stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
                      if(tmpXWidthCalc < stringWidth)
                      {
                        tmpXWidthCalc = stringWidth + 15;
                      }
                      
                      //Update the drawing coordinates.
                      tmpSMWThreadDataElement.setMDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
                    }
                  }
                }
              }
              else
              {
                for(Enumeration e7 = tmpThreadDataElementList.elements(); e7.hasMoreElements() ;)
                {
                  tmpSMWThreadDataElement = (SMWThreadDataElement) e7.nextElement();
                  
                  if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
                  {
                    //For consistancy in drawing, the y coord is updated at the beggining of the loop.
                    yCoord = yCoord + (barSpacing);
                    
                    //Now select whether to draw this thread based on clip rectangle.
                    if((yCoord >= yBeg) && (yCoord <= yEnd)){
                      //Set tmpDataValue to the correct value.
                      tmpDataValue = tmpSMWThreadDataElement.getExclusiveValue();
                      
                      //Figure out how long the bar should be.
                      int xLength;
                      double tmpDouble;
                      tmpDouble = (tmpDataValue / maxExclusiveValue);
                      xLength = (int) (tmpDouble * defaultBarLength);
                      if(xLength == 0)
                        xLength = 1;
                      
                      //Now set the color values for drawing!
                      //Get the appropriate color.
                      tmpColor = tmpSMWThreadDataElement.getMappingColor();
                      g.setColor(tmpColor);
                      
                      if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
                      {
                        g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
                        
                        if((tmpSMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
                        {
                          g.setColor(trial.getColorChooser().getHighlightColor());
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                          g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        }
                        else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                        {
                          g.setColor(trial.getColorChooser().getGroupHighlightColor());
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                          g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        }
                        else
                        {
                          g.setColor(Color.black);
                          g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        }
                      }
                      else
                      {
                        if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
                          g.setColor(trial.getColorChooser().getHighlightColor());
                        else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                          g.setColor(trial.getColorChooser().getGroupHighlightColor());
                        else
                        {
                          tmpColor = tmpSMWThreadDataElement.getMappingColor();
                          g.setColor(tmpColor);
                        }
                        
                        g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
                      }
                      
                      //Now print the percentage to the left of the bar.
                      g.setColor(Color.black);
                      
                      //Check to see what the units are.
                      if((mDWindow.units()).equals("Seconds"))
                      {
                        tmpString = new String(Double.toString(
                          UtilFncs.adjustDoublePresision((tmpDataValue / 1000000), defaultNumberPrecision)));
                        stringWidth = fmFont.stringWidth(tmpString);
                        stringStart = barXCoord - xLength - stringWidth - 5;
                        g.drawString(tmpString, stringStart, yCoord);
                      }
                      else if((mDWindow.units()).equals("Milliseconds"))
                      {
                        tmpString = new String(Double.toString(
                          UtilFncs.adjustDoublePresision((tmpDataValue / 1000), defaultNumberPrecision)));
                        stringWidth = fmFont.stringWidth(tmpString);
                        stringStart = barXCoord - xLength - stringWidth - 5;
                        g.drawString(tmpString, stringStart, yCoord);
                      }
                      else
                      {
                        tmpString = new String(Double.toString(
                          UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)));
                        stringWidth = fmFont.stringWidth(tmpString);
                        stringStart = barXCoord - xLength - stringWidth - 5;
                        g.drawString(tmpString, stringStart, yCoord);
                      }       
                      
                      //Now print the node,context,thread to the right of the bar.
                      tmpString = s1;
                      g.drawString(tmpString, (barXCoord + 5), yCoord);
                      
                      //Figure out how wide that string was for x coord reasons.
                      stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
                      if(tmpXWidthCalc < stringWidth)
                      {
                        tmpXWidthCalc = stringWidth + 15;
                      }
                      
                      //Update the drawing coordinates.
                      tmpSMWThreadDataElement.setMDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
                    }
                  }
                }
              }
            }
            else if((mDWindow.getMetric()).equals("Number of Calls"))
            {
              
              for(Enumeration e7 = tmpThreadDataElementList.elements(); e7.hasMoreElements() ;)
              {
                tmpSMWThreadDataElement = (SMWThreadDataElement) e7.nextElement();
                
                if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
                {
                  //For consistancy in drawing, the y coord is updated at the beggining of the loop.
                  yCoord = yCoord + (barSpacing);
                  //Now select whether to draw this thread based on clip rectangle.
                  if((yCoord >= yBeg) && (yCoord <= yEnd)){
                    //Set tmpDataValue to the correct value.
                    tmpDataValue = tmpSMWThreadDataElement.getNumberOfCalls();
                    
                    //Figure out how long the bar should be.
                    int xLength;
                    double tmpDouble;
                    tmpDouble = (tmpDataValue / maxNumberOfCalls);
                    xLength = (int) (tmpDouble * defaultBarLength);
                    if(xLength == 0)
                      xLength = 1;
                    
                    //Now set the color values for drawing!
                    //Get the appropriate color.
                    tmpColor = tmpSMWThreadDataElement.getMappingColor();
                    g.setColor(tmpColor);
                    
                    if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
                    {
                      g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
                      
                      if((tmpSMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
                      {
                        g.setColor(trial.getColorChooser().getHighlightColor());
                        g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                      }
                      else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                      {
                        g.setColor(trial.getColorChooser().getGroupHighlightColor());
                        g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                      }
                      else
                      {
                        g.setColor(Color.black);
                        g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                      }
                    }
                    else
                    {
                      if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
                        g.setColor(trial.getColorChooser().getHighlightColor());
                      else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                        g.setColor(trial.getColorChooser().getGroupHighlightColor());
                      else
                      {
                        tmpColor = tmpSMWThreadDataElement.getMappingColor();
                        g.setColor(tmpColor);
                      }
                      
                      g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
                    }
                    
                    //Now print the percentage to the left of the bar.
                    g.setColor(Color.black);
                    
                    tmpString = new String(Double.toString(tmpDataValue));
                    stringWidth = fmFont.stringWidth(tmpString);
                    stringStart = barXCoord - xLength - stringWidth - 5;
                    g.drawString((Double.toString(tmpDataValue)), stringStart, yCoord);
                    
                    //Now print the node,context,thread to the right of the bar.
                    tmpString = s1;
                    g.drawString(tmpString, (barXCoord + 5), yCoord);
                    
                    //Figure out how wide that string was for x coord reasons.
                    stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
                    if(tmpXWidthCalc < stringWidth)
                    {
                      tmpXWidthCalc = stringWidth + 15;
                    }
                    
                    //Update the drawing coordinates.
                    tmpSMWThreadDataElement.setMDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
                  }
                }
              }
            }
            else if((mDWindow.getMetric()).equals("Number of Subroutines"))
            {
              
              for(Enumeration e7 = tmpThreadDataElementList.elements(); e7.hasMoreElements() ;)
              {
                tmpSMWThreadDataElement = (SMWThreadDataElement) e7.nextElement();
                
                if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
                {
                  //For consistancy in drawing, the y coord is updated at the beggining of the loop.
                  yCoord = yCoord + (barSpacing);
                  //Now select whether to draw this thread based on clip rectangle.
                  if((yCoord >= yBeg) && (yCoord <= yEnd)){
                    //Set tmpDataValue to the correct value.
                    tmpDataValue = tmpSMWThreadDataElement.getNumberOfSubRoutines();
                    
                    //Figure out how long the bar should be.
                    int xLength;
                    double tmpDouble;
                    tmpDouble = (tmpDataValue / maxNumberOfSubroutines);
                    xLength = (int) (tmpDouble * defaultBarLength);
                    if(xLength == 0)
                      xLength = 1;
                    
                    //Now set the color values for drawing!
                    //Get the appropriate color.
                    tmpColor = tmpSMWThreadDataElement.getMappingColor();
                    g.setColor(tmpColor);
                    
                    if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
                    {
                      g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
                      
                      if((tmpSMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
                      {
                        g.setColor(trial.getColorChooser().getHighlightColor());
                        g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                      }
                      else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                      {
                        g.setColor(trial.getColorChooser().getGroupHighlightColor());
                        g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                      }
                      else
                      {
                        g.setColor(Color.black);
                        g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                      }
                    }
                    else
                    {
                      if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
                        g.setColor(trial.getColorChooser().getHighlightColor());
                      else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                        g.setColor(trial.getColorChooser().getGroupHighlightColor());
                      else
                      {
                        tmpColor = tmpSMWThreadDataElement.getMappingColor();
                        g.setColor(tmpColor);
                      }
                      
                      g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
                    }
                    
                    //Now print the percentage to the left of the bar.
                    g.setColor(Color.black);
                    
                    tmpString = new String(Double.toString(tmpDataValue));
                    stringWidth = fmFont.stringWidth(tmpString);
                    stringStart = barXCoord - xLength - stringWidth - 5;
                    g.drawString((Double.toString(tmpDataValue)), stringStart, yCoord);
                    
                    //Now print the node,context,thread to the right of the bar.
                    tmpString = s1;
                    g.drawString(tmpString, (barXCoord + 5), yCoord);
                    
                    //Figure out how wide that string was for x coord reasons.
                    stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
                    if(tmpXWidthCalc < stringWidth)
                    {
                      tmpXWidthCalc = stringWidth + 15;
                    }
                    
                    //Update the drawing coordinates.
                    tmpSMWThreadDataElement.setMDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
                  }
                }
              }
            }
            else if((mDWindow.getMetric()).equals("Per Call Value"))
            {
              
              for(Enumeration e7 = tmpThreadDataElementList.elements(); e7.hasMoreElements() ;)
              {
                tmpSMWThreadDataElement = (SMWThreadDataElement) e7.nextElement();
                
                if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
                {
                  //For consistancy in drawing, the y coord is updated at the beggining of the loop.
                  yCoord = yCoord + (barSpacing);
                  //Now select whether to draw this thread based on clip rectangle.
                  if((yCoord >= yBeg) && (yCoord <= yEnd)){
                    //Set tmpDataValue to the correct value.
                    tmpDataValue = tmpSMWThreadDataElement.getUserSecPerCall();
                    
                    //Figure out how long the bar should be.
                    int xLength;
                    double tmpDouble;
                    tmpDouble = (tmpDataValue / maxUserSecPerCall);
                    xLength = (int) (tmpDouble * defaultBarLength);
                    if(xLength == 0)
                      xLength = 1;
                    
                    //Now set the color values for drawing!
                    //Get the appropriate color.
                    tmpColor = tmpSMWThreadDataElement.getMappingColor();
                    g.setColor(tmpColor);
                    
                    if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
                    {
                      g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
                      
                      if((tmpSMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
                      {
                        g.setColor(trial.getColorChooser().getHighlightColor());
                        g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                      }
                      else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                      {
                        g.setColor(trial.getColorChooser().getGroupHighlightColor());
                        g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                        g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                      }
                      else
                      {
                        g.setColor(Color.black);
                        g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                      }
                    }
                    else
                    {
                      if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
                        g.setColor(trial.getColorChooser().getHighlightColor());
                      else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
                        g.setColor(trial.getColorChooser().getGroupHighlightColor());
                      else
                      {
                        tmpColor = tmpSMWThreadDataElement.getMappingColor();
                        g.setColor(tmpColor);
                      }
                      
                      g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
                    }
                    
                    //Now print the percentage to the left of the bar.
                    g.setColor(Color.black);
                    
                    tmpString = new String(Double.toString(
                      UtilFncs.adjustDoublePresision(tmpDataValue, defaultNumberPrecision)));
                    stringWidth = fmFont.stringWidth(tmpString);
                    stringStart = barXCoord - xLength - stringWidth - 5;
                    g.drawString(tmpString, stringStart, yCoord);
                    
                    //Now print the node,context,thread to the right of the bar.
                    tmpString = s1;
                    g.drawString(tmpString, (barXCoord + 5), yCoord);
                    
                    //Figure out how wide that string was for x coord reasons.
                    stringWidth =  (barXCoord + fmFont.stringWidth(s1) + 5); 
                    if(tmpXWidthCalc < stringWidth)
                    {
                      tmpXWidthCalc = stringWidth + 15;
                    }
                    
                    //Update the drawing coordinates.
                    tmpSMWThreadDataElement.setMDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
                  }
                }
              }
            }
            
          threadNumber++;
            
          }
          
        contextNumber++;
        
        }
      
      serverNumber++;
      
      }
      g.setColor(Color.white);
      g.fillRect(0, yBeg, tmpXWidthCalc, 2*nameMetricHeight+12);
      g.setColor(Color.black);
      
      g.setFont(nameMetricFont);
      //**********
      //Draw the counter name if required.
      counterName = trial.getCounterName();
      if(counterName != null){
        g.drawString("COUNTER NAME: " + counterName, 5, yBeg + nameMetricHeight + 3);
      }
      //End - Draw the counter name if required.
      //**********
      
      
      //**********
      //Draw the mapping name.
      g.drawString("FUNCTION NAME: " + mappingName, 5, yBeg+2*nameMetricHeight+6);
      //Calculate its width.
      tmpXWidthCalc = fmFont.stringWidth(mappingName);
      //End - Draw the mapping name.
      //**********
      
      boolean sizeChange = false;   
      //Resize the panel if needed.
      if(tmpXWidthCalc > 550){
        xPanelSize = tmpXWidthCalc + 1;
        sizeChange = true;
      }
      
      if(yCoord > 550){
        yPanelSize = yCoord + 1;
        sizeChange = true;
      }
      
      if(sizeChange)
        revalidate();
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MDWP03");
    }
    
    
  }
  
  //******************************
  //Event listener code!!
  //******************************
  
  
  //ActionListener code.
  public void actionPerformed(ActionEvent evt)
  {
    try
    {
      Object EventSrc = evt.getSource();
      
      if(EventSrc instanceof JMenuItem)
      {
        String arg = evt.getActionCommand();
        if(arg.equals("Change Function Color"))
        { 
          GlobalMapping globalMappingReference = trial.getGlobalMapping();
          GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
          
          Color tmpCol = tmpGME.getMappingColor();
          
          JColorChooser tmpJColorChooser = new JColorChooser();
          tmpCol = tmpJColorChooser.showDialog(this, "Please select a new color", tmpCol);
          if(tmpCol != null)
          {
            tmpGME.setSpecificColor(tmpCol);
            tmpGME.setColorFlag(true);
            
            trial.getSystemEvents().updateRegisteredObjects("colorEvent");
          }
        }
        
        else if(arg.equals("Reset to Generic Color"))
        { 
          GlobalMapping globalMappingReference = trial.getGlobalMapping();
          GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
          
          tmpGME.setColorFlag(false);
          trial.getSystemEvents().updateRegisteredObjects("colorEvent");
        }
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MDWP04");
    }
  }
  
  //Ok, now the mouse listeners for this panel.
  public void mouseClicked(MouseEvent evt)
  {
    try
    {
      //For the moment, I am just showing the popup menu anywhere.
      //For a future release, there will be more here.
      if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0)
      {
        popup.show(this, evt.getX(), evt.getY());
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MDWP05");
    }
  }
  
  public void mousePressed(MouseEvent evt) {}
  public void mouseReleased(MouseEvent evt) {}
  public void mouseEntered(MouseEvent evt) {}
  public void mouseExited(MouseEvent evt) {}
  
  public void changeInMultiples()
  {
    computeDefaultBarLength();
    this.repaint();
  }
  
  public Dimension getPreferredSize()
  {
    return new Dimension(xPanelSize, (yPanelSize + 10));
  }
  
  public void computeDefaultBarLength()
  {
    try
    {
      double sliderValue = (double) mDWindow.getSliderValue();
      double sliderMultiple = mDWindow.getSliderMultiple();
      double result = 250*sliderValue*sliderMultiple;
      
      defaultBarLength = (int) result;
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MDWP06");
    }
  }
  //******************************
  //Instance data.
  //******************************
  private Vector staticNodeList;
  
  private int newXPanelSize = 0;
  private int newYPanelSize = 0;
  
  private String counterName = null;
  
  private int mappingID = -1;
  private String mappingName;
  
  
  private int barHeight = -1;
  private int barSpacing = -1;
  private int defaultBarLength = 250;
  private int maxXLength = 0;
  
  private double maxInclusiveValue = 0;
  private double maxExclusiveValue = 0;
  private double maxInclusivePercentValue = 0;
  private double maxExclusivePercentValue = 0;
  private double maxNumberOfCalls = 0;
  private double maxNumberOfSubroutines = 0;
  private double maxUserSecPerCall = 0;
  
  private int serverNumber = -1;
  private int contextNumber = -1;
  private int threadNumber = -1;
  
  private Trial trial = null;
  private MappingDataWindow mDWindow = null;
  
  private StaticMainWindowData sMWData = null;
  
  private SMWServer tmpSMWServer = null;
  private SMWContext tmpSMWContext = null;
  private SMWThread tmpSMWThread = null;
  private SMWThreadDataElement tmpSMWThreadDataElement = null;
  private Vector tmpContextList = null;
  private Vector tmpThreadList = null;
  private Vector tmpThreadDataElementList = null;
  
  //**********
  //Popup menu definitions.
  private JPopupMenu popup = new JPopupMenu();
  //**********
  
  //******************************
  //End - Instance data.
  //******************************
}

