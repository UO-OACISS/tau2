/* 
  
  BinWindowPanel.java
  
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
import java.awt.print.*;
import java.awt.geom.*;
//import javax.print.*;


public class BinWindowPanel extends JPanel implements ActionListener, MouseListener, PopupMenuListener

{
  //******************************
  //Instance data.
  //******************************
  private Trial trial = null;
  BinWindow binWindow = null;
  boolean normalBin = true;
  int mappingID = -1;
  int xPanelSize = 600;
  int yPanelSize = 400;
  
  //**********
  //Popup menu definitions.
  private JPopupMenu popup = new JPopupMenu();
  private JPopupMenu popup2 = new JPopupMenu();
  
  JMenuItem tUESWItem = new JMenuItem("Show Total User Event Statistics Windows");
  
  //**********
  
  //**********
  //Some place holder definitions - used for cycling through data lists.
  Vector contextList = null;
  Vector threadList = null;
  Vector threadDataList = null;
  SMWServer sMWServer = null;
  SMWContext sMWContext = null;
  SMWThread sMWThread = null;
  SMWThreadDataElement sMWThreadDataElement = null;
  SMWMeanDataElement sMWMeanDataElement = null;
  //End - Place holder definitions.
  //**********
  
  //**********
  //Other useful variables for getToolTipText, mouseEvents, and paintComponent.
  int xCoord = -1;
  int yCoord = -1;
  Object clickedOnObject = null;
  //End - Other useful variables for getToolTipText, mouseEvents, and paintComponent.
  //**********
  
  //**********
  //Some misc stuff for the paintComponent function.
  String counterName = null;
  private int defaultBarLength = 500;
  String tmpString = null;
  double tmpSum = -1;
  double tmpDataValue = -1;
  Color tmpColor = null;
  boolean highlighted = false;
  int barXStart = -1;
  int numberOfColors = 0;
  //End - Some misc stuff for the paintComponent function.
  //**********
  
  
  //**********
  //The constructors!!
  public BinWindowPanel()
  {
    try{
      //Set the default tool tip for this panel.
      this.setToolTipText("Incorrect Constructor!!!");
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "SMWP01");
    }
  }
  
  public BinWindowPanel(Trial inTrial, BinWindow inBW, boolean normal, int id)
  {
    try{
      //Set the default tool tip for this panel.
      this.setToolTipText("ParaProf bar graph draw window!");
      setBackground(Color.white);
      
      normalBin = normal;
      mappingID = id;
      //Add this object as a mouse listener.
      addMouseListener(this);
      
      //Set instance variables.
      trial = inTrial;
      binWindow = inBW;
      barXStart = 100;
      
      //Add items to the first popup menu.
      JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
      mappingDetailsItem.addActionListener(this);
      popup.add(mappingDetailsItem);
      
      JMenuItem changeColorItem = new JMenuItem("Change Function Color");
      changeColorItem.addActionListener(this);
      popup.add(changeColorItem);
      
      JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
      maskMappingItem.addActionListener(this);
      popup.add(maskMappingItem);
      
      JMenuItem highlightMappingItem = new JMenuItem("Highlight this Function");
      highlightMappingItem.addActionListener(this);
      popup.add(highlightMappingItem);
      
      JMenuItem unHighlightMappingItem = new JMenuItem("Un-Highlight this Function");
      unHighlightMappingItem.addActionListener(this);
      popup.add(unHighlightMappingItem);
      
      //Add items to the second popup menu.
      popup2.addPopupMenuListener(this);
      
      JMenuItem tSWItem = new JMenuItem("Show Total Statistics Windows");
      tSWItem.addActionListener(this);
      popup2.add(tSWItem);
      
      tUESWItem = new JMenuItem("Show Total User Event Statistics Windows");
      tUESWItem.addActionListener(this);
      popup2.add(tUESWItem);
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "SMWP02");
    }
  }
  //End - The constructors!!
  //**********
  
  public String getToolTipText(MouseEvent evt){
    return null;  
  }
  
  //******************************
  //Event listener code!!
  //******************************
  
  
  //ActionListener code.
  public void actionPerformed(ActionEvent evt){
  }
  
  //**********
  //Mouse listeners for this panel.
  public void mouseClicked(MouseEvent evt){}
  public void mousePressed(MouseEvent evt){}
  public void mouseReleased(MouseEvent evt){}
  public void mouseEntered(MouseEvent evt){}
  public void mouseExited(MouseEvent evt){}
  
  //End - Mouse listeners for this panel.
  //**********
  
  
  
  
  public void paintComponent(Graphics g)
  {
    try
    {
      super.paintComponent(g);
      
      Graphics2D g2 = (Graphics2D) g;
      
      //Set the numberOfColors variable.
      numberOfColors = trial.getColorChooser().getNumberOfColors();
      
      //Check to see if selected groups only are being displayed.
      GlobalMapping tmpGM = trial.getGlobalMapping();
      
      boolean isSelectedGroupOn = false;
      int selectedGroupID = 0;
      
      if(tmpGM.getIsSelectedGroupOn()){
        isSelectedGroupOn = true;
        selectedGroupID = tmpGM.getSelectedGroupID();
      } 
      
      //**********
      //Other initializations.
      highlighted = false;
      xCoord = yCoord = 0;
      //End - Other initializations.
      //**********
      
      //**********
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
      //End - Do the standard font and spacing stuff.
      //**********
      
      //Set local spacing and bar heights.
      int barSpacing = trial.getPreferences().getBarSpacing();
      int barHeight = trial.getPreferences().getBarHeight();
      
      //Create font.
      Font font = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), barHeight);
      g.setFont(font);
      FontMetrics fmFont = g.getFontMetrics(font);
      
      
      //**********
      //Calculating the starting positions of drawing.
      String tmpString2 = new String("n,c,t 99,99,99");
      int stringWidth = fmFont.stringWidth(tmpString2);
      barXStart = stringWidth + 15;
      int tmpXWidthCalc = barXStart + defaultBarLength;
      int barXCoord = barXStart;
      yCoord = yCoord + (barSpacing);
      //End - Calculating the starting positions of drawing.
      //**********
      
      
      Vector tmpVector = trial.getGlobalMapping().getMapping(0);
      
      //**********
      //Draw the counter name if required.
      counterName = trial.getCounterName();
      if(counterName != null){
        g.drawString("COUNTER NAME: " + counterName, 5, yCoord);
        
        GlobalMappingElement tmpGME1 = (GlobalMappingElement) tmpVector.elementAt(mappingID);
        String name = tmpGME1.getMappingName();
        g.drawString(name, 360, yCoord);
        
        yCoord = yCoord + (barSpacing);
      }
      //End - Draw the counter name if required.
      //**********
      
      
      Rectangle clipRect = g.getClipBounds();
    
      int yBeg = (int) clipRect.getY();
      int yEnd = (int) (yBeg + clipRect.getHeight());
      //Because tooltip redraw can louse things up.  Add an extra one to draw.
      yEnd = yEnd + barSpacing;
      
      yCoord = yCoord + (barSpacing);
        
      
      //Set the drawing color to the text color ... in this case, black.
      g.setColor(Color.black);
      
      g.drawLine(35, 430, 35, 30);
      g.drawLine(35,430,585, 430);
      
      
      double maxValue = 0;
      double minValue = 0;
      boolean start = true;
      if(normalBin){
        for(int i=0;i<tmpVector.size();i++){
          GlobalMappingElement tmpGME = (GlobalMappingElement) tmpVector.elementAt(i);
          double tmpDouble = tmpGME.getTotalExclusiveValue(trial.getCurValLoc());
          if(tmpDouble > maxValue)
            maxValue = tmpDouble;
        }
      }
      else{
        
        SMWServer tmpSMWServer = null;
        SMWContext tmpSMWContext = null;
        SMWThread tmpSMWThread = null;
        SMWThreadDataElement tmpSMWThreadDataElement = null;
        Vector tmpContextList = null;
        Vector tmpThreadList = null;
        Vector tmpThreadDataElementList = null;
        for(Enumeration e1 = (binWindow.getStaticMainWindowSystemData()).elements(); e1.hasMoreElements() ;)
        {
          //Get the name of the server.
          tmpSMWServer = (SMWServer) e1.nextElement();
          
          tmpContextList = tmpSMWServer.getContextList();
          for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
          {
            //Get the next context.
            tmpSMWContext = (SMWContext) e2.nextElement();
            
            //Now draw the thread stuff for this context.
            tmpThreadList = tmpSMWContext.getThreadList();
            
            for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
            {
              tmpSMWThread = (SMWThread) e3.nextElement();
              tmpThreadDataElementList = tmpSMWThread.getThreadDataList();
              
              for(Enumeration e4 = tmpThreadDataElementList.elements(); e4.hasMoreElements() ;){
                tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
                if((tmpSMWThreadDataElement.getMappingID()) == mappingID){
                  double tmpDataValue = tmpSMWThreadDataElement.getExclusiveValue();
                  if(start){
                    minValue = tmpDataValue;
                    start = false;
                  }
                  System.out.println("value is: " + tmpDataValue);
                  if(tmpDataValue > maxValue)
                    maxValue = tmpDataValue;
                  if(tmpDataValue < minValue)
                    minValue = tmpDataValue;
                }
              }
            }
          }
        }
        
        
        //This max is not working.
        //System.out.println("Mapping ID is: " + mappingID);
        //GlobalMappingElement tmpGME = (GlobalMappingElement) tmpVector.elementAt(0);
        //maxValue = tmpGME.getMaxExclusiveValue(trial.getCurValLoc());
      }
      
      
      //maxValue = maxValue/1000;
      
      System.out.println("maxValue: " + maxValue);
      
      double increment = maxValue / 10;
      
      
      for(int i=0; i<10; i++){
        g.drawLine(30,30+i*40,35,30+i*40);
        g.drawString(""+(10*(10-i)), 5, 33+i*40);
      }
      
      for(int i=1; i<11; i++){
        g.drawLine(35+i*55,430,35+i*55,435);
      }
      
      AffineTransform currentTransform = g2.getTransform();


      AffineTransform test = new AffineTransform();
      test.translate(30,495);
      test.rotate(Math.toRadians(90));
      g2.setTransform(test);
      
      AffineTransform first = new AffineTransform();
      first.translate(27.5,0);
      test.preConcatenate(first);
      g2.setTransform(test);
      
      AffineTransform concat = new AffineTransform();
      concat.translate(55,0);
      
      
      /*for(int i=1; i<10; i++){
        double rightBound = i*increment;
        g.drawString(i + "/10th MV", 0, 0);
        test.preConcatenate(concat);
        g2.setTransform(test);
      }*/
      
      g2.setTransform(currentTransform);
      g.drawString("Min Value  = " + minValue, 35, 450);              
      g.drawString("Max Value = " + maxValue, 552, 450);
      
      int[] intArray = new int[10];
      
      for(int i=0;i<10;i++){
        intArray[i]=0;
      }
      
      if(!normalBin){
      
        SMWServer tmpSMWServer = null;
        SMWContext tmpSMWContext = null;
        SMWThread tmpSMWThread = null;
        SMWThreadDataElement tmpSMWThreadDataElement = null;
        Vector tmpContextList = null;
        Vector tmpThreadList = null;
        Vector tmpThreadDataElementList = null;
        for(Enumeration e1 = (binWindow.getStaticMainWindowSystemData()).elements(); e1.hasMoreElements() ;)
        {
          //Get the name of the server.
          tmpSMWServer = (SMWServer) e1.nextElement();
          
          tmpContextList = tmpSMWServer.getContextList();
          for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
          {
            //Get the next context.
            tmpSMWContext = (SMWContext) e2.nextElement();
            
            //Now draw the thread stuff for this context.
            tmpThreadList = tmpSMWContext.getThreadList();
            
            for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
            {
              tmpSMWThread = (SMWThread) e3.nextElement();
              tmpThreadDataElementList = tmpSMWThread.getThreadDataList();
              
              for(Enumeration e4 = tmpThreadDataElementList.elements(); e4.hasMoreElements() ;){
                tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
                if((tmpSMWThreadDataElement.getMappingID()) == mappingID){
                  double tmpDataValue = tmpSMWThreadDataElement.getExclusiveValue();
                  for(int j=10;j>0;j--){
                    if(tmpDataValue <= (minValue + ((maxValue-minValue)/j))){
                      intArray[10-j]++;
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }
      else{
        for(int i=0;i<tmpVector.size();i++){    
          GlobalMappingElement tmpGME = (GlobalMappingElement) tmpVector.elementAt(i);
          double tmpDouble = tmpGME.getTotalExclusiveValue(trial.getCurValLoc());
          for(int j=10;j>0;j--){
            if(tmpDouble <= (maxValue/j)){
              intArray[10-j]++;
              break;
            }
          }
        }
      }
      
      for(int i=0;i<10;i++){
        System.out.println("Arr. Loc. " + i + " is: " + intArray[i]);
      }
      
      g.setColor(Color.red);
      
      int num = 512;
      System.out.println("Number of gm is: " + num);
      for(int i=0; i<10; i++){
        if(intArray[i] != 0){
          double tmp1 = intArray[i];
          
          double per = (tmp1/num) * 100;
          int result = (int) per;
          System.out.println("result is: " + result);
          g.fillRect(38+i*55,430-(result*4),49,result*4);}
      }
      
      
      boolean sizeChange = false;   
      //Resize the panel if needed.
      if(tmpXWidthCalc > 600){
        xPanelSize = tmpXWidthCalc + 1;
        sizeChange = true;
      }
      
      if(yCoord > 300){
        yPanelSize = yCoord + 1;
        sizeChange = true;
      }
      
      if(sizeChange)
        revalidate();
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "SMWP06");
    }
  }
  
  //******************************
  //PopupMenuListener code.
  //******************************
  public void popupMenuWillBecomeVisible(PopupMenuEvent evt){
    try
    {
      if(trial.userEventsPresent()){
        tUESWItem.setEnabled(true);
      }
      else{
        tUESWItem.setEnabled(false);
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "SMW03");
    }
  }
  public void popupMenuWillBecomeInvisible(PopupMenuEvent evt){}
  public void popupMenuCanceled(PopupMenuEvent evt){}
  //******************************
  //End - PopupMenuListener code.
  //****************************** 
  
  public Dimension getPreferredSize()
  {
    return new Dimension(xPanelSize + 10, (yPanelSize + 10));
  }
}
