
//MappingLedgerWindowPanel.

/* 
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


public class MappingLedgerWindowPanel extends JPanel implements ActionListener, MouseListener
{
  int xPanelSize = 300;
  int yPanelSize = 400;
  
  int barHeight;
  int barSpacing;
  
  
  public MappingLedgerWindowPanel()
  {
    try{
      setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
      
      //Schedule a repaint of this panel.
      this.repaint();
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MLWP01");
    }
  
  }
  
  public MappingLedgerWindowPanel(Trial inTrial, Vector inNameIDMapping,  int inMappingSelection)
  {
    try{
      setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
      setBackground(Color.white);
      
      trial = inTrial;
      NameIDMapping = inNameIDMapping;
      mappingSelection = inMappingSelection;
      
      //Add this object as a mouse listener.
      addMouseListener(this);
      
      if(mappingSelection == 0)
      {
        //Add items to the popu menu.
        JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
        mappingDetailsItem.addActionListener(this);
        popup.add(mappingDetailsItem);
        
        JMenuItem changeColorItem = new JMenuItem("Change Function Color");
        changeColorItem.addActionListener(this);
        popup.add(changeColorItem);
      }
      
      if(mappingSelection == 2)
      {
        //Add items to the popu menu.
        JMenuItem mappingDetailsItem = new JMenuItem("Show User Event Details");
        mappingDetailsItem.addActionListener(this);
        popup.add(mappingDetailsItem);
        
        JMenuItem changeColorItem = new JMenuItem("Change User Event Color");
        changeColorItem.addActionListener(this);
        popup.add(changeColorItem);
      }
      
      JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
      maskMappingItem.addActionListener(this);
      popup.add(maskMappingItem);
      
      if(mappingSelection == 1)
      {
        //Add items to the popu menu.
        JMenuItem showThisMappingOnlyItem = new JMenuItem("Show This Group Only");
        showThisMappingOnlyItem.addActionListener(this);
        popup.add(showThisMappingOnlyItem);
        
        //Add items to the popu menu.
        JMenuItem showAllButMappingsItem = new JMenuItem("Show All Groups Except This One");
        showAllButMappingsItem.addActionListener(this);
        popup.add(showAllButMappingsItem);
        
        //Add items to the popu menu.
        JMenuItem showAllMappingsItem = new JMenuItem("Show All Groups");
        showAllMappingsItem.addActionListener(this);
        popup.add(showAllMappingsItem);
      }
      
      //JMenuItem toGenericColorItem = new JMenuItem("Mask Mapping");
      //toGenericColorItem.addActionListener(this);
      //popup.add(toGenericColorItem);
      
      
      //Schedule a repaint of this panel.
      this.repaint();
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MLWP02");
    }
    
    
  
  }
  

  public void paintComponent(Graphics g)
  {
    try{
      super.paintComponent(g);
      
      //Set the numberOfColors variable.
      numberOfColors = trial.getColorChooser().getNumberOfColors();
      
      //Cycle through the id mapping list.
      GlobalMappingElement globalMappingElement = null;
      String tmpString;
      Color tmpColor;
    
      //Note that trying to set the coords of the top left corner of this componant
      //caused problems with redrawing.  Specifically, with the scroll pane, negative
      //values were sometimes returned.  So,  I am just starting at 0,0.  It would
      //be nice to know why this was occuring!
      int xCoord = 0;
      int yCoord = 0;
      
      int nWidth = 0;
      int cWidth = 0;
      int tWidth = 0;
      
      //An XCoord used in drawing the bar graphs.
      int barXCoord = 0;
      
      int tmpXWidthCalc = 0;
    
      xCoord = 5;
      
      //Do the standard font and spacing stuff.
      if(!(trial.getPreferences().areBarDetailsSet()))
      {
        
        //Create font.
        Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(), 12);
        g.setFont(font);
        FontMetrics fmFont = g.getFontMetrics(font);
        
        //Set up the bar details.
        
        //Compute the font metrics.
        int maxFontAscent = fmFont.getAscent();
        
        trial.getPreferences().setBarDetails(maxFontAscent, maxFontAscent);
        
        trial.getPreferences().setSliders(maxFontAscent, maxFontAscent);
      }
      
      //Set local spacing and bar heights.
      barSpacing = trial.getPreferences().getBarSpacing();
      barHeight = trial.getPreferences().getBarHeight();
      
      //Create font.
      Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(), barHeight);
      g.setFont(font);
      FontMetrics fmFont = g.getFontMetrics(font);
    
      yCoord = yCoord + (barSpacing);
      
      //**********
      //Draw the counter name if required.
      counterName = trial.getCounterName();
      if(counterName != null){
        g.drawString("COUNTER NAME: " + counterName, 5, yCoord);
        yCoord = yCoord + (barSpacing);
      }
      //End - Draw the counter name if required.
      //**********
    
      int tmpCounter = 0;
      
      for(Enumeration e1 = NameIDMapping.elements(); e1.hasMoreElements() ;)
      {
        globalMappingElement = (GlobalMappingElement) e1.nextElement();
        if((globalMappingElement.getMappingName()) != null)
        {
        
            //For consistancy in drawing, the y coord is updated at the beginning of the loop.
            yCoord = yCoord + (barSpacing);
            
            //First draw the mapping color box.
            tmpColor = globalMappingElement.getMappingColor();
            g.setColor(tmpColor);
            
            g.fillRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
            
            
            if(mappingSelection == 2)
            {
              if((globalMappingElement.getGlobalID()) == (trial.getColorChooser().getUEHCMappingID()))
              {
                g.setColor(trial.getColorChooser().getUEHC());
                g.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
                g.drawRect(xCoord + 1, (yCoord - barHeight) + 1, barHeight - 2, barHeight - 2);
              }
              else
              {
                g.setColor(Color.black);
                g.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
              }
            }
            else if(mappingSelection == 1)
            {
              if((globalMappingElement.getGlobalID()) == (trial.getColorChooser().getGHCMID()))
              {
                g.setColor(trial.getColorChooser().getGroupHighlightColor());
                g.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
                g.drawRect(xCoord + 1, (yCoord - barHeight) + 1, barHeight - 2, barHeight - 2);
              }
              else
              {
                g.setColor(Color.black);
                g.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
              }
            }
            else
            {
              if((globalMappingElement.getGlobalID()) == (trial.getColorChooser().getHighlightColorMappingID()))
              {
                g.setColor(trial.getColorChooser().getHighlightColor());
                g.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
                g.drawRect(xCoord + 1, (yCoord - barHeight) + 1, barHeight - 2, barHeight - 2);
              }
              else
              {
                g.setColor(Color.black);
                g.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
              }
            }
            
            //Update the xCoord to draw the mapping name.
            xCoord = xCoord + (barHeight + 10);
            //Reset the drawing color to the text color ... in this case, black.
            g.setColor(Color.black);
            
            //Draw the mapping name.
            tmpString = globalMappingElement.getMappingName();
            
            g.drawString(tmpString, xCoord, yCoord);
            
            //Figure out how wide that string was for x coord reasons.
            int tmpWidth = 5 + barHeight + (fmFont.stringWidth(tmpString));
            
            //Figure out how wide that string was for x coord reasons.
            if(tmpXWidthCalc < tmpWidth)
            {
              tmpXWidthCalc = (tmpWidth + 11);
            }
            
            globalMappingElement.setDrawCoords(0, tmpWidth, (yCoord - barHeight), yCoord);
            
            //Reset the xCoord.
            xCoord = xCoord - (barHeight + 10);
        }
            
                        
        tmpCounter++;
                                    

      }
      
      //Resize the panel if needed.
      if((yCoord >= yPanelSize) || (tmpXWidthCalc >= xPanelSize))
      {
        yPanelSize = yCoord + 1;
        xPanelSize = tmpXWidthCalc + 1;
        
        revalidate();
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MLWP03");
    }
    
  }
  
  //ActionListener code.
  public void actionPerformed(ActionEvent evt)
  {
    try{
      Object EventSrc = evt.getSource();
      
      GlobalMappingElement tmpGlobalMappingElement = null;
      
      if(EventSrc instanceof JMenuItem)
      {
        String arg = evt.getActionCommand();
        if(arg.equals("Show Function Details"))
        {
          
          if(clickedOnObject instanceof GlobalMappingElement)
          {
            tmpGlobalMappingElement = (GlobalMappingElement) clickedOnObject;
            //Bring up an expanded data window for this mapping, and set this mapping as highlighted.
            trial.getColorChooser().setHighlightColorMappingID(tmpGlobalMappingElement.getGlobalID());
            MappingDataWindow tmpRef = new MappingDataWindow(trial, tmpGlobalMappingElement.getGlobalID(), trial.getStaticMainWindow().getSMWData());
            trial.getSystemEvents().addObserver(tmpRef);
            tmpRef.show();
          }
        }
        else if(arg.equals("Show User Event Details"))
        {
          
          if(clickedOnObject instanceof GlobalMappingElement)
          {
            tmpGlobalMappingElement = (GlobalMappingElement) clickedOnObject;
            //Bring up an expanded data window for this mapping, and set this mapping as highlighted.
            trial.getColorChooser().setUEHCMappingID(tmpGlobalMappingElement.getGlobalID());
            UserEventWindow tmpRef = new UserEventWindow(trial, tmpGlobalMappingElement.getGlobalID(), trial.getStaticMainWindow().getSMWData());
            trial.getSystemEvents().addObserver(tmpRef);
            tmpRef.show();
          }
        }
        else if((arg.equals("Change Function Color")) || (arg.equals("Change User Event Color")))
        { 
          if(clickedOnObject instanceof GlobalMappingElement)
            tmpGlobalMappingElement = (GlobalMappingElement) clickedOnObject;
          
          Color tmpCol = tmpGlobalMappingElement.getMappingColor();
          
          JColorChooser tmpJColorChooser = new JColorChooser();
          tmpCol = tmpJColorChooser.showDialog(this, "Please select a new color", tmpCol);
          if(tmpCol != null)
          {
            tmpGlobalMappingElement.setSpecificColor(tmpCol);
            tmpGlobalMappingElement.setColorFlag(true);
            
            trial.getSystemEvents().updateRegisteredObjects("colorEvent");
          }
        }
        else if(arg.equals("Reset to Generic Color"))
        { 
          
          if(clickedOnObject instanceof GlobalMappingElement)
            tmpGlobalMappingElement = (GlobalMappingElement) clickedOnObject;
          
          tmpGlobalMappingElement.setColorFlag(false);
          
          trial.getSystemEvents().updateRegisteredObjects("colorEvent");
        }
        else if(arg.equals("Show This Group Only"))
        { 
          
          if(clickedOnObject instanceof GlobalMappingElement)
            tmpGlobalMappingElement = (GlobalMappingElement) clickedOnObject;
          
          GlobalMapping tmpGM = trial.getGlobalMapping();
          tmpGM.setIsSelectedGroupOn(true);
          tmpGM.setIsAllExceptGroupOn(false);
          tmpGM.setSelectedGroupID(tmpGlobalMappingElement.getGlobalID());
          
          trial.getSystemEvents().updateRegisteredObjects("dataEvent");
        }
        else if(arg.equals("Show All Groups Except This One"))
        { 
          
          if(clickedOnObject instanceof GlobalMappingElement)
            tmpGlobalMappingElement = (GlobalMappingElement) clickedOnObject;
          
          GlobalMapping tmpGM = trial.getGlobalMapping();
          tmpGM.setIsSelectedGroupOn(true);
          tmpGM.setIsAllExceptGroupOn(true);
          tmpGM.setSelectedGroupID(tmpGlobalMappingElement.getGlobalID());
          
          trial.getSystemEvents().updateRegisteredObjects("dataEvent");
        }
        else if(arg.equals("Show All Groups"))
        { 
          
          if(clickedOnObject instanceof GlobalMappingElement)
            tmpGlobalMappingElement = (GlobalMappingElement) clickedOnObject;
          
          GlobalMapping tmpGM = trial.getGlobalMapping();
          tmpGM.setIsSelectedGroupOn(false);
          tmpGM.setIsAllExceptGroupOn(false);
          tmpGM.setSelectedGroupID(-1);
          
          trial.getSystemEvents().updateRegisteredObjects("dataEvent");
        }
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MLWP04");
    }
  }
  
  //Ok, now the mouse listeners for this panel.
  public void mouseClicked(MouseEvent evt)
  {
    try{
      //Get the location of the mouse.
      int xCoord = evt.getX();
      int yCoord = evt.getY();
      
      //Get the number of times clicked.
      int clickCount = evt.getClickCount();
      
      //Cycle through the id mapping list.
      GlobalMappingElement globalMappingElement = null;
      
      for(Enumeration e1 = NameIDMapping.elements(); e1.hasMoreElements() ;)
      {
        globalMappingElement = (GlobalMappingElement) e1.nextElement();
                  
        if(yCoord <= (globalMappingElement.getYEnd()))
        {
          if((yCoord >= (globalMappingElement.getYBeg())) && (xCoord >= (globalMappingElement.getXBeg()))
                                      && (xCoord <= (globalMappingElement.getXEnd())))
          {
            if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0)
            {
              //Set the clickedSMWMeanDataElement.
              clickedOnObject = globalMappingElement;
              popup.show(this, evt.getX(), evt.getY());
              
              //Return from this mapping.
              return;
            }
            else
            {
              if(mappingSelection == 2)
              {
                
                //Want to set the clicked on mapping to the current highlight color or, if the one
                //clicked on is already the current highlighted one, set it back to normal.
                if((trial.getColorChooser().getUEHCMappingID()) == -1)
                {
                  trial.getColorChooser().setUEHCMappingID(globalMappingElement.getGlobalID());
                }
                else
                {
                  if(!((trial.getColorChooser().getUEHCMappingID()) == (globalMappingElement.getGlobalID())))
                    trial.getColorChooser().setUEHCMappingID(globalMappingElement.getGlobalID());
                  else
                    trial.getColorChooser().setUEHCMappingID(-1);
                }
              }
              else if(mappingSelection == 1)
              {
                
                //Want to set the clicked on mapping to the current highlight color or, if the one
                //clicked on is already the current highlighted one, set it back to normal.
                if((trial.getColorChooser().getGHCMID()) == -1)
                {
                  trial.getColorChooser().setGroupHighlightColorMappingID(globalMappingElement.getGlobalID());
                }
                else
                {
                  if(!((trial.getColorChooser().getGHCMID()) == (globalMappingElement.getGlobalID())))
                    trial.getColorChooser().setGroupHighlightColorMappingID(globalMappingElement.getGlobalID());
                  else
                    trial.getColorChooser().setGroupHighlightColorMappingID(-1);
                }
              }
              else
              {
                //Want to set the clicked on mapping to the current highlight color or, if the one
                //clicked on is already the current highlighted one, set it back to normal.
                if((trial.getColorChooser().getHighlightColorMappingID()) == -1)
                {
                  trial.getColorChooser().setHighlightColorMappingID(globalMappingElement.getGlobalID());
                }
                else
                {
                  if(!((trial.getColorChooser().getHighlightColorMappingID()) == (globalMappingElement.getGlobalID())))
                    trial.getColorChooser().setHighlightColorMappingID(globalMappingElement.getGlobalID());
                  else
                    trial.getColorChooser().setHighlightColorMappingID(-1);
                }
              }
            }
            //Nothing more to do ... return.
            return;
          }
          else
          {
            //If we get here, it means that we are outside the mapping draw area.  That is, we
            //are either to the left or right of the draw area, or just above it.
            //It is better to return here as we do not want the system to cycle through the
            //rest of the objects, which would be pointless as we know that it will not be
            //one of the others.  Significantly improves performance.
            return;
          }
        }
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "MLWP05");
    }
  }
  
  public void mousePressed(MouseEvent evt) {}
  public void mouseReleased(MouseEvent evt) {}
  public void mouseEntered(MouseEvent evt) {}
  public void mouseExited(MouseEvent evt) {}
  
  
  public Dimension getPreferredSize()
  {
    return new Dimension((xPanelSize + 10), (yPanelSize + 10));
  }
  
  
  
  //******************************
  //Instance data.
  //******************************
  Trial trial = null;
  Vector NameIDMapping;
  int mappingSelection = -1;
  int numberOfColors = 0;
  
  //**********
  //Popup menu definitions.
  private JPopupMenu popup = new JPopupMenu();
  //**********
  
  //**********
  //Other useful variables.
  String counterName = null;
  Object clickedOnObject = null;
  //End - Other useful variables.
  //**********
    
  
  //******************************
  //End - Instance data.
  //******************************
}
