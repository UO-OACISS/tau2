/* 
  
TotalStatWindowPanel.java
  
Title:      ParaProf
Author:     Robert Bell
Description:  
*/

package paraprof;

import java.util.*;
import java.text.*;
import java.awt.font.TextAttribute;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;


public class TotalStatWindowPanel extends JPanel implements ActionListener, MouseListener
{   
    public TotalStatWindowPanel()
    {
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
      
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "TSWP01");
	    }
  
    }
  
  
    public TotalStatWindowPanel(Trial inTrial,
				int inServerNumber,
				int inContextNumber,
				int inThreadNumber,
				TotalStatWindow inTSWindow)
    {
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);

	    serverNumber = inServerNumber;
	    contextNumber = inContextNumber;
	    threadNumber = inThreadNumber;

	    trial = inTrial;
	    tSWindow = inTSWindow;
      
	    //Add this object as a mouse listener.
	    addMouseListener(this);
      
	    //Add items to the popu menu.
	    JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
	    mappingDetailsItem.addActionListener(this);
	    popup.add(mappingDetailsItem);
      
	    JMenuItem changeColorItem = new JMenuItem("Change Function Color");
	    changeColorItem.addActionListener(this);
	    popup.add(changeColorItem);
      
	    JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
	    maskMappingItem.addActionListener(this);
	    popup.add(maskMappingItem);
      
	    this.repaint();
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "TSWP02");
	    }
  
    }
  

    public void paintComponent(Graphics g)
    {
	try{
	    super.paintComponent(g);
      
	    double tmpSum;
	    double tmpDataValue;
	    Color tmpColor;
	    int yCoord = 0;
      
      
	    //In this window, a Monospaced font has to be used.  This will probably not be the same
	    //font as the rest of ParaProf.  As a result, some extra work will have to be done to calculate
	    //spacing.
	    int fontSize = trial.getPreferences().getBarHeight();
	    spacing = trial.getPreferences().getBarSpacing();
      
	    int tmpXWidthCalc = 0;
      
	    String tmpString = null;
	    String dashString = "";
      
	    //Create font.
	    MonoFont = new Font("Monospaced", trial.getPreferences().getFontStyle(), fontSize);
	    //Compute the font metrics.
	    fmMonoFont = g.getFontMetrics(MonoFont);
	    maxFontAscent = fmMonoFont.getMaxAscent();
	    maxFontDescent = fmMonoFont.getMaxDescent();
	    g.setFont(MonoFont);
      
	    if(spacing <= (maxFontAscent + maxFontDescent))
		{
		    spacing = spacing + 1;
		}
      
	    //Grab the appropriate thread.
      
	    tmpThreadDataElementList = tSWindow.getStaticMainWindowSystemData();
      
	    //With group support present, it is possible that the number of mappings in
	    //our data list is zero.  If so, just return.
	    if((tmpThreadDataElementList.size()) == 0)
		return;
      
      
	    Rectangle clipRect = g.getClipBounds();
      
	    int yBeg = (int) clipRect.getY();
	    int yEnd = (int) (yBeg + clipRect.getHeight());
	    int startThreadElement = 0;
	    int endThreadElement = 0;
      
      
	    yCoord = yCoord + (spacing);
      
	    //**********
	    //Draw the counter name if required.
	    String counterName = trial.getCounterName();
	    if(counterName != null){
		g.drawString("COUNTER NAME: " + counterName + UtilFncs.getUnitsString(tSWindow.units()), 5, yCoord);
		yCoord = yCoord + (spacing);
	    }
	    //End - Draw the counter name if required.
	    //**********
      
      
	    //To be on the safe side, have an alternative to the clip rectangle.
	    if ((clipRect != null))
		{
		    //Draw the heading!
		    tmpString = GlobalThreadDataElement.getTStatStringHeading(trial);
		    int tmpInt = tmpString.length();
        
		    for(int i=0; i<tmpInt; i++)
			{
			    dashString = dashString + "-";
			}
        
		    g.setColor(Color.black);
        
		    AttributedString dashStringAS = new AttributedString(dashString);
		    dashStringAS.addAttribute(TextAttribute.FONT, MonoFont);
        
		    //Draw the first dashed string.
		    yCoord = yCoord + spacing;
		    g.drawString(dashStringAS.getIterator(), 20, yCoord);
		    yCoord = yCoord + spacing + 10;
        
		    //Draw the heading.
		    AttributedString headingStringAS = new AttributedString(tmpString);
		    headingStringAS.addAttribute(TextAttribute.FONT, MonoFont);
		    g.drawString(headingStringAS.getIterator(), 20, yCoord);
		    yCoord = yCoord + spacing + 10;
        
		    //Draw the second dashed string.
		    g.drawString(dashStringAS.getIterator(), 20, yCoord);
        
		    startLocation = yCoord;
        
		    //Set up some panel dimensions.
		    newYPanelSize = yCoord + ((tmpThreadDataElementList.size() + 1) * spacing);
          
		    startThreadElement = ((yBeg - yCoord) / spacing) - 1;
		    endThreadElement  = ((yEnd - yCoord) / spacing) + 1;
          
		    if((yCoord > yBeg) || (yCoord > yEnd))
			{
			    if(yCoord > yBeg)
				{
				    startThreadElement = 0;
				}
            
			    if(yCoord > yEnd)
				{
				    endThreadElement = 0;
				}
			}
          
		    if(startThreadElement < 0)
			startThreadElement = 0;
            
		    if(endThreadElement < 0)
			endThreadElement = 0;
          
		    if(startThreadElement > (tmpThreadDataElementList.size() - 1))
			startThreadElement = (tmpThreadDataElementList.size() - 1);
            
		    if(endThreadElement > (tmpThreadDataElementList.size() - 1))
			endThreadElement = (tmpThreadDataElementList.size() - 1);
          
		    yCoord = yCoord + (startThreadElement * spacing);
          
		    for(int i = startThreadElement; i <= endThreadElement; i++)
			{ 
			    tmpSMWThreadDataElement = (SMWThreadDataElement) tmpThreadDataElementList.elementAt(i);
			    tmpString = tmpSMWThreadDataElement.getTStatString(tSWindow.units());
          
			    yCoord = yCoord + spacing;
          
			    g.setColor(Color.black);
            
			    AttributedString as = new AttributedString(tmpString);
			    as.addAttribute(TextAttribute.FONT, MonoFont);
          
			    if((tmpSMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
				as.addAttribute(TextAttribute.FOREGROUND, 
						(trial.getColorChooser().getHighlightColor()),
						GlobalThreadDataElement.getPositionOfName(), tmpString.length());
			    else if((tmpSMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
				as.addAttribute(TextAttribute.FOREGROUND, 
						(trial.getColorChooser().getGroupHighlightColor()),
						GlobalThreadDataElement.getPositionOfName(), tmpString.length());
			    else
				as.addAttribute(TextAttribute.FOREGROUND, 
						(tmpSMWThreadDataElement.getMappingColor()),
						GlobalThreadDataElement.getPositionOfName(), tmpString.length());
          
			    g.drawString(as.getIterator(), 20, yCoord);
          
			    //Figure out how wide that string was for x coord reasons.
			    if(tmpXWidthCalc < 2*fmMonoFont.stringWidth(tmpString))
				{
				    tmpXWidthCalc = (20 + 2*fmMonoFont.stringWidth(tmpString));
				}
			}
            
		    //Resize the panel if needed.
		    if((newYPanelSize >= yPanelSize) || (tmpXWidthCalc  >= xPanelSize)){
			yPanelSize = newYPanelSize + 1;
			xPanelSize = tmpXWidthCalc + 1;
			
			revalidate();
		    } 
          
		}
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "TSWP03");
	    }
    
    }
  
    //******************************
    //Event listener code!!
    //******************************
  
  
    //ActionListener code.
    public void actionPerformed(ActionEvent evt)
    {
	try{
	    Object EventSrc = evt.getSource();
      
	    SMWThreadDataElement tmpSMWThreadDataElement = null;
      
	    if(EventSrc instanceof JMenuItem)
		{
		    String arg = evt.getActionCommand();
		    if(arg.equals("Show Function Details"))
			{
          
			    if(clickedOnObject instanceof SMWThreadDataElement)
				{
				    tmpSMWThreadDataElement = (SMWThreadDataElement) clickedOnObject;
				    //Bring up an expanded data window for this mapping, and set this mapping as highlighted.
				    trial.getColorChooser().setHighlightColorMappingID(tmpSMWThreadDataElement.getMappingID());
				    MappingDataWindow tmpRef = new MappingDataWindow(trial, tmpSMWThreadDataElement.getMappingID(), trial.getStaticMainWindow().getSMWData());
				    trial.getSystemEvents().addObserver(tmpRef);
				    tmpRef.show();
				}
			}
		    else if(arg.equals("Change Function Color"))
			{ 
			    int mappingID = -1;
          
			    //Get the clicked on object.
			    if(clickedOnObject instanceof SMWThreadDataElement)
				mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
          
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
          
			    int mappingID = -1;
          
			    //Get the clicked on object.
			    if(clickedOnObject instanceof SMWThreadDataElement)
				mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
          
			    GlobalMapping globalMappingReference = trial.getGlobalMapping();
			    GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
          
			    tmpGME.setColorFlag(false);
			    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
			}
		}
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "TSWP04");
	    }
    }
  
    //Ok, now the mouse listeners for this panel.
    public void mouseClicked(MouseEvent evt)
    {
    
	try{
	    //Get the location of the mouse.
	    //Get the location of the mouse.
	    int xCoord = evt.getX();
	    int yCoord = evt.getY();
      
	    int fontSize = trial.getPreferences().getBarHeight();
      
	    //Get the number of times clicked.
	    int clickCount = evt.getClickCount();
      
	    int tmpInt1 = yCoord - startLocation;
	    int tmpInt2 = tmpInt1 / spacing;
	    int tmpInt3 = (tmpInt2 + 1) * spacing;
	    int tmpInt4 = tmpInt3 - maxFontAscent;
      
	    if((tmpInt1 >= tmpInt4) && (tmpInt1 <= tmpInt3))
		{
		    if(tmpInt2 < (tmpThreadDataElementList.size())) 
			{
			    tmpSMWThreadDataElement = (SMWThreadDataElement) tmpThreadDataElementList.elementAt(tmpInt2);
          
          
			    if(fmMonoFont != null)
				{
				    String tmpString = tmpSMWThreadDataElement.getTStatString(tSWindow.units());
				    int stringWidth = fmMonoFont.stringWidth(tmpString) + 20;
          
				    if(xCoord <= stringWidth)
					{
            
					    if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0)
						{
						    //Set the clickedSMWMeanDataElement.
						    clickedOnObject = tmpSMWThreadDataElement;
						    popup.show(this, evt.getX(), evt.getY());
						}
					    else
						{
						    //Want to set the clicked on mapping to the current highlight color or, if the one
						    //clicked on is already the current highlighted one, set it back to normal.
						    if((trial.getColorChooser().getHighlightColorMappingID()) == -1)
							{
							    trial.getColorChooser().setHighlightColorMappingID(tmpSMWThreadDataElement.getMappingID());
							}
						    else
							{
							    if(!((trial.getColorChooser().getHighlightColorMappingID()) == (tmpSMWThreadDataElement.getMappingID())))
								trial.getColorChooser().setHighlightColorMappingID(tmpSMWThreadDataElement.getMappingID());
							    else
								trial.getColorChooser().setHighlightColorMappingID(-1);
							}
						}
					}
				}
			}
		}
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "TSWP05");
	    }
    }
  
    public void mousePressed(MouseEvent evt) {}
    public void mouseReleased(MouseEvent evt) {}
    public void mouseEntered(MouseEvent evt) {}
    public void mouseExited(MouseEvent evt) {}
  
    public Dimension getPreferredSize()
    {
	return new Dimension(xPanelSize, (yPanelSize + 10));
    }
  
    //Instance stuff.
  
    int xPanelSize = 800;
    int yPanelSize = 600;
    int newXPanelSize = 0;
    int newYPanelSize = 0;
  
  
    //Some drawing details.
    int startLocation = 0;
    int maxFontAscent = 0;
    int maxFontDescent = 0;
    int spacing = 0;
  
    int serverNumber;
    int contextNumber;
    int threadNumber;
    private Trial trial = null;
    TotalStatWindow tSWindow;
    Vector tmpThreadDataElementList;
    SMWThreadDataElement tmpSMWThreadDataElement;
  
    Font MonoFont = null;
    FontMetrics fmMonoFont = null;
  
    //**********
    //Popup menu definitions.
    private JPopupMenu popup = new JPopupMenu();
    //**********
  
    //**********
    //Other useful variables.
    Object clickedOnObject = null;
    //End - Other useful variables.
    //**********
}
