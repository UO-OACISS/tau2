/* 
	
	TotalStatUEWindowPanel.java
	
	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.text.*;
import java.awt.font.TextAttribute;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;


public class TotalStatUEWindowPanel extends JPanel implements ActionListener, MouseListener
{		
	public TotalStatUEWindowPanel()
	{
		try{
			setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
			
			//Schedule a repaint of this panel.
			this.repaint();
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSUEWP01");
		}
	
	}
	
	
	public TotalStatUEWindowPanel(int inServerNumber,
								int inContextNumber,
								int inThreadNumber,
								TotalStatUEWindow inTSUEWindow)
	{
		try{
			setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
			setBackground(Color.white);

			serverNumber = inServerNumber;
			contextNumber = inContextNumber;
			threadNumber = inThreadNumber;

			tSUEWindow = inTSUEWindow;
			
			//Add this object as a mouse listener.
			addMouseListener(this);
			
			//Add items to the popu menu.
			JMenuItem mappingDetailsItem = new JMenuItem("Show User Event Details");
			mappingDetailsItem.addActionListener(this);
			popup.add(mappingDetailsItem);
			
			JMenuItem changeColorItem = new JMenuItem("Change User Event Color");
			changeColorItem.addActionListener(this);
			popup.add(changeColorItem);
			
			JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
			maskMappingItem.addActionListener(this);
			popup.add(maskMappingItem);
			
			this.repaint();
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSUEWP02");
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
			//font as the rest of jRacy.  As a result, some extra work will have to be done to calculate
			//spacing.
			int fontSize = jRacy.jRacyPreferences.getBarHeight();
			spacing = jRacy.jRacyPreferences.getBarSpacing();
			
			int tmpXWidthCalc = 0;
			
			String tmpString = null;
			String dashString = "";
			
			
			//Create font.
			MonoFont = new Font("Monospaced", jRacy.jRacyPreferences.getFontStyle(), fontSize);
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
			
			tmpThreadDataElementList = tSUEWindow.getStaticMainWindowSystemData();
			
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
			String counterName = jRacy.staticSystemData.getCounterName();
			if(counterName != null){
				g.drawString("COUNTER NAME: " + counterName, 5, yCoord);
				yCoord = yCoord + (spacing);
			}
			//End - Draw the counter name if required.
			//**********
			
			
		    //To be on the safe side, have an alternative to the clip rectangle.
		    if ((clipRect != null))
		    {
		    	//Draw the heading!
				tmpString = jRacy.staticSystemData.getUserEventHeading();
				int tmpInt = tmpString.length();
				
				for(int i=0; i<tmpInt; i++)
				{
					dashString = dashString + "-";
				}
				
				g.setColor(Color.black);
				yCoord = yCoord + spacing;
				g.drawString(dashString, 20, yCoord);
				yCoord = yCoord + spacing + 10;
				g.drawString(tmpString, 20, yCoord);
				yCoord = yCoord + spacing + 10;
				g.drawString(dashString, 20, yCoord);
				
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
					tmpString = tmpSMWThreadDataElement.getUserEventStatString();
					
					yCoord = yCoord + spacing;
					
		    		g.setColor(Color.black);
						
					AttributedString as = new AttributedString(tmpString);
					as.addAttribute(TextAttribute.FONT, MonoFont);
					
					if((tmpSMWThreadDataElement.getUserEventID()) == (jRacy.clrChooser.getUEHCMappingID()))
						as.addAttribute(TextAttribute.FOREGROUND, 
							(jRacy.clrChooser.getUEHC()),
							jRacy.staticSystemData.getPositionOfUserEventName(), tmpString.length());
					else
						as.addAttribute(TextAttribute.FOREGROUND, 
							(tmpSMWThreadDataElement.getUserEventMappingColor()),
							jRacy.staticSystemData.getPositionOfUserEventName(), tmpString.length());
					
					g.drawString(as.getIterator(), 20, yCoord);
					
					//Figure out how wide that string was for x coord reasons.
					if(tmpXWidthCalc < (20 + fmMonoFont.stringWidth(tmpString) + 5))
					{
						tmpXWidthCalc = (20 + fmMonoFont.stringWidth(tmpString) + 15);
					}
				}
		    		
				//Resize the panel if needed.
				if((newYPanelSize >= yPanelSize) || (tmpXWidthCalc  >= xPanelSize))
				{
					yPanelSize = newYPanelSize + 1;
					xPanelSize = tmpXWidthCalc + 1;
					
					revalidate();
				}	
					
			}
			else
			{
				//Draw the heading!
				tmpString = jRacy.staticSystemData.getUserEventHeading();
				int tmpInt = tmpString.length();
				
				for(int i=0; i<tmpInt; i++)
				{
					dashString = dashString + "-";
				}
				
				g.setColor(Color.black);
				yCoord = yCoord + spacing;
				g.drawString(dashString, 20, yCoord);
				yCoord = yCoord + spacing + 10;
				g.drawString(tmpString, 20, yCoord);
				yCoord = yCoord + spacing + 10;
				g.drawString(dashString, 20, yCoord);
				
				startLocation = yCoord;
				
		    	//Set up some panel dimensions.
		    	newYPanelSize = yCoord + ((tmpThreadDataElementList.size() + 1) * spacing);
				
				//Cycle through the elements getting the strings.
				for(Enumeration e1 = tmpThreadDataElementList.elements(); e1.hasMoreElements() ;)
				{	
					tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
					tmpString = tmpSMWThreadDataElement.getUserEventStatString();
					
					if(tmpString != null)
					{
						
						yCoord = yCoord + spacing;
						
						g.setColor(Color.black);
						
						AttributedString as = new AttributedString(tmpString);
						as.addAttribute(TextAttribute.FONT, MonoFont);
						
						if((tmpSMWThreadDataElement.getUserEventID()) == (jRacy.clrChooser.getUEHCMappingID()))
							as.addAttribute(TextAttribute.FOREGROUND, 
								(jRacy.clrChooser.getUEHC()),
								jRacy.staticSystemData.getPositionOfUserEventName(), tmpString.length());
						else
							as.addAttribute(TextAttribute.FOREGROUND, 
								(tmpSMWThreadDataElement.getUserEventMappingColor()),
								jRacy.staticSystemData.getPositionOfUserEventName(), tmpString.length());
						
						g.drawString(as.getIterator(), 20, yCoord);
						
						//Figure out how wide that string was for x coord reasons.
						if(tmpXWidthCalc < (20 + fmMonoFont.stringWidth(tmpString) + 5))
						{
							tmpXWidthCalc = (20 + fmMonoFont.stringWidth(tmpString) + 15);
						}
					}
				}
						
				//Resize the panel if needed.
				if((newYPanelSize >= yPanelSize) || (tmpXWidthCalc  >= xPanelSize))
				{
					yPanelSize = newYPanelSize + 1;
					xPanelSize = tmpXWidthCalc + 1;
					
					revalidate();
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSUEWP03");
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
				if(arg.equals("Show User Event Details"))
				{
					
					if(clickedOnObject instanceof SMWThreadDataElement)
					{
						tmpSMWThreadDataElement = (SMWThreadDataElement) clickedOnObject;
						//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
						jRacy.clrChooser.setUEHCMappingID(tmpSMWThreadDataElement.getMappingID());
						UserEventWindow tmpRef = new UserEventWindow(tmpSMWThreadDataElement.getMappingID(), jRacy.staticMainWindow.getSMWData());
						jRacy.systemEvents.addObserver(tmpRef);
						tmpRef.show();
					}
				}
				else if(arg.equals("Change User Event Color"))
				{	
					int mappingID = -1;
					
					//Get the clicked on object.
					if(clickedOnObject instanceof SMWThreadDataElement)
						mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
					
					GlobalMapping globalMappingReference = jRacy.staticSystemData.getGlobalMapping();
					GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
					
					Color tmpCol = tmpGME.getMappingColor();
					
					JColorChooser tmpJColorChooser = new JColorChooser();
					tmpCol = tmpJColorChooser.showDialog(this, "Please select a new color", tmpCol);
					if(tmpCol != null)
					{
						tmpGME.setSpecificColor(tmpCol);
						tmpGME.setColorFlag(true);
						
						jRacy.systemEvents.updateRegisteredObjects("colorEvent");
					}
				}
				else if(arg.equals("Reset to Generic Color"))
				{	
					
					int mappingID = -1;
					
					//Get the clicked on object.
					if(clickedOnObject instanceof SMWThreadDataElement)
						mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
					
					GlobalMapping globalMappingReference = jRacy.staticSystemData.getGlobalMapping();
					GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
					
					tmpGME.setColorFlag(false);
					jRacy.systemEvents.updateRegisteredObjects("colorEvent");
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSUEWP04");
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
			
			int fontSize = jRacy.jRacyPreferences.getBarHeight();
			
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
						String tmpString = tmpSMWThreadDataElement.getUserEventStatString();
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
								if((jRacy.clrChooser.getUEHCMappingID()) == -1)
								{
									jRacy.clrChooser.setUEHCMappingID((tmpSMWThreadDataElement.getMappingID()));
								}
								else
								{
									if(!((jRacy.clrChooser.getUEHCMappingID()) == (tmpSMWThreadDataElement.getMappingID())))
										jRacy.clrChooser.setUEHCMappingID(tmpSMWThreadDataElement.getMappingID());
									else
										jRacy.clrChooser.setUEHCMappingID(-1);
								}
							}
						}
					}
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TSUEWP05");
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
	int	contextNumber;
	int	threadNumber;
 	TotalStatUEWindow tSUEWindow;
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