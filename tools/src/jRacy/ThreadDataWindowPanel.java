/* 
	
	ThreadDataWindowPanel.java
	
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


public class ThreadDataWindowPanel extends JPanel implements ActionListener, MouseListener
{
	int xPanelSize = 700;
	int yPanelSize = 450;
	
	public ThreadDataWindowPanel()
	{
		
		try
		{
			setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
			
			//Schedule a repaint of this panel.
			this.repaint();
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TDWP01");
		}
	
	}
	
	
	public ThreadDataWindowPanel(int inServerNumber,
								 int inContextNumber,
								 int inThreadNumber,
								 ThreadDataWindow inTDWindow,
								 StaticMainWindowData inSMWData)
	{
		try
		{
		setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
			setBackground(Color.white);
			
			//Add this object as a mouse listener.
			addMouseListener(this);
			
			serverNumber = inServerNumber;
			contextNumber = inContextNumber;
			threadNumber = inThreadNumber;
			tDWindow = inTDWindow;
			sMWData = inSMWData;
			
			
			//Find the correct global thread.
			//This should remain constant throughout the life of this window. Thus, it is
			//safe to grab it here, and not have to grab in every paint component call.
			GlobalServer tmpGS = (GlobalServer) (jRacy.staticSystemData.getStaticServerList()).elementAt(serverNumber);
			Vector tmpGlobalContextList = tmpGS.getContextList();
			GlobalContext tmpGC = (GlobalContext) tmpGlobalContextList.elementAt(contextNumber);
			Vector tmpGlobalThreadList = tmpGC.getThreadList();
			tmpGT = (GlobalThread) tmpGlobalThreadList.elementAt(threadNumber);

			
			//**********
			//Add items to the popu menu.
			JMenuItem mappingDetailsItem = new JMenuItem("Show Mapping Details");
			mappingDetailsItem.addActionListener(this);
			popup.add(mappingDetailsItem);
			
			JMenuItem changeColorItem = new JMenuItem("Change Mapping Color");
			changeColorItem.addActionListener(this);
			popup.add(changeColorItem);
			
			JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
			maskMappingItem.addActionListener(this);
			popup.add(maskMappingItem);
			//End - Add items to the popu menu.
			//**********
			
			//Schedule a repaint of this panel.
			this.repaint();
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TDWP02");
		}
	
	}
	

	public void paintComponent(Graphics g)
	{
		try
		{
			super.paintComponent(g);
			
			//Set the numberOfColors variable.
			numberOfColors = jRacy.clrChooser.getNumberOfColors();
			
			//**********
			//Do the standard font and spacing stuff.
			if(!(jRacy.jRacyPreferences.areBarDetailsSet())){
				Font font = new Font(jRacy.jRacyPreferences.getJRacyFont(), jRacy.jRacyPreferences.getFontStyle(), 12);
				g.setFont(font);
				FontMetrics fmFont = g.getFontMetrics(font);
				
				int maxFontAscent = fmFont.getAscent();
				int maxFontDescent = fmFont.getMaxDescent();
				int tmpInt = maxFontAscent + maxFontDescent;
				jRacy.jRacyPreferences.setBarDetails(maxFontAscent, (tmpInt + 5));
				jRacy.jRacyPreferences.setSliders(maxFontAscent, (tmpInt + 5));}
			
			//Set local spacing and bar heights.
			barSpacing = jRacy.jRacyPreferences.getBarSpacing();
			barHeight = jRacy.jRacyPreferences.getBarHeight();
			
			Font font = new Font(jRacy.jRacyPreferences.getJRacyFont(), jRacy.jRacyPreferences.getFontStyle(), barHeight);
			g.setFont(font);
			FontMetrics fmFont = g.getFontMetrics(font);
			//Do the standard font and spacing stuff.
			//**********

			//**********
			//Some declarations.
			double tmpSum = 0.00;
			double tmpDataValue;
			Color tmpColor;
			String tmpString;
			int stringWidth;
			int stringStart;
			int yCoord = 0;
			int tmpXWidthCalc = 0;
			int barXCoord = 0;
			//End - Some declarations.
			//**********
			
			yCoord = yCoord + (barSpacing);
			
			//**********
			//Draw the counter name if required.
			counterName = jRacy.staticSystemData.getCounterName();
			if(counterName != null){
				g.drawString("COUNTER NAME: " + counterName, 5, yCoord);
				yCoord = yCoord + (barSpacing);}
			//End - Draw the counter name if required.
			//**********
		
			//Grab the appropriate thread.
			tmpThreadDataElementList = tDWindow.getStaticMainWindowSystemData();
			
			//With group support present, it is possible that the number of mappings in
			//our data list is zero.  If so, just return.
			if((tmpThreadDataElementList.size()) == 0)
				return;
			
			Rectangle clipRect = g.getClipBounds();
			
			int yBeg = (int) clipRect.getY();
			int yEnd = (int) (yBeg + clipRect.getHeight());
			int startThreadElement = 0;
			int endThreadElement = 0;
			
		    if ((clipRect != null))
		    {	
		    	//@@@In the clipping section. - This comment aids in matching up if/else statements.@@@
		    	
		    	
		    	//Set up some panel dimensions.
		    	newYPanelSize = yCoord + ((tmpThreadDataElementList.size() + 1) * barSpacing);
		    	
		    	startThreadElement = ((yBeg - yCoord) / barSpacing) - 1;
		    	endThreadElement  = ((yEnd - yCoord) / barSpacing) + 1;
		    	
		    	if(startThreadElement < 0)
		    		startThreadElement = 0;
		    		
		    	if(endThreadElement < 0)
		    		endThreadElement = 0;
		    	
		    	if(startThreadElement > (tmpThreadDataElementList.size() - 1))
		    		startThreadElement = (tmpThreadDataElementList.size() - 1);
		    		
		    	if(endThreadElement > (tmpThreadDataElementList.size() - 1))
		    		endThreadElement = (tmpThreadDataElementList.size() - 1);
		    	
		    	yCoord = yCoord + (startThreadElement * barSpacing);
		    	
				//Test for the different menu options for this window.
				if((tDWindow.isInclusive()))
				{
				
					//@@@In the inclusive section. - This comment aids in matching up if/else statements.@@@
				
					
					double maxValue = tmpGT.getMaxInclusivePercentValue();
					int tmpValue = (int) ((maxValue / 100.00) * (defaultBarLength));
					barXCoord = tmpValue + 60;
					
					if(tDWindow.isPercent())
					{
						
						//@@@In the percent section. - This comment aids in matching up if/else statements.@@@
								
						
						//Need to figure out how long the percentage string will be.
						tmpString = new String(maxValue + "%");
						stringWidth = fmFont.stringWidth(tmpString);
						barXCoord = barXCoord + stringWidth;
						
						for(int i = startThreadElement; i <= endThreadElement; i++)
		    			{		
		    				tmpSMWThreadDataElement = (SMWThreadDataElement) tmpThreadDataElementList.elementAt(i);
							
							yCoord = yCoord + (barSpacing);
							
							tmpDataValue = tmpSMWThreadDataElement.getInclusivePercentValue();
							
							int xLength;
							double tmpDouble;
							tmpDouble = (tmpDataValue / 100.00);
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
								
								if((tmpSMWThreadDataElement.getMappingID()) == (jRacy.clrChooser.getHighlightColorMappingID()))
								{
									g.setColor(jRacy.clrChooser.getHighlightColor());
									g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
									g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
								}
								else if((tmpSMWThreadDataElement.isGroupMember(jRacy.clrChooser.getGHCMID())))
								{
									g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
								if((tmpSMWThreadDataElement.getMappingID()) == (jRacy.clrChooser.getHighlightColorMappingID()))
									g.setColor(jRacy.clrChooser.getHighlightColor());
								else if((tmpSMWThreadDataElement.isGroupMember(jRacy.clrChooser.getGHCMID())))
									g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
							tmpString = new String(tmpDataValue + "%");
							stringWidth = fmFont.stringWidth(tmpString);
							//Now draw the percent value to the left of the bar.
							stringStart = barXCoord - xLength - stringWidth - 5;
							g.drawString(tmpDataValue + "%", stringStart, yCoord);
							
							//Now print the name of the mapping to the right of the bar.
							tmpString = tmpSMWThreadDataElement.getMappingName();
							g.drawString(tmpString, (barXCoord + 5), yCoord);
							
							//Figure out how wide that string was for x coord reasons.
							stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
							if(tmpXWidthCalc < stringWidth)
							{
								tmpXWidthCalc = stringWidth + 15;
							}
							
							//Update the drawing coordinates.
							tmpSMWThreadDataElement.setTDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
						}
					}
					else
					{//@@@ End - isPercent
					
					
						//@@@In the value section. - This comment aids in matching up if/else statements.@@@
					
				
						//Now get the raw values for printing at the end of the bars.
						tmpDataValue = tmpGT.getMaxInclusiveValue();
						//Check to see what the units are.
						if((tDWindow.units()).equals("Seconds"))
						{
							tmpString = new String((Double.toString((tmpDataValue / 1000000.00))));
							stringWidth = fmFont.stringWidth(tmpString);
							barXCoord = barXCoord + stringWidth;
						}
						else if((tDWindow.units()).equals("Milliseconds"))
						{
							tmpString = new String((Double.toString((tmpDataValue / 1000))));
							stringWidth = fmFont.stringWidth(tmpString);
							barXCoord = barXCoord + stringWidth;
						}
						else
						{
							tmpString = new String(Double.toString(tmpDataValue));
							stringWidth = fmFont.stringWidth(tmpString);
							barXCoord = barXCoord + stringWidth;
						}
						
						for(int i = startThreadElement; i <= endThreadElement; i++)
		    			{		
		    				tmpSMWThreadDataElement = (SMWThreadDataElement) tmpThreadDataElementList.elementAt(i);
							
							yCoord = yCoord + (barSpacing);
							
							tmpDataValue = tmpSMWThreadDataElement.getInclusivePercentValue();
							
							int xLength;
							double tmpDouble;
							tmpDouble = (tmpDataValue / 100.00);
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
								
								if((tmpSMWThreadDataElement.getMappingID()) == (jRacy.clrChooser.getHighlightColorMappingID()))
								{
									g.setColor(jRacy.clrChooser.getHighlightColor());
									g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
									g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
								}
								else if((tmpSMWThreadDataElement.isGroupMember(jRacy.clrChooser.getGHCMID())))
								{
									g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
								if((tmpSMWThreadDataElement.getMappingID()) == (jRacy.clrChooser.getHighlightColorMappingID()))
									g.setColor(jRacy.clrChooser.getHighlightColor());
								else if((tmpSMWThreadDataElement.isGroupMember(jRacy.clrChooser.getGHCMID())))
									g.setColor(jRacy.clrChooser.getGroupHighlightColor());
								else
								{
									tmpColor = tmpSMWThreadDataElement.getMappingColor();
									g.setColor(tmpColor);
								}
								
								g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
							}
							
							//Now print the percentage to the left of the bar.
							g.setColor(Color.black);
							
							//Now get the raw values for printing at the end of the bars.
							tmpDataValue = tmpSMWThreadDataElement.getInclusiveValue();
							//Check to see what the units are.
							if((tDWindow.units()).equals("Seconds"))
							{
								tmpString = new String((Double.toString((tmpDataValue / 1000000.00))));
								stringWidth = fmFont.stringWidth(tmpString);
								stringStart = barXCoord - xLength - stringWidth - 5;
								g.drawString((Double.toString((tmpDataValue / 1000000.00))), stringStart, yCoord);
							}
							else if((tDWindow.units()).equals("Milliseconds"))
							{
								tmpString = new String((Double.toString((tmpDataValue / 1000))));
								stringWidth = fmFont.stringWidth(tmpString);
								stringStart = barXCoord - xLength - stringWidth - 5;
								g.drawString((Double.toString((tmpDataValue / 1000))), stringStart, yCoord);
							}
							else
							{
								tmpString = new String(Double.toString(tmpDataValue));
								stringWidth = fmFont.stringWidth(tmpString);
								stringStart = barXCoord - xLength - stringWidth - 5;
								g.drawString((Double.toString(tmpDataValue)), stringStart, yCoord);
							}				
							
							//Now print the name of the mapping to the right of the bar.
							tmpString = tmpSMWThreadDataElement.getMappingName();
							g.drawString(tmpString, (barXCoord + 5), yCoord);
							
							//Figure out how wide that string was for x coord reasons.
							stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
							if(tmpXWidthCalc < stringWidth)
							{
								tmpXWidthCalc = stringWidth + 15;
							}
							
							//Update the drawing coordinates.
							tmpSMWThreadDataElement.setTDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
						}
					}
				}
				else	//Case of exclusive selection.
				{
					
					
					//@@@In the exclusive section. - This comment aids in matching up if/else statements.@@@
					
					
					
					//Case of exclusive selection.
					//Get the maximum on this thread, and set the draw positions appropriately.
					double maxValue = tmpGT.getMaxExclusivePercentValue();
					int tmpValue = (int) ((maxValue / 100.00) * (defaultBarLength));
					barXCoord = tmpValue + 60;
					
					if(tDWindow.isPercent())
					{	
						
						
						//@@@In the percent section. - This comment aids in matching up if/else statements.@@@
						
						
						//Need to figure out how long the percentage string will be.
						tmpString = new String(maxValue + "%");
						stringWidth = fmFont.stringWidth(tmpString);
						barXCoord = barXCoord + stringWidth;
						
						for(int i = startThreadElement; i <= endThreadElement; i++)
		    			{		
		    				tmpSMWThreadDataElement = (SMWThreadDataElement) tmpThreadDataElementList.elementAt(i);
							
							yCoord = yCoord + (barSpacing);
							
							tmpDataValue = tmpSMWThreadDataElement.getExclusivePercentValue();
							
							int xLength;
							double tmpDouble;
							tmpDouble = (tmpDataValue / 100.00);
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
								
								if((tmpSMWThreadDataElement.getMappingID()) == (jRacy.clrChooser.getHighlightColorMappingID()))
								{
									g.setColor(jRacy.clrChooser.getHighlightColor());
									g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
									g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
								}
								else if((tmpSMWThreadDataElement.isGroupMember(jRacy.clrChooser.getGHCMID())))
								{
									g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
								if((tmpSMWThreadDataElement.getMappingID()) == (jRacy.clrChooser.getHighlightColorMappingID()))
									g.setColor(jRacy.clrChooser.getHighlightColor());
								else if((tmpSMWThreadDataElement.isGroupMember(jRacy.clrChooser.getGHCMID())))
									g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
							tmpString = new String(tmpDataValue + "%");
							stringWidth = fmFont.stringWidth(tmpString);
							stringStart = barXCoord - xLength - stringWidth - 5;
							//Now draw the percent value to the left of the bar.
							g.drawString(tmpDataValue + "%", stringStart, yCoord);
							
							//Now print the name of the mapping to the right of the bar.
							tmpString = tmpSMWThreadDataElement.getMappingName();
							g.drawString(tmpString, (barXCoord + 5), yCoord);
							
							//Figure out how wide that string was for x coord reasons.
							stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
							if(tmpXWidthCalc < stringWidth)
							{
								tmpXWidthCalc = stringWidth + 15;
							}
							
							//Update the drawing coordinates.
							tmpSMWThreadDataElement.setTDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
						}
					}
					else //@@@ End - isPercent.
					{
					
					
						//@@@In the value section. - This comment aids in matching up if/else statements.@@@
					
					
						//Add the correct amount to barXCoord.
						tmpDataValue = tmpGT.getMaxExclusiveValue();
						
						if((tDWindow.units()).equals("Seconds"))
						{
							tmpString = new String((Double.toString((tmpDataValue / 1000000.00))));
							stringWidth = fmFont.stringWidth(tmpString);
							barXCoord = barXCoord + stringWidth;
						}
						else if((tDWindow.units()).equals("Milliseconds"))
						{
							tmpString = new String((Double.toString((tmpDataValue / 1000))));
							stringWidth = fmFont.stringWidth(tmpString);
							barXCoord = barXCoord + stringWidth;
						}
						else
						{
							tmpString = new String(Double.toString(tmpDataValue));
							stringWidth = fmFont.stringWidth(tmpString);
							barXCoord = barXCoord + stringWidth;
						}
						
						
						
						for(int i = startThreadElement; i <= endThreadElement; i++)
		    			{		
		    				tmpSMWThreadDataElement = (SMWThreadDataElement) tmpThreadDataElementList.elementAt(i);
							
							yCoord = yCoord + (barSpacing);
							
							tmpDataValue = tmpSMWThreadDataElement.getExclusivePercentValue();
							
							int xLength;
							double tmpDouble;
							tmpDouble = (tmpDataValue / 100.00);
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
								
								if((tmpSMWThreadDataElement.getMappingID()) == (jRacy.clrChooser.getHighlightColorMappingID()))
								{
									g.setColor(jRacy.clrChooser.getHighlightColor());
									g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
									g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
								}
								else if((tmpSMWThreadDataElement.isGroupMember(jRacy.clrChooser.getGHCMID())))
								{
									g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
								if((tmpSMWThreadDataElement.getMappingID()) == (jRacy.clrChooser.getHighlightColorMappingID()))
									g.setColor(jRacy.clrChooser.getHighlightColor());
								else if((tmpSMWThreadDataElement.isGroupMember(jRacy.clrChooser.getGHCMID())))
									g.setColor(jRacy.clrChooser.getGroupHighlightColor());
								else
								{
									tmpColor = tmpSMWThreadDataElement.getMappingColor();
									g.setColor(tmpColor);
								}
								
								g.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
							}
							
							//Now print the percentage to the left of the bar.
							g.setColor(Color.black);
							
							//Now get the raw values for printing at the end of the bars.
							tmpDataValue = tmpSMWThreadDataElement.getExclusiveValue();
							
							//Check to see what the units are.
							if((tDWindow.units()).equals("Seconds"))
							{
								tmpString = new String((Double.toString((tmpDataValue / 1000000.00))));
								stringWidth = fmFont.stringWidth(tmpString);
								stringStart = barXCoord - xLength - stringWidth - 5;
								g.drawString((Double.toString((tmpDataValue / 1000000.00))), stringStart, yCoord);
							}
							else if((tDWindow.units()).equals("Milliseconds"))
							{
								tmpString = new String((Double.toString((tmpDataValue / 1000))));
								stringWidth = fmFont.stringWidth(tmpString);
								stringStart = barXCoord - xLength - stringWidth - 5;
								g.drawString((Double.toString((tmpDataValue / 1000))), stringStart, yCoord);
							}
							else
							{
								tmpString = new String(Double.toString(tmpDataValue));
								stringWidth = fmFont.stringWidth(tmpString);
								stringStart = barXCoord - xLength - stringWidth - 5;
								g.drawString((Double.toString(tmpDataValue)), stringStart, yCoord);
							}				
							
							//Now print the name of the mapping to the right of the bar.
							tmpString = tmpSMWThreadDataElement.getMappingName();
							g.drawString(tmpString, (barXCoord + 5), yCoord);
							
							//Figure out how wide that string was for x coord reasons.
							stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
							if(tmpXWidthCalc < stringWidth)
							{
								tmpXWidthCalc = stringWidth + 15;
							}
							
							//Update the drawing coordinates.
							tmpSMWThreadDataElement.setTDWDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);											
						}
					}
				}
			}
					
			boolean sizeChange = false;		
			//Resize the panel if needed.
			if(tmpXWidthCalc > 550){
				xPanelSize = tmpXWidthCalc + 1;
				sizeChange = true;
			}
			
			if(newYPanelSize > 550){
				yPanelSize = newYPanelSize + 1;
				sizeChange = true;
			}
			
			if(sizeChange)
				revalidate();
		}
		catch(Exception e)
		{
			System.out.println(e);
			jRacy.systemError(null, "TDWP03");
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
			
			SMWThreadDataElement tmpSMWThreadDataElement = null;
			
			if(EventSrc instanceof JMenuItem)
			{
				String arg = evt.getActionCommand();
				if(arg.equals("Show Mapping Details"))
				{
					
					if(clickedOnObject instanceof SMWThreadDataElement)
					{
						tmpSMWThreadDataElement = (SMWThreadDataElement) clickedOnObject;
						//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
						jRacy.clrChooser.setHighlightColorMappingID(tmpSMWThreadDataElement.getMappingID());
						MappingDataWindow tmpRef = new MappingDataWindow(tmpSMWThreadDataElement.getMappingName(), sMWData);
						jRacy.systemEvents.addObserver(tmpRef);
						tmpRef.show();
					}
				}
				else if(arg.equals("Change Mapping Color"))
				{	
					int mappingID = -1;
					
					//Get the clicked on object.
					if(clickedOnObject instanceof SMWThreadDataElement)
						mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
					
					GlobalMapping globalMappingReference = jRacy.staticSystemData.getGlobalMapping();
					GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
					
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
					GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
					
					tmpGME.setColorFlag(false);
					jRacy.systemEvents.updateRegisteredObjects("colorEvent");
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TDWP04");
		}
	}
	
	//Ok, now the mouse listeners for this panel.
	public void mouseClicked(MouseEvent evt)
	{
		try
		{
			//Get the location of the mouse.
			int xCoord = evt.getX();
			int yCoord = evt.getY();
			
			//Get the number of times clicked.
			int clickCount = evt.getClickCount();
			
			for(Enumeration e1 = tmpThreadDataElementList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
									
				if(yCoord <= (tmpSMWThreadDataElement.getTDWYEnd()))
				{
					if((yCoord >= (tmpSMWThreadDataElement.getTDWYBeg())) && (xCoord >= (tmpSMWThreadDataElement.getTDWXBeg()))
																		  && (xCoord <= (tmpSMWThreadDataElement.getTDWXEnd())))
					{
						if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0)
						{
							//Set the clickedSMWMeanDataElement.
							clickedOnObject = tmpSMWThreadDataElement;
							popup.show(this, evt.getX(), evt.getY());
							
							//Return from this function.
							return;
						}
						else
						{
							//Want to set the clicked on mapping to the current highlight color or, if the one
							//clicked on is already the current highlighted one, set it back to normal.
							if((jRacy.clrChooser.getHighlightColorMappingID()) == -1)
							{
								jRacy.clrChooser.setHighlightColorMappingID(tmpSMWThreadDataElement.getMappingID());
							}
							else
							{
								if(!((jRacy.clrChooser.getHighlightColorMappingID()) == (tmpSMWThreadDataElement.getMappingID())))
									jRacy.clrChooser.setHighlightColorMappingID(tmpSMWThreadDataElement.getMappingID());
								else
									jRacy.clrChooser.setHighlightColorMappingID(-1);
							}
						}
						//Nothing more to do ... return.
						return;
					}
					else
					{
						//If we get here, it means that we are outside the mapping draw area.  That is, we
						//are either to the left or right of the draw area, or just above it.
						//It is better to return here as we do not want the sysstem to cycle through the
						//rest of the objects, which would be pointless as we know that it will not be
						//one of the others.  Significantly improves performance.
						return;
					}
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "TDWP05");
			System.out.println("Please email Robert Bell at: bertie@cs.uoregon.edu");
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
			double sliderValue = (double) tDWindow.getSliderValue();
			double sliderMultiple = tDWindow.getSliderMultiple();
			double result = 250*sliderValue*sliderMultiple;
			
			defaultBarLength = (int) result;
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "FDWP06");
		}
	}
	
	//******************************
	//Instance data.
	//******************************
	private int newXPanelSize = 0;
	private int newYPanelSize = 0;
	
	String counterName = null;
	
	private int barHeight = -1;
	private int barSpacing = -1;
	private int defaultBarLength = 250;
	private int maxXLength = 0;
	private int numberOfColors = 0;
	
	private int serverNumber = -1;
	private int	contextNumber = -1;
	private int	threadNumber = -1;
	
	private StaticMainWindowData sMWData = null;
	private ThreadDataWindow tDWindow = null;
	private GlobalThread tmpGT = null;	
 	private Vector tmpThreadDataElementList = null;
 	private SMWThreadDataElement tmpSMWThreadDataElement = null;
 	
 	//**********
	//Popup menu definitions.
	private JPopupMenu popup = new JPopupMenu();
	//**********
 	
 	//**********
	//Other useful variables.
	private Object clickedOnObject = null;
	//End - Other useful variables.
	//**********
	
	//******************************
	//End - Instance data.
	//******************************
}