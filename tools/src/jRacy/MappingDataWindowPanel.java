/* 
	
	MappingDataWindowPanel.java
	
	Title:			jRacy
	Author:			Robert Bell
	Description:
	
	Things to do:
	
	1) Add clipping support to this window.	
*/

package jRacy;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;


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
			jRacy.systemError(null, "MDWP01");
		}
	
	}
	
	
	public MappingDataWindowPanel(int inMappingID, MappingDataWindow inMDWindow)
	{
		try{
			setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
			setBackground(Color.white);
			
			//Add this object as a mouse listener.
			addMouseListener(this);
			
			
			//Grab the appropriate global mapping element.
			GlobalMapping tmpGM = jRacy.staticSystemData.getGlobalMapping();
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
			jRacy.systemError(null, "MDWP02");
		}
	
	}
	

	public void paintComponent(Graphics g)
	{
		try
		{
			super.paintComponent(g);
			
			//Do the standard font and spacing stuff.
			if(!(jRacy.jRacyPreferences.areBarDetailsSet()))
			{
				
				//Create font.
				Font font = new Font(jRacy.jRacyPreferences.getJRacyFont(), jRacy.jRacyPreferences.getFontStyle(), 12);
				g.setFont(font);
				FontMetrics fmFont = g.getFontMetrics(font);
				
				//Set up the bar details.
				
				//Compute the font metrics.
				int maxFontAscent = fmFont.getAscent();
				int maxFontDescent = fmFont.getMaxDescent();
				
				int tmpInt = maxFontAscent + maxFontDescent;
				
				jRacy.jRacyPreferences.setBarDetails(maxFontAscent, (tmpInt + 5));
				
				jRacy.jRacyPreferences.setSliders(maxFontAscent, (tmpInt + 5));
			}
			
			//Set local spacing and bar heights.
			barSpacing = jRacy.jRacyPreferences.getBarSpacing();
			barHeight = jRacy.jRacyPreferences.getBarHeight();
			
			//Create font.
			Font font = new Font(jRacy.jRacyPreferences.getJRacyFont(), jRacy.jRacyPreferences.getFontStyle(), barHeight);
			g.setFont(font);
			FontMetrics fmFont = g.getFontMetrics(font);

			double tmpSum;
			double tmpDataValue;
			Color tmpColor;
			String tmpString;
			int stringWidth;
			int stringStart;
			
			//Convenient counters.
			int colorCounter = 0;
		
			int yCoord = 0;
			
			int tmpXWidthCalc = 0;
			
			//An XCoord used in drawing the bars.
			int barXCoord = defaultBarLength + 60;
			yCoord = 0;
			
			//Grab the appropriate global mapping element.
			GlobalMapping tmpGM = jRacy.staticSystemData.getGlobalMapping();
			GlobalMappingElement tmpGME = tmpGM.getGlobalMappingElement(mappingName, 0);
			
			mappingID = tmpGME.getGlobalID();
			
			//Set the max values for this mapping.
			maxInclusiveValue = tmpGME.getMaxInclusiveValue();
			maxExclusiveValue = tmpGME.getMaxExclusiveValue();
			maxInclusivePercentValue = tmpGME.getMaxInclusivePercentValue();
			maxExclusivePercentValue = tmpGME.getMaxExclusivePercentValue();
			
			yCoord = yCoord + (barSpacing);
			
			//**********
			//Draw the counter name if required.
			counterName = jRacy.staticSystemData.getCounterName();
			if(counterName != null){
				g.drawString("COUNTER NAME: " + counterName, 5, yCoord);
				yCoord = yCoord + (barSpacing);
			}
			//End - Draw the counter name if required.
			//**********
			
			
			//**********
			//Draw the mapping name.
			g.drawString("FUNCTION NAME: " + mappingName, 5, yCoord);
			//Calculate its width.
			tmpXWidthCalc = fmFont.stringWidth(mappingName);
			yCoord = yCoord + (barSpacing);
			//End - Draw the mapping name.
			//**********
			
			//Get some string lengths.
			
			if((mDWindow.isInclusive())){
				if(mDWindow.isPercent()){
					//Need to figure out how long the percentage string will be.
					tmpString = new String(maxInclusivePercentValue + "%");
					stringWidth = fmFont.stringWidth(tmpString);
					barXCoord = barXCoord + stringWidth;
				}
				else{
					//Check to see what the units are.
					if((mDWindow.units()).equals("Seconds"))
					{
						tmpString = new String((Double.toString((maxInclusiveValue / 1000000.00))));
						stringWidth = fmFont.stringWidth(tmpString);
						barXCoord = barXCoord + stringWidth;
					}
					else if((mDWindow.units()).equals("Milliseconds"))
					{
						tmpString = new String((Double.toString((maxInclusiveValue / 1000))));
						stringWidth = fmFont.stringWidth(tmpString);
						barXCoord = barXCoord + stringWidth;
					}
					else
					{
						tmpString = new String(Double.toString(maxInclusiveValue));
						stringWidth = fmFont.stringWidth(tmpString);
						barXCoord = barXCoord + stringWidth;
					}
				}
			}
			else{
				if(mDWindow.isPercent()){
					//Need to figure out how long the percentage string will be.
					tmpString = new String(maxExclusivePercentValue + "%");
					stringWidth = fmFont.stringWidth(tmpString);
					barXCoord = barXCoord + stringWidth;
				}
				else{
				
					//Add the correct amount to barXCoord.
					if((mDWindow.units()).equals("Seconds"))
					{
						tmpString = new String((Double.toString((maxExclusiveValue / 1000000.00))));
						stringWidth = fmFont.stringWidth(tmpString);
						barXCoord = barXCoord + stringWidth;
					}
					else if((mDWindow.units()).equals("Milliseconds"))
					{
						tmpString = new String((Double.toString((maxExclusiveValue / 1000))));
						stringWidth = fmFont.stringWidth(tmpString);
						barXCoord = barXCoord + stringWidth;
					}
					else
					{
						tmpString = new String(Double.toString(maxExclusiveValue));
						stringWidth = fmFont.stringWidth(tmpString);
						barXCoord = barXCoord + stringWidth;
					}
				}
			}
			
			//******************************
			//Do the mean bar.
			//******************************
			
			//Build the node,context,thread string.
			String s1 = "mean";
			
			//Test for the different menu options for this window.
			if((mDWindow.isInclusive()))
			{	
				if(mDWindow.isPercent())
				{		
					yCoord = yCoord + (barSpacing);
					
					tmpDataValue = tmpGME.getMeanInclusivePercentValue();
					
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
						
						if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
						{
							g.setColor(jRacy.clrChooser.getHighlightColor());
							g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
							g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
						}
						else if((tmpGME.isGroupMember(jRacy.clrChooser.getGHCMID())))
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
						if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
							g.setColor(jRacy.clrChooser.getHighlightColor());
						else if((tmpGME.isGroupMember(jRacy.clrChooser.getGHCMID())))
							g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
					tmpString = new String(tmpDataValue + "%");
					stringWidth = fmFont.stringWidth(tmpString);
					//Now draw the percent value to the left of the bar.
					stringStart = barXCoord - xLength - stringWidth - 5;
					g.drawString(tmpDataValue + "%", stringStart, yCoord);
					
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
					tmpDataValue = tmpGME.getMeanInclusiveValue();
					
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
						
						if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
						{
							g.setColor(jRacy.clrChooser.getHighlightColor());
							g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
							g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
						}
						else if((tmpGME.isGroupMember(jRacy.clrChooser.getGHCMID())))
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
						if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
							g.setColor(jRacy.clrChooser.getHighlightColor());
						else if((tmpGME.isGroupMember(jRacy.clrChooser.getGHCMID())))
							g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
						tmpString = new String((Double.toString((tmpDataValue / 1000000.00))));
						stringWidth = fmFont.stringWidth(tmpString);
						stringStart = barXCoord - xLength - stringWidth - 5;
						g.drawString((Double.toString((tmpDataValue / 1000000.00))), stringStart, yCoord);
					}
					else if((mDWindow.units()).equals("Milliseconds"))
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
			else	//Case of exclusive selection.
			{	
				if(mDWindow.isPercent())
				{
					//For consistancy in drawing, the y coord is updated at the beggining of the loop.
					yCoord = yCoord + (barSpacing);
					
					tmpDataValue = tmpGME.getMeanExclusivePercentValue();
					
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
						
						if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
						{
							g.setColor(jRacy.clrChooser.getHighlightColor());
							g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
							g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
						}
						else if((tmpGME.isGroupMember(jRacy.clrChooser.getGHCMID())))
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
						if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
							g.setColor(jRacy.clrChooser.getHighlightColor());
						else if((tmpGME.isGroupMember(jRacy.clrChooser.getGHCMID())))
							g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
					tmpString = new String(tmpDataValue + "%");
					stringWidth = fmFont.stringWidth(tmpString);
					stringStart = barXCoord - xLength - stringWidth - 5;
					//Now draw the percent value to the left of the bar.
					g.drawString(tmpDataValue + "%", stringStart, yCoord);
					
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
					tmpDataValue = tmpGME.getMeanExclusiveValue();
					
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
						
						if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
						{
							g.setColor(jRacy.clrChooser.getHighlightColor());
							g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
							g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
						}
						else if((tmpGME.isGroupMember(jRacy.clrChooser.getGHCMID())))
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
						if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
							g.setColor(jRacy.clrChooser.getHighlightColor());
						else if((tmpGME.isGroupMember(jRacy.clrChooser.getGHCMID())))
							g.setColor(jRacy.clrChooser.getGroupHighlightColor());
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
						tmpString = new String((Double.toString((tmpDataValue / 1000000.00))));
						stringWidth = fmFont.stringWidth(tmpString);
						stringStart = barXCoord - xLength - stringWidth - 5;
						g.drawString((Double.toString((tmpDataValue / 1000000.00))), stringStart, yCoord);
					}
					else if((mDWindow.units()).equals("Milliseconds"))
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
			
			//******************************
			//End - Do the mean bar.
			//******************************
			
			
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
						if((mDWindow.isInclusive()))
						{
							if(mDWindow.isPercent())
							{
								for(Enumeration e4 = tmpThreadDataElementList.elements(); e4.hasMoreElements() ;)
								{
									tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
									
									if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
									{
										//For consistancy in drawing, the y coord is updated at the beggining of the loop.
										yCoord = yCoord + (barSpacing);
										
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
											if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
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
							else
							{//@@@
								for(Enumeration e5 = tmpThreadDataElementList.elements(); e5.hasMoreElements() ;)
								{
									tmpSMWThreadDataElement = (SMWThreadDataElement) e5.nextElement();
									
									if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
									{
										//For consistancy in drawing, the y coord is updated at the beggining of the loop.
										yCoord = yCoord + (barSpacing);
										
										
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
											if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
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
										
										//Check to see what the units are.
										if((mDWindow.units()).equals("Seconds"))
										{
											tmpString = new String((Double.toString((tmpDataValue / 1000000.00))));
											stringWidth = fmFont.stringWidth(tmpString);
											stringStart = barXCoord - xLength - stringWidth - 5;
											g.drawString((Double.toString((tmpDataValue / 1000000.00))), stringStart, yCoord);
										}
										else if((mDWindow.units()).equals("Milliseconds"))
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
						else	//Case of exclusive selection.
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
											if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
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
							else
							{
								for(Enumeration e7 = tmpThreadDataElementList.elements(); e7.hasMoreElements() ;)
								{
									tmpSMWThreadDataElement = (SMWThreadDataElement) e7.nextElement();
									
									if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
									{
										//For consistancy in drawing, the y coord is updated at the beggining of the loop.
										yCoord = yCoord + (barSpacing);
										
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
											if(mappingID == (jRacy.clrChooser.getHighlightColorMappingID()))
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
										
										//Check to see what the units are.
										if((mDWindow.units()).equals("Seconds"))
										{
											tmpString = new String((Double.toString((tmpDataValue / 1000000.00))));
											stringWidth = fmFont.stringWidth(tmpString);
											stringStart = barXCoord - xLength - stringWidth - 5;
											g.drawString((Double.toString((tmpDataValue / 1000000.00))), stringStart, yCoord);
										}
										else if((mDWindow.units()).equals("Milliseconds"))
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
			jRacy.systemError(null, "MDWP03");
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
					GlobalMapping globalMappingReference = jRacy.staticSystemData.getGlobalMapping();
					GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
					
					tmpGME.setColorFlag(false);
					jRacy.systemEvents.updateRegisteredObjects("colorEvent");
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "MDWP04");
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
			jRacy.systemError(null, "MDWP05");
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
			jRacy.systemError(null, "MDWP06");
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
	
	private int serverNumber = -1;
	private int	contextNumber = -1;
	private int	threadNumber = -1;
	
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

