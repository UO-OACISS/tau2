/* 
	
	MeanDataWindowPanel.java
	
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


public class MeanDataWindowPanel extends JPanel implements ActionListener, MouseListener
{
	int xPanelSize = 700;
	int yPanelSize = 450;
	
	public MeanDataWindowPanel()
	{
		try{
			setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
			
			//Schedule a repaint of this panel.
			this.repaint();
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "MDWP01");
		}
	
	}
	
	
	public MeanDataWindowPanel(Trial inTrial, MeanDataWindow inMDWindow, StaticMainWindowData inSMWData)
	{
		try{
			setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
			setBackground(Color.white);
			
			//Add this object as a mouse listener.
			addMouseListener(this);
			
			staticNodeList = inTrial.getStaticServerList();

			trial = inTrial;
			mDWindow = inMDWindow;
			sMWData = inSMWData;
			
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
			
			//Schedule a repaint of this panel.
			this.repaint();
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "MDWP02");
		}
	
	}
	

	public void paintComponent(Graphics g)
	{
		try{
			super.paintComponent(g);
			
			//Set the numberOfColors variable.
			numberOfColors = trial.getColorChooser().getNumberOfColors();
			
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
			
			//Create font.
			Font font = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), barHeight);
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
			
			//Grab the appropriate thread.
			tmpMeanDataElementList = mDWindow.getStaticMainWindowSystemData();
			
			//With group support present, it is possible that the number of mappings in
			//our data list is zero.  If so, just return.
			if((tmpMeanDataElementList.size()) == 0)
				return;
			
			//Cycle through the data values for this thread to get the total.
			tmpSum = 0.00;
			
			Rectangle clipRect = g.getClipBounds();
			
			int yBeg = (int) clipRect.getY();
			int yEnd = (int) (yBeg + clipRect.getHeight());
			int startMeanElement = 0;
			int endMeanElement = 0;
			
		    //To be on the safe side, have an alternative to the clip rectangle.
		    if ((clipRect != null))
		    {	
		    	//@@@In the clipping section. - This comment aids in matching up if/else statements.@@@
		    	
		    	
		    	//Set up some panel dimensions.
		    	newYPanelSize = yCoord + ((tmpMeanDataElementList.size() + 1) * barSpacing);
		    	
		    	startMeanElement = ((yBeg - yCoord) / barSpacing) - 1;
		    	endMeanElement  = ((yEnd - yCoord) / barSpacing) + 1;
		    	
		    	if(startMeanElement < 0)
		    		startMeanElement = 0;
		    		
		    	if(endMeanElement < 0)
		    		endMeanElement = 0;
		    	
		    	if(startMeanElement > (tmpMeanDataElementList.size() - 1))
		    		startMeanElement = (tmpMeanDataElementList.size() - 1);
		    		
		    	if(endMeanElement > (tmpMeanDataElementList.size() - 1))
		    		endMeanElement = (tmpMeanDataElementList.size() - 1);
		    	
		    	yCoord = yCoord + (startMeanElement * barSpacing);
		    	
		    	//Set the max values for this mapping.
				maxInclusiveValue = trial.getMaxMeanInclusiveValue(trial.getCurRunValLoc());
				maxExclusiveValue = trial.getMaxMeanExclusiveValue(trial.getCurRunValLoc());
				maxInclusivePercentValue = trial.getMaxMeanInclusivePercentValue(trial.getCurRunValLoc());
				maxExclusivePercentValue = trial.getMaxMeanExclusivePercentValue(trial.getCurRunValLoc());
				maxNumberOfCalls = trial.getMaxMeanNumberOfCalls();
				maxNumberOfSubroutines = trial.getMaxMeanNumberOfSubRoutines();
				maxUserSecPerCall = trial.getMaxMeanUserSecPerCall(trial.getCurRunValLoc());
				
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
		    	
				//Test for the different menu options for this window.
				if((mDWindow.getMetric()).equals("Inclusive"))
				{
					if(mDWindow.isPercent())
					{
						for(int i = startMeanElement; i <= endMeanElement; i++)
		    			{		
		    				tmpSMWMeanDataElement = (SMWMeanDataElement) tmpMeanDataElementList.elementAt(i);
							
							yCoord = yCoord + (barSpacing);
							
							tmpDataValue = tmpSMWMeanDataElement.getMeanInclusivePercentValue();
							
							int xLength;
							double tmpDouble;
							tmpDouble = (tmpDataValue / maxInclusivePercentValue);
							xLength = (int) (tmpDouble * defaultBarLength);
							if(xLength == 0)
								xLength = 1;
							
							//Now set the color values for drawing!
							//Get the appropriate color.
							tmpColor = tmpSMWMeanDataElement.getMappingColor();
							g.setColor(tmpColor);
							
							if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
							{
								g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
								
								if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
								{
									g.setColor(trial.getColorChooser().getHighlightColor());
									g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
									g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
								}
								else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
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
								if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
									g.setColor(trial.getColorChooser().getHighlightColor());
								else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
									g.setColor(trial.getColorChooser().getGroupHighlightColor());
								else
								{
									tmpColor = tmpSMWMeanDataElement.getMappingColor();
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
							tmpString = tmpSMWMeanDataElement.getMappingName();
							g.drawString(tmpString, (barXCoord + 5), yCoord);
							
							//Figure out how wide that string was for x coord reasons.
							stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
							if(tmpXWidthCalc < stringWidth)
							{
								tmpXWidthCalc = stringWidth + 15;
							}
							
							//Update the drawing coordinates.
							tmpSMWMeanDataElement.setDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
																			
						}
					}
					else
					{//@@@ End - isPercent
					
					
						//@@@In the value section. - This comment aids in matching up if/else statements.@@@
					
						for(int i = startMeanElement; i <= endMeanElement; i++)
		    			{		
		    				tmpSMWMeanDataElement = (SMWMeanDataElement) tmpMeanDataElementList.elementAt(i);
							
							yCoord = yCoord + (barSpacing);
							
							tmpDataValue = tmpSMWMeanDataElement.getMeanInclusiveValue();
							
							int xLength;
							double tmpDouble;
							tmpDouble = (tmpDataValue / maxInclusiveValue);
							xLength = (int) (tmpDouble * defaultBarLength);
							if(xLength == 0)
								xLength = 1;
							
							//Now set the color values for drawing!
							//Get the appropriate color.
							tmpColor = tmpSMWMeanDataElement.getMappingColor();
							g.setColor(tmpColor);
							
							if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
							{
								g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
								
								if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
								{
									g.setColor(trial.getColorChooser().getHighlightColor());
									g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
									g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
								}
								else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
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
								if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
									g.setColor(trial.getColorChooser().getHighlightColor());
								else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
									g.setColor(trial.getColorChooser().getGroupHighlightColor());
								else
								{
									tmpColor = tmpSMWMeanDataElement.getMappingColor();
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
							
							//Now print the name of the mapping to the right of the bar.
							tmpString = tmpSMWMeanDataElement.getMappingName();
							g.drawString(tmpString, (barXCoord + 5), yCoord);
							
							//Figure out how wide that string was for x coord reasons.
							stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
							if(tmpXWidthCalc < stringWidth)
							{
								tmpXWidthCalc = stringWidth + 15;
							}
							
							//Update the drawing coordinates.
							tmpSMWMeanDataElement.setDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);																	
						}
					}
				}
				else if((mDWindow.getMetric()).equals("Exclusive"))
				{
					if(mDWindow.isPercent())
					{
						for(int i = startMeanElement; i <= endMeanElement; i++)
		    			{		
		    				tmpSMWMeanDataElement = (SMWMeanDataElement) tmpMeanDataElementList.elementAt(i);
							
							yCoord = yCoord + (barSpacing);
							
							tmpDataValue = tmpSMWMeanDataElement.getMeanExclusivePercentValue();
							
							int xLength;
							double tmpDouble;
							tmpDouble = (tmpDataValue / maxExclusivePercentValue);
							xLength = (int) (tmpDouble * defaultBarLength);
							if(xLength == 0)
								xLength = 1;
								
							//Now set the color values for drawing!
							//Get the appropriate color.
							tmpColor = tmpSMWMeanDataElement.getMappingColor();
							g.setColor(tmpColor);
							
							if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
							{
								g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
								
								if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
								{
									g.setColor(trial.getColorChooser().getHighlightColor());
									g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
									g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
								}
								else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
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
								if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
									g.setColor(trial.getColorChooser().getHighlightColor());
								else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
									g.setColor(trial.getColorChooser().getGroupHighlightColor());
								else
								{
									tmpColor = tmpSMWMeanDataElement.getMappingColor();
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
							tmpString = tmpSMWMeanDataElement.getMappingName();
							
							g.drawString(tmpString, (barXCoord + 5), yCoord);
							
							//Figure out how wide that string was for x coord reasons.
							stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
							if(tmpXWidthCalc < stringWidth)
							{
								tmpXWidthCalc = stringWidth + 15;
							}
							
							//Update the drawing coordinates.
							tmpSMWMeanDataElement.setDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
						}
					}
					else //@@@ End - isPercent.
					{
						for(int i = startMeanElement; i <= endMeanElement; i++)
		    			{		
		    				tmpSMWMeanDataElement = (SMWMeanDataElement) tmpMeanDataElementList.elementAt(i);
							
							yCoord = yCoord + (barSpacing);
							
							tmpDataValue = tmpSMWMeanDataElement.getMeanExclusiveValue();
							
							int xLength;
							double tmpDouble;
							tmpDouble = (tmpDataValue / maxExclusiveValue);
							xLength = (int) (tmpDouble * defaultBarLength);
							if(xLength == 0)
								xLength = 1;
							
							//Now set the color values for drawing!
							//Get the appropriate color.
							tmpColor = tmpSMWMeanDataElement.getMappingColor();
							g.setColor(tmpColor);
							
							if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
							{
								g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
								
								if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
								{
									g.setColor(trial.getColorChooser().getHighlightColor());
									g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
									g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
								}
								else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
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
								if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
									g.setColor(trial.getColorChooser().getHighlightColor());
								else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
									g.setColor(trial.getColorChooser().getGroupHighlightColor());
								else
								{
									tmpColor = tmpSMWMeanDataElement.getMappingColor();
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
							
							//Now print the name of the mapping to the right of the bar.
							tmpString = tmpSMWMeanDataElement.getMappingName();
							g.drawString(tmpString, (barXCoord + 5), yCoord);
							
							//Figure out how wide that string was for x coord reasons.
							stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
							if(tmpXWidthCalc < stringWidth)
							{
								tmpXWidthCalc = stringWidth + 15;
							}
							
							//Update the drawing coordinates.
							tmpSMWMeanDataElement.setDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
						}
					}
				}
				else if((mDWindow.getMetric()).equals("Number of Calls"))
				{
					for(int i = startMeanElement; i <= endMeanElement; i++)
	    			{		
	    				tmpSMWMeanDataElement = (SMWMeanDataElement) tmpMeanDataElementList.elementAt(i);
						
						yCoord = yCoord + (barSpacing);
						
						tmpDataValue = tmpSMWMeanDataElement.getMeanNumberOfCalls();
						
						int xLength;
						double tmpDouble;
						tmpDouble = (tmpDataValue / maxNumberOfCalls);
						xLength = (int) (tmpDouble * defaultBarLength);
						if(xLength == 0)
							xLength = 1;
						
						//Now set the color values for drawing!
						//Get the appropriate color.
						tmpColor = tmpSMWMeanDataElement.getMappingColor();
						g.setColor(tmpColor);
						
						if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
						{
							g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
							
							if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
							{
								g.setColor(trial.getColorChooser().getHighlightColor());
								g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
								g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
							}
							else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
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
							if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
								g.setColor(trial.getColorChooser().getHighlightColor());
							else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
								g.setColor(trial.getColorChooser().getGroupHighlightColor());
							else
							{
								tmpColor = tmpSMWMeanDataElement.getMappingColor();
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
						
						//Now print the name of the mapping to the right of the bar.
						tmpString = tmpSMWMeanDataElement.getMappingName();
						g.drawString(tmpString, (barXCoord + 5), yCoord);
						
						//Figure out how wide that string was for x coord reasons.
						stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
						if(tmpXWidthCalc < stringWidth)
						{
							tmpXWidthCalc = stringWidth + 15;
						}
						
						//Update the drawing coordinates.
						tmpSMWMeanDataElement.setDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
					}
				}
				else if((mDWindow.getMetric()).equals("Number of Subroutines"))
				{
					for(int i = startMeanElement; i <= endMeanElement; i++)
	    			{		
	    				tmpSMWMeanDataElement = (SMWMeanDataElement) tmpMeanDataElementList.elementAt(i);
						
						yCoord = yCoord + (barSpacing);
						
						tmpDataValue = tmpSMWMeanDataElement.getMeanNumberOfSubRoutines();
						
						int xLength;
						double tmpDouble;
						tmpDouble = (tmpDataValue / maxNumberOfSubroutines);
						xLength = (int) (tmpDouble * defaultBarLength);
						if(xLength == 0)
							xLength = 1;
						
						//Now set the color values for drawing!
						//Get the appropriate color.
						tmpColor = tmpSMWMeanDataElement.getMappingColor();
						g.setColor(tmpColor);
						
						if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
						{
							g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
							
							if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
							{
								g.setColor(trial.getColorChooser().getHighlightColor());
								g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
								g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
							}
							else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
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
							if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
								g.setColor(trial.getColorChooser().getHighlightColor());
							else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
								g.setColor(trial.getColorChooser().getGroupHighlightColor());
							else
							{
								tmpColor = tmpSMWMeanDataElement.getMappingColor();
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
						
						//Now print the name of the mapping to the right of the bar.
						tmpString = tmpSMWMeanDataElement.getMappingName();
						g.drawString(tmpString, (barXCoord + 5), yCoord);
						
						//Figure out how wide that string was for x coord reasons.
						stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
						if(tmpXWidthCalc < stringWidth)
						{
							tmpXWidthCalc = stringWidth + 15;
						}
						
						//Update the drawing coordinates.
						tmpSMWMeanDataElement.setDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
					}
				}
				else if((mDWindow.getMetric()).equals("Per Call Value"))
				{
					for(int i = startMeanElement; i <= endMeanElement; i++)
	    			{		
	    				tmpSMWMeanDataElement = (SMWMeanDataElement) tmpMeanDataElementList.elementAt(i);
						
						yCoord = yCoord + (barSpacing);
						
						tmpDataValue = tmpSMWMeanDataElement.getMeanUserSecPerCall();
						
						int xLength;
						double tmpDouble;
						tmpDouble = (tmpDataValue / maxUserSecPerCall);
						xLength = (int) (tmpDouble * defaultBarLength);
						if(xLength == 0)
							xLength = 1;
						
						//Now set the color values for drawing!
						//Get the appropriate color.
						tmpColor = tmpSMWMeanDataElement.getMappingColor();
						g.setColor(tmpColor);
						
						if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
						{
							g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
							
							if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
							{
								g.setColor(trial.getColorChooser().getHighlightColor());
								g.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
								g.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
							}
							else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
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
							if((tmpSMWMeanDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
								g.setColor(trial.getColorChooser().getHighlightColor());
							else if((tmpSMWMeanDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
								g.setColor(trial.getColorChooser().getGroupHighlightColor());
							else
							{
								tmpColor = tmpSMWMeanDataElement.getMappingColor();
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
						
						//Now print the name of the mapping to the right of the bar.
						tmpString = tmpSMWMeanDataElement.getMappingName();
						g.drawString(tmpString, (barXCoord + 5), yCoord);
						
						//Figure out how wide that string was for x coord reasons.
						stringWidth =  (barXCoord + fmFont.stringWidth(tmpString) + 5); 
						if(tmpXWidthCalc < stringWidth)
						{
							tmpXWidthCalc = stringWidth + 15;
						}
						
						//Update the drawing coordinates.
						tmpSMWMeanDataElement.setDrawCoords(stringStart, stringWidth, (yCoord - barHeight), yCoord);
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
			
			/*
			//Resize the panel if needed.
			if((newYPanelSize >= yPanelSize) || (tmpXWidthCalc  >= xPanelSize))
			{
				yPanelSize = newYPanelSize + 1;
				xPanelSize = tmpXWidthCalc + 1;
				
				revalidate();
			}
			*/
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "MDWP03");
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
			
			SMWMeanDataElement tmpSMWMeanDataElement = null;
			
			if(EventSrc instanceof JMenuItem)
			{
				String arg = evt.getActionCommand();
				if(arg.equals("Show Function Details"))
				{
					
					if(clickedOnObject instanceof SMWMeanDataElement)
					{
						tmpSMWMeanDataElement = (SMWMeanDataElement) clickedOnObject;
						//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
						trial.getColorChooser().setHighlightColorMappingID(tmpSMWMeanDataElement.getMappingID());
						MappingDataWindow tmpRef = new MappingDataWindow(trial, tmpSMWMeanDataElement.getMappingID(), sMWData);
						trial.getSystemEvents().addObserver(tmpRef);
						tmpRef.show();
					}
				}
				else if(arg.equals("Change Function Color"))
				{	
					int mappingID = -1;
					
					//Get the clicked on object.
					if(clickedOnObject instanceof SMWMeanDataElement)
						mappingID = ((SMWMeanDataElement) clickedOnObject).getMappingID();
					
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
					if(clickedOnObject instanceof SMWMeanDataElement)
						mappingID = ((SMWMeanDataElement) clickedOnObject).getMappingID();
					
					GlobalMapping globalMappingReference = trial.getGlobalMapping();
					GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
					
					tmpGME.setColorFlag(false);
					trial.getSystemEvents().updateRegisteredObjects("colorEvent");
				}
			}
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "MDWP04");
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
			
			for(Enumeration e1 = tmpMeanDataElementList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
									
				if(yCoord <= (tmpSMWMeanDataElement.getYEnd()))
				{
					if((yCoord >= (tmpSMWMeanDataElement.getYBeg())) && (xCoord >= (tmpSMWMeanDataElement.getXBeg()))
																		  && (xCoord <= (tmpSMWMeanDataElement.getXEnd())))
					{
						if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0)
						{
							//Set the clickedSMWMeanDataElement.
							clickedOnObject = tmpSMWMeanDataElement;
							popup.show(this, evt.getX(), evt.getY());
							
							//Return from this function.
							return;
						}
						else
						{
							//Want to set the clicked on mapping to the current highlight color or, if the one
							//clicked on is already the current highlighted one, set it back to normal.
							if((trial.getColorChooser().getHighlightColorMappingID()) == -1)
							{
								trial.getColorChooser().setHighlightColorMappingID(tmpSMWMeanDataElement.getMappingID());
							}
							else
							{
								if(!((trial.getColorChooser().getHighlightColorMappingID()) == (tmpSMWMeanDataElement.getMappingID())))
									trial.getColorChooser().setHighlightColorMappingID(tmpSMWMeanDataElement.getMappingID());
								else
									trial.getColorChooser().setHighlightColorMappingID(-1);
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
			jRacy.systemError(e, null, "MDWP05");
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
			jRacy.systemError(e, null, "MDWP06");
		}
	}
	
	//Instance stuff.
	Vector staticNodeList;
	
	int newXPanelSize = 0;
	int newYPanelSize = 0;
	
	
	String counterName = null;
	
	int barHeight;
	int barSpacing;
	int defaultBarLength = 250;
	int maxXLength = 0;
	int numberOfColors = 0;
	
	private Trial trial = null;
	
	MeanDataWindow mDWindow;
 	
 	StaticMainWindowData sMWData;
 	Vector tmpMeanDataElementList;
 	SMWMeanDataElement tmpSMWMeanDataElement;
 	
 	private double maxInclusiveValue = 0;
	private double maxExclusiveValue = 0;
	private double maxInclusivePercentValue = 0;
	private double maxExclusivePercentValue = 0;
	private double maxNumberOfCalls = 0;
	private double maxNumberOfSubroutines = 0;
	private double maxUserSecPerCall = 0;
 	
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

//Now compute the length of the bar for this object.
							//The default length for the bar shall be 200.
							
							/*  Some notes on the drawing.
								
								Drawing of the bars starts from position 200, and goes to the left.
								The percent value is then draw to the left of it.  The mapping name
								is then drawn to the right of the bar.
								
							*/