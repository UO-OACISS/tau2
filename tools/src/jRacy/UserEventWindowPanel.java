/* 
	
	UserEventWindowPanel.java
	
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


public class UserEventWindowPanel extends JPanel implements ActionListener, MouseListener
{
	int xPanelSize = 550;
	int yPanelSize = 550;
	
	public UserEventWindowPanel()
	{
		try{
			setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
			
			//Schedule a repaint of this panel.
			this.repaint();
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "UEWP01");
		}
	
	}
	
	
	public UserEventWindowPanel(Trial inTrial, int inMappingID, UserEventWindow inUEWindow)
	{
		try{
			setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
			setBackground(Color.white);
			
			//Add this object as a mouse listener.
			addMouseListener(this);
			
			
			trial = inTrial;
			
			mappingID = inMappingID;
			
			
			//Grab the appropriate global mapping element.
			GlobalMapping tmpGM = trial.getGlobalMapping();
			GlobalMappingElement tmpGME = tmpGM.getGlobalMappingElement(mappingID, 2);
			
			mappingName = tmpGME.getMappingName();
			uEWindow = inUEWindow;
			
			//Add items to the popu menu.
			JMenuItem changeColorItem = new JMenuItem("Change User Event Color");
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
			jRacy.systemError(e, null, "UEWP02");
		}
	
	}
	

	public void paintComponent(Graphics g)
	{
		try
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
			
			//Create font.
			Font font = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), barHeight);
			g.setFont(font);
			FontMetrics fmFont = g.getFontMetrics(font);

			double tmpSum;
			double tmpDataValue;
			double tmpMaxValue; 
			Color tmpColor;
			String tmpString;
			int stringWidth;
			int stringStart;
			
			//Convenient counters.
			int colorCounter = 0;
			
			//Convenient flags.
			int displayType = -1;
		
			int yCoord = 0;
			
			int tmpXWidthCalc = 0;
			
			//An XCoord used in drawing the bars.
			int barXCoord = defaultBarLength + 60;
			yCoord = 0;
			
			//Grab the appropriate global mapping element.
			GlobalMapping tmpGM = trial.getGlobalMapping();
			GlobalMappingElement tmpGME = tmpGM.getGlobalMappingElement(mappingID, 2);
			
			//Set the max values for this mapping.
			maxUserEventNumberValue = tmpGME.getMaxUserEventNumberValue();
			maxUserEventMinValue = tmpGME.getMaxUserEventMinValue();
			maxUserEventMaxValue = tmpGME.getMaxUserEventMaxValue();
			maxUserEventMeanValue = tmpGME.getMaxUserEventMeanValue();
			
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
			
			
			//**********
			//Draw the mapping name.
			g.drawString("USER EVENT NAME: " + mappingName, 5, yCoord);
			//Calculate its width.
			tmpXWidthCalc = fmFont.stringWidth(mappingName);
			yCoord = yCoord + (barSpacing);
			//End - Draw the mapping name.
			//**********
			
			
			if((uEWindow.userEventValue()).equals("value"))
			{
				displayType = 0;
				tmpMaxValue = maxUserEventNumberValue;
				tmpString = new String(Double.toString(tmpMaxValue));
				stringWidth = fmFont.stringWidth(tmpString);
				barXCoord = barXCoord + stringWidth;
			}
			else if((uEWindow.userEventValue()).equals("min"))
			{
				displayType = 1;
				tmpMaxValue = maxUserEventMinValue;
				tmpString = new String(Double.toString(tmpMaxValue));
				stringWidth = fmFont.stringWidth(tmpString);
				barXCoord = barXCoord + stringWidth;
			}
			else if((uEWindow.userEventValue()).equals("max"))
			{
				displayType = 2;
				tmpMaxValue = maxUserEventMaxValue;
				tmpString = new String(Double.toString(tmpMaxValue));
				stringWidth = fmFont.stringWidth(tmpString);
				barXCoord = barXCoord + stringWidth;
			}
			else
			{
				displayType = 3;
				tmpMaxValue = maxUserEventMeanValue;
				tmpString = new String(Double.toString(tmpMaxValue));
				stringWidth = fmFont.stringWidth(tmpString);
				barXCoord = barXCoord + stringWidth;
			}
			
			serverNumber = 0;
				
			for(Enumeration e1 = (uEWindow.getStaticMainWindowSystemData()).elements(); e1.hasMoreElements() ;)
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
						String s1 = "n,c,t   " + serverNumber + "," + contextNumber + "," + threadNumber;
						
						tmpSMWThread = (SMWThread) e3.nextElement();
						tmpThreadDataElementList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e5 = tmpThreadDataElementList.elements(); e5.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e5.nextElement();
							
							if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
							{
								//For consistancy in drawing, the y coord is updated at the beggining of the loop.
								yCoord = yCoord + (barSpacing);
								
								if(displayType == 0)
								{
									tmpDataValue = tmpSMWThreadDataElement.getUserEventNumberValue();
								}
								else if(displayType == 1)
								{
									tmpDataValue = tmpSMWThreadDataElement.getUserEventMinValue();
								}
								else if(displayType == 2)
								{
									tmpDataValue = tmpSMWThreadDataElement.getUserEventMaxValue();
								}
								else
								{
									tmpDataValue = tmpSMWThreadDataElement.getUserEventMeanValue();
								}
								
								int xLength;
								double tmpDouble;
								
								tmpDouble = (tmpDataValue / tmpMaxValue);
								xLength = (int) (tmpDouble * defaultBarLength);
								if(xLength == 0)
									xLength = 1;
								
								//Now set the color values for drawing!
								//Get the appropriate color.
								tmpColor = tmpSMWThreadDataElement.getUserEventMappingColor();
								g.setColor(tmpColor);
										
								if((xLength > 2) && (barHeight > 2)) //Otherwise, do not use boxes ... not enough room.
								{
									g.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
									
									if(mappingID == (trial.getColorChooser().getUEHCMappingID()))
									{
										g.setColor(trial.getColorChooser().getUEHC());
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
									if(mappingID == (trial.getColorChooser().getUEHCMappingID()))
										g.setColor(trial.getColorChooser().getUEHC());
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
			jRacy.systemError(e, null, "UEWP03");
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
				if(arg.equals("Change User Event Color"))
				{	
					GlobalMapping globalMappingReference = trial.getGlobalMapping();
					GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
					
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
					GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
					
					tmpGME.setColorFlag(false);
					trial.getSystemEvents().updateRegisteredObjects("colorEvent");
				}
			}
		
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "UEWP04");
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
			jRacy.systemError(e, null, "UEWP05");
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
			double sliderValue = (double) uEWindow.getSliderValue();
			double sliderMultiple = uEWindow.getSliderMultiple();
			double result = 250*sliderValue*sliderMultiple;
			
			defaultBarLength = (int) result;
		}
		catch(Exception e)
		{
			jRacy.systemError(e, null, "UEWP06");
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
	
	private int maxUserEventNumberValue = 0;
	private double maxUserEventMinValue = 0;
	private double maxUserEventMaxValue = 0;
	private double maxUserEventMeanValue = 0;
	
	private int serverNumber = -1;
	private int	contextNumber = -1;
	private int	threadNumber = -1;
	
	private UserEventWindow uEWindow = null;
 	
 	private Trial trial = null;
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

