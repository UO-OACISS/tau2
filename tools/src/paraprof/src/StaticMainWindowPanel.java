/* 
  
StaticMainWindowPanel.java
  
Title:      ParaProf
Author:     Robert Bell
Description:  
Things to do:
1)Add printing support.
2)Add image support.
3)Try to bring the paintComponent/renderIt function in line with other schemes in ParaProf.
4)Fix panel sizing to match the way other windows organize the computation.
5)Don't let this window do total calculations. Want to off load that job to a more central
location.
6)Investigate the code to find better a way of registering clicks and tooltips - looks a
bit messy at the moment.
7)Linked to the last point, do a bit of a code review.
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.print.*;
import java.awt.geom.*;

public class StaticMainWindowPanel extends JPanel implements ActionListener, MouseListener, PopupMenuListener, Printable, ParaProfImageInterface{
  
    public StaticMainWindowPanel(){
	try{
	    //Set the default tool tip for this panel.
	    this.setToolTipText("Incorrect Constructor!!!");
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP01");
	}
    }
  
    public StaticMainWindowPanel(ParaProfTrial trial, StaticMainWindow sMWindow, boolean debug){
	try{
	    //Set the default tool tip for this panel.
	    this.setToolTipText("ParaProf bar graph draw window!");
	    setBackground(Color.white);
      
	    //Add this object as a mouse listener.
	    addMouseListener(this);
      
	    //Set instance variables.
	    this.trial = trial;
	    this.sMWindow = sMWindow;
	    this.debug = debug;
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

	    threadCallpathItem = new JMenuItem("Show Call Path Thread Relations");
	    threadCallpathItem.addActionListener(this);
	    popup2.add(threadCallpathItem);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP02");
	}
    }
  
    public String getToolTipText(MouseEvent evt){
	String S = null;
	try{
	    int tmpYBegin = 0;
	    int tmpYEnd = 0;
	    //Get the location of the mouse.
	    xCoord = evt.getX();
	    yCoord = evt.getY();
      
	    //Check to see if the click occured in the mean values bar.
	    //Grab the first member.
	    if(!(list[1].isEmpty())){
		sMWThreadDataElement = (SMWThreadDataElement) list[1].elementAt(0);
		if((yCoord >= sMWThreadDataElement.getYBeg()) && (yCoord <= sMWThreadDataElement.getYEnd())){
		    //We are inside the mean values bar.  So, cycle through the elements.
		    for(Enumeration gM1 = list[1].elements(); gM1.hasMoreElements() ;){
			sMWThreadDataElement = (SMWThreadDataElement) gM1.nextElement();
			//Now we are going accross in the X direction.
			if(xCoord < barXStart){
			    //Output data to the help window if it is showing.
			    if(ParaProf.helpWindow.isShowing()){
				//Clear the window fisrt.
				ParaProf.helpWindow.clearText();
                  
				//Now send the help info.
				ParaProf.helpWindow.writeText("You are to the left of the mean bar.");
				ParaProf.helpWindow.writeText("");
				ParaProf.helpWindow.writeText("Using either the right or left mouse buttons, click once" +
							      " to display more detailed data about the" +
							      " mean values for the functions in the system.");
			    }
			    //Return a string indicating that clicking before the display bar
			    //will cause thread data to be displayed.
			    return new String("Left click - detailed display/Right click - Total Statistics.");
			}
			else if(xCoord < sMWThreadDataElement.getXEnd()){
			    //Output data to the help window if it is showing.
			    if(ParaProf.helpWindow.isShowing()){
				//Clear the window fisrt.
				ParaProf.helpWindow.clearText();
                  
				//Now send the help info.
				ParaProf.helpWindow.writeText("Your mouse is over the mean draw bar!");
				ParaProf.helpWindow.writeText("");
				ParaProf.helpWindow.writeText("Current function name is: " + sMWThreadDataElement.getMappingName());
				ParaProf.helpWindow.writeText("");
				ParaProf.helpWindow.writeText("The mean draw bars give a visual representation of the" +
							      " mean values for the functions which have run in the system." +
							      "  The funtions are assigned a color from the current" +
							      " ParaProf color set.  The colors are cycled through when the" +
							      " number of funtions exceeds the number of available" +
							      " colors. In the preferences section, you can add more colors." +
							      "  Use the right and left mouse buttons " +
							      "to give additional information.");
			    }
                
			    //Return the name of the mapping in the current thread data object.
			    return sMWThreadDataElement.getMappingName();
			}     
		    }
            
		    //If in here, and at this position, it means that the mouse is not over
		    //a bar. However, we might be over the misc. mapping section.  Check for this.
		    if(xCoord <= (barXStart + defaultBarLength)){
			//Output data to the help window if it is showing.
			if(ParaProf.helpWindow.isShowing()){
			    //Clear the window fisrt.
			    ParaProf.helpWindow.clearText();
                
			    //Now send the help info.
			    ParaProf.helpWindow.writeText("Your mouse is over the misc. function section!");
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("These are functions which have a non zero value," +
							  " but whose screen representation is less than a pixel.");
			    ParaProf.helpWindow.writeText("");
			    ParaProf.helpWindow.writeText("To view these functions, right or left click to the left of" +
							  " this bar to bring up windows which will show more detailed information.");
			}
              
			//Return the name of the mapping in the current thread data object.
			return "Misc function section ... see help window for details";
		    }
		}
	    }
        
	    for(Enumeration e1 = list[0].elements(); e1.hasMoreElements() ;){
		sMWServer = (SMWServer) e1.nextElement();
		if(yCoord <= (sMWServer.getYDrawCoord())){
		    //Enter the context loop for this server.
		    contextList = sMWServer.getContextList();
		    for(Enumeration e2 = contextList.elements(); e2.hasMoreElements() ;){
			sMWContext = (SMWContext) e2.nextElement();
			if(yCoord <= (sMWContext.getYDrawCoord())){
			    //Enter the thread loop for this context.
			    threadList = sMWContext.getThreadList();
			    for(Enumeration e3 = threadList.elements(); e3.hasMoreElements() ;){
				sMWThread = (SMWThread) e3.nextElement();
				if(yCoord <= (sMWThread.getYDrawCoord())){
				    //Now enter the thread loop for this thread.
				    threadDataList = sMWThread.getFunctionList();
				    sMWThreadDataElementCounter = 0;
				    for(Enumeration e4 = threadDataList.elements(); e4.hasMoreElements() ;){
					sMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
					//Get the yBeg and yEnd for this thread.
					tmpYBegin = sMWThreadDataElement.getYBeg();
					tmpYEnd = sMWThreadDataElement.getYEnd();
					
					//Now we are going accross in the X direction.
					if(xCoord < barXStart){
					    //Make sure that the mouse is not above or below the bar
					    //for this thread.  The y values from the first thread data
					    //object will indicate this.
					    if((yCoord >= tmpYBegin) && (yCoord <= tmpYEnd)){
						//Output data to the help window if it is showing.
						if(ParaProf.helpWindow.isShowing()){
						    //Clear the window fisrt.
						    ParaProf.helpWindow.clearText();
						    
						    //Now send the help info.
						    ParaProf.helpWindow.writeText("n,c,t stands for: Node, Context and Thread.");
						    ParaProf.helpWindow.writeText("");
						    ParaProf.helpWindow.writeText("Using either the right or left mouse buttons, click once" +
										  " to display more detailed data about this" +
										  " thread.");
						}
						
						//Return a string indicating that clicking before the display bar
						//will cause thread data to be displayed.
						return new String("Left click - detailed display/Right click - Total Statistics.");
					    }
					    else{ 
						//We do not want to keep cycling through if we have already
						//established that we are not going to draw.
						return S;
					    }
					}
					else if(xCoord < sMWThreadDataElement.getXEnd()){
					    if((yCoord >= tmpYBegin) && (yCoord <= tmpYEnd)){
						//Output data to the help window if it is showing.
						if(ParaProf.helpWindow.isShowing()){
						    //Clear the window fisrt.
						    ParaProf.helpWindow.clearText();
						    
						    //Now send the help info.
						    ParaProf.helpWindow.writeText("Your mouse is over one of the thread draw bars!");
						    ParaProf.helpWindow.writeText("");
						    ParaProf.helpWindow.writeText("Current function name is: " + sMWThreadDataElement.getMappingName());
						    ParaProf.helpWindow.writeText("");
						    ParaProf.helpWindow.writeText("The thread draw bars give a visual representation" +
										  " functions which have run on this thread." +
										  "  The funtions are assigned a color from the current" +
										  " Racy color set.  The colors are cycled through when the" +
										  " number of funtions exceeds the number of available" +
										  " colors." +
										  "  Use the right and left mouse buttons " +
										  "to give additional information.");
						}
						//Return the name of the mapping in the current thread data object.
						return sMWThreadDataElement.getMappingName();
					    }
					    else{
						//We do not want to keep cycling through if we have already
						//established that we are not going to draw.
						return S;
					    }
					}
					else{
					    //Update the counter.
					    sMWThreadDataElementCounter = (sMWThreadDataElementCounter + 1);
					}
				    }
				    //If in here, and at this position, it means that the mouse is not over
				    //a bar. However, we might be over the misc. mapping section.  Check for this.
				    if((yCoord >= tmpYBegin) && (yCoord <= tmpYEnd)){
					if(xCoord <= (barXStart + defaultBarLength)){
					    //Output data to the help window if it is showing.
					    if(ParaProf.helpWindow.isShowing()){
						//Clear the window fisrt.
						ParaProf.helpWindow.clearText();
						
						//Now send the help info.
						ParaProf.helpWindow.writeText("Your mouse is over the misc. function section!");
						ParaProf.helpWindow.writeText("");
						ParaProf.helpWindow.writeText("These are functions which have a non zero value," +
									      " but whose screen representation is less than a pixel.");
						ParaProf.helpWindow.writeText("");
						ParaProf.helpWindow.writeText("To view these functions, right or left click to the left of" +
									      " this bar to bring up windows which will show more detailed information.");
					    }
					    //Return the name of the mapping in the current thread data object.
					    return "Misc function section ... see help window for details";
					}
				    }
				    return S;
				}
			    }
			}
		    }
		    //At this point, we drop out of the mapping, returning the default string.
		    return S;
		}
	    }
	    //If here, means that we are not on one of the bars and so return the default string.
	    return S;
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP03");
	}
	return S;
    }

    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    renderIt((Graphics2D) g, 0, false);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP06");
	}
    }

    public int print(Graphics g, PageFormat pf, int page){
	
	if(pf.getOrientation() == PageFormat.PORTRAIT)
	    System.out.println("PORTRAIT");
	else if(pf.getOrientation() == PageFormat.LANDSCAPE)
	    System.out.println("LANDSCAPE");
	
	if(page >=3)
	    return Printable.NO_SUCH_PAGE;
	Graphics2D g2 = (Graphics2D)g;
	g2.translate(pf.getImageableX(), pf.getImageableY());
	g2.draw(new Rectangle2D.Double(0,0, pf.getImageableWidth(), pf.getImageableHeight()));
    
	renderIt(g2, 2, false);
    
	return Printable.PAGE_EXISTS;
    }
  
    public void renderIt(Graphics2D g2D, int instruction, boolean header){
	try{
	    list = sMWindow.getData();
	    
	    //Set the numberOfColors variable.
	    numberOfColors = trial.getColorChooser().getNumberOfColors();
	    
	    //Check to see if selected groups only are being displayed.
	    GlobalMapping tmpGM = trial.getGlobalMapping();

	    //######
	    //Other initializations.
	    //######
	    highlighted = false;
	    xCoord = yCoord = 0;
	    int yBeg = 0;
	    int yEnd = 0;
	    //######
	    //End - Other initializations.
	    //######
		
	    //To make sure the bar details are set, this
	    //method must be called.
	    trial.getPreferences().setBarDetails(g2D);
		
	    //Now safe to grab spacing and bar heights.
	    int barSpacing = trial.getPreferences().getBarSpacing();
	    int barHeight = trial.getPreferences().getBarHeight();
		
	    //Create font.
	    Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(), barHeight);
	    g2D.setFont(font);
	    FontMetrics fmFont = g2D.getFontMetrics(font);


	    //######
	    //Set panel size.
	    //######
	    //Now we have reached here, we can calculate the size this panel
	    //needs to be.  We might have to call a revalidate to increase
	    //its size.
	    int yHeightNeeded = ((3*barSpacing) + ((trial.getTotalNumberOfThreads())*barSpacing));
	    int xWidthNeeded = barXStart + defaultBarLength;

	    boolean sizeChange = false;   
	    //Resize the panel if needed.
	    if(xWidthNeeded > xPanelSize){
		xPanelSize = xWidthNeeded+10;
		sizeChange = true;
	    }
	    if(yHeightNeeded > yPanelSize){
		yPanelSize = yHeightNeeded+10;
		sizeChange = true;
	    }
	    //Only need to call revalidate when we are actually
	    //drawing to the display. And of course if we need a
	    //bigger panel.
	    if(sizeChange && (instruction==0))
		revalidate();
	    //######
	    //End - Set panel size. 
	    //######
	    
	    //######
	    //Calculating the starting positions of drawing.
	    //######
	    int[] maxNCT = trial.getMaxNCTNumbers();
	    int stringWidth = fmFont.stringWidth("n,c,t "+maxNCT[0]+","+maxNCT[1]+","+maxNCT[2]);
	    barXStart = stringWidth + 15;
	    int tmpXWidthCalc = barXStart + defaultBarLength;
	    int barXCoord = barXStart;
	    yCoord = yCoord + barSpacing;
	    //######
	    //End - Calculating the starting positions of drawing.
	    //######

	    //######
	    //Set clipping.
	    //######
	    //Only do clipping when this is a display call.
	    if(instruction==0){
		Rectangle clipRect = g2D.getClipBounds();
		yBeg = (int) clipRect.getY();
		yEnd = (int) (yBeg + clipRect.getHeight());
		//Because tooltip redraw can louse things up.  Add an extra one to draw.
		yEnd = yEnd + barSpacing;
	    }
	    else{
		//Set the begin and end to the panel size.
		yBeg = 0;
		yEnd = yPanelSize;
	    }
	    //######
	    //End - Set clipping.
	    //######

	    //######
	    //Draw the header if required.
	    //######
	    if(header){
		yCoord = yCoord + (barSpacing);
		String headerString = sMWindow.getHeaderString();
		//Need to split the string up into its separate lines.
		StringTokenizer st = new StringTokenizer(headerString, "'\n'");
		while(st.hasMoreTokens()){
		    g2D.drawString(st.nextToken(), 15, yCoord);
		    yCoord = yCoord + (barSpacing);
		}
		lastHeaderEndPosition = yCoord;
	    }
	    //######
	    //End - Draw the header if required.
	    //######

	    //######
	    //Drawing the mean bar.
	    //######
	    String meanString = "Mean";
	    int tmpMeanStringWidth = fmFont.stringWidth(meanString);
	    g2D.drawString(meanString, (barXStart - tmpMeanStringWidth - 5), yCoord);
		
	    //Now draw the bar of values.
		
	    //Cycle through the mean data values to get the total.
	    tmpSum = 0.0;
	    for(Enumeration gM1 = list[1].elements(); gM1.hasMoreElements() ;){
		sMWThreadDataElement = (SMWThreadDataElement) gM1.nextElement();
		tmpSum = tmpSum + (sMWThreadDataElement.getMeanExclusiveValue());
	    }
		
	    //Now that we have the total, can begin drawing.
	    colorCounter = 0;
	    barXCoord = barXStart;
	    for(Enumeration gM2 = list[1].elements(); gM2.hasMoreElements() ;){
		sMWThreadDataElement = (SMWThreadDataElement) gM2.nextElement();
		    
		tmpDataValue = sMWThreadDataElement.getMeanExclusiveValue();
		    
		if(tmpDataValue > 0.0){    
		    //Don't want to draw a bar if the value is zero.
		    //Now compute the length of the bar for this object.
		    //The default length for the bar shall be 200.
		    int xLength;
		    double tmpDouble;
		    tmpDouble = (tmpDataValue / tmpSum);
		    xLength = (int) (tmpDouble * defaultBarLength);
			
		    if(xLength > 2){
			//Only draw if there is something to draw.   
			if(barHeight > 2){
			    tmpColor = sMWThreadDataElement.getColor();
			    g2D.setColor(tmpColor);
			    g2D.fillRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
				
			    if((sMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorID())){ 
				highlighted = true;
				g2D.setColor(trial.getColorChooser().getHighlightColor());
				g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
				g2D.drawRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
			    }
			    else if((sMWThreadDataElement.isGroupMember(trial.getColorChooser().getGroupHighlightColorID()))){
				highlighted = true;
				g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
				g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
				g2D.drawRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
			    }
			    else{
				g2D.setColor(Color.black);
				if(highlighted){
				    //Manually draw in the lines for consistancy.
				    g2D.drawLine(barXCoord + 1, (yCoord - barHeight), barXCoord + 1 + xLength, (yCoord - barHeight));
				    g2D.drawLine(barXCoord + 1, yCoord, barXCoord + 1 + xLength, yCoord);
				    g2D.drawLine(barXCoord + 1 + xLength, (yCoord - barHeight), barXCoord + 1 + xLength, yCoord);
					
				    //g2D.drawRect(barXCoord + 1, (yCoord - barHeight), xLength, barHeight);
				    highlighted = false;
				}
				else{
				    g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
				}
			    }
				
			    //Set the draw coords.
			    if(instruction==0)
				sMWThreadDataElement.setDrawCoords(barXCoord, (barXCoord + xLength), (yCoord - barHeight), yCoord);
				
			    //Update barXCoord.
			    barXCoord = (barXCoord + xLength);
			}
			else{
			    //Now set the color values for drawing!
			    //Get the appropriate color.
				
			    if((sMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorID()))
				g2D.setColor(trial.getColorChooser().getHighlightColor());
			    else if((sMWThreadDataElement.isGroupMember(trial.getColorChooser().getGroupHighlightColorID()))){
				g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
			    }
			    else{
				tmpColor = sMWThreadDataElement.getColor();
				g2D.setColor(tmpColor);
			    }
			    g2D.fillRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
			    g2D.setColor(Color.black);
			    g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
				
			    //Set the draw coords.
			    if(instruction==0)
				sMWThreadDataElement.setDrawCoords(barXCoord, (barXCoord + xLength), (yCoord - barHeight), yCoord);
				
			    //Update barXCoord.
			    barXCoord = (barXCoord + xLength);
			}
		    }
			
		    //Still want to set the draw coords for this mapping, were it to be none zero.
		    //This aids in mouse click and tool tip events.
		    if(instruction==0)
			sMWThreadDataElement.setDrawCoords(barXCoord, barXCoord, (yCoord - barHeight), yCoord);
			
		}
		else{
		    //Still want to set the draw coords for this mapping, were it to be none zero.
		    //This aids in mouse click and tool tip events.
		    if(instruction==0)
			sMWThreadDataElement.setDrawCoords(barXCoord, barXCoord, (yCoord - barHeight), yCoord);
		}
		colorCounter = (colorCounter + 1) % numberOfColors;   //Want to cycle to the next color
		//whether we have drawn or not.
	    }
		
	    //We have reached the end of the cycle for this thread.  However, we might be less
	    //than the max length of the bar.  Therefore, fill in the rest of the bar with the
	    //misc. mapping colour.
	    if(barXCoord < (defaultBarLength + barXStart)){
		g2D.setColor(trial.getColorChooser().getMiscMappingsColor());
		g2D.fillRect(barXCoord, (yCoord - barHeight), ((defaultBarLength + barXStart) - barXCoord), barHeight);
		g2D.setColor(Color.black);
		g2D.drawRect(barXCoord, (yCoord - barHeight), ((defaultBarLength + barXStart) - barXCoord), barHeight);
	    }
	    //End - Drawing the mean bar.
	    //**********
		
	    //Set the drawing color to the text color ... in this case, black.
	    g2D.setColor(Color.black);
		
	    //**********
	    //Draw the thread data bars.
	    for(Enumeration e1 = list[0].elements(); e1.hasMoreElements() ;){
		//Get the next server.
		sMWServer = (SMWServer) e1.nextElement();
		//Get the context list for this server.
		contextList = sMWServer.getContextList();
		for(Enumeration e2 = contextList.elements(); e2.hasMoreElements() ;){
		    //Get the next context.
		    sMWContext = (SMWContext) e2.nextElement();
		    //Get the thread list for this context.
		    threadList = sMWContext.getThreadList();
		    for(Enumeration e3 = threadList.elements(); e3.hasMoreElements() ;){
			//Reset the highlighted boolean.
			highlighted = false;
			    
			//For consistancy in drawing, the y coord is updated at the beggining of the loop.
			yCoord = yCoord + (barSpacing);
			    
			//Get the current thread object.
			sMWThread = (SMWThread) e3.nextElement();
			    
			//Now select whether to draw this thread.
			if((yCoord >= yBeg) && (yCoord <= yEnd)){
			    //Draw the n,c,t string to the left of the bar start position.
			    String s1 = "n,c,t   " + sMWServer.getNodeID() + "," + sMWContext.getContextID() + "," + sMWThread.getThreadID();
			    int tmpStringWidth = fmFont.stringWidth(s1);
			    g2D.drawString(s1, (barXStart - tmpStringWidth - 5), yCoord);
				
			    threadDataList = sMWThread.getFunctionList();
			    //Cycle through the data values for this thread to get the total.
			    tmpSum = 0.00;
			    for(Enumeration e4 = threadDataList.elements(); e4.hasMoreElements() ;){
				sMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
				if(tmpGM.displayMapping(sMWThreadDataElement.getMappingID()))
				    tmpSum = tmpSum + (sMWThreadDataElement.getExclusiveValue());
			    }
							
			    //Now that we have the total, can begin drawing.
			    colorCounter = 0;
			    barXCoord = barXStart;
			    for(Enumeration e5 = threadDataList.elements(); e5.hasMoreElements() ;){
				//@@@@1
				sMWThreadDataElement = (SMWThreadDataElement) e5.nextElement();
				//@@@@2
				tmpDataValue = sMWThreadDataElement.getExclusiveValue();
				if(tmpDataValue > 0.0){
				    //Don't want to draw a bar if the value is zero.
				    //Now compute the length of the bar for this object.
				    //The default length for the bar shall be 200.
				    int xLength;
				    double tmpDouble;
				    tmpDouble = (tmpDataValue / tmpSum);
				    xLength = (int) (tmpDouble * defaultBarLength);
				    if(xLength > 2){
					//Only draw if there is something to draw.   
					if(barHeight > 2){
					    tmpColor = sMWThreadDataElement.getColor();
					    g2D.setColor(tmpColor);
					    g2D.fillRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
					    
					    if((sMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorID())){ 
						highlighted = true;
						g2D.setColor(trial.getColorChooser().getHighlightColor());
						g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
						g2D.drawRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
					    }
					    else if((sMWThreadDataElement.isGroupMember(trial.getColorChooser().getGroupHighlightColorID()))){
						highlighted = true;
						g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
						g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
						g2D.drawRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
					    }
					    else{
						g2D.setColor(Color.black);
						if(highlighted){
						    //Manually draw in the lines for consistancy.
						    g2D.drawLine(barXCoord + 1, (yCoord - barHeight), barXCoord + 1 + xLength, (yCoord - barHeight));
						    g2D.drawLine(barXCoord + 1, yCoord, barXCoord + 1 + xLength, yCoord);
						    g2D.drawLine(barXCoord + 1 + xLength, (yCoord - barHeight), barXCoord + 1 + xLength, yCoord);
						    
						    //g2D.drawRect(barXCoord + 1, (yCoord - barHeight), xLength, barHeight);
						    highlighted = false;
						}
						else{
						    g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
						}
					    }
					    
					    //Set the draw coords.
					    if(instruction==0)
						sMWThreadDataElement.setDrawCoords(barXCoord, (barXCoord + xLength), (yCoord - barHeight), yCoord);
					    
					    //Update barXCoord.
					    barXCoord = (barXCoord + xLength);
					}
					else{
					    //Now set the color values for drawing!
					    //Get the appropriate color.
					    if((sMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorID()))
						g2D.setColor(trial.getColorChooser().getHighlightColor());
					    else if((sMWThreadDataElement.isGroupMember(trial.getColorChooser().getGroupHighlightColorID()))){
						g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
					    }
					    else{
						tmpColor = sMWThreadDataElement.getColor();
						g2D.setColor(tmpColor);
					    }
					    
					    g2D.fillRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
					    g2D.setColor(Color.black);
					    g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
					    
					    //Set the draw coords.
					    if(instruction==0)
						sMWThreadDataElement.setDrawCoords(barXCoord, (barXCoord + xLength), (yCoord - barHeight), yCoord);
					    
					    //Update barXCoord.
					    barXCoord = (barXCoord + xLength);
					}
					
				    }
				    
				    //Still want to set the draw coords for this mapping, were it to be none zero.
				    //This aids in mouse click and tool tip events.
				    if(instruction==0)
					sMWThreadDataElement.setDrawCoords(barXCoord, barXCoord, (yCoord - barHeight), yCoord);
				    
				}
				else{
				    //Still want to set the draw coords for this mapping, were it to be none zero.
				    //This aids in mouse click and tool tip events.
				    if(instruction==0)
					sMWThreadDataElement.setDrawCoords(barXCoord, barXCoord, (yCoord - barHeight), yCoord);
				}
				colorCounter = (colorCounter + 1) % numberOfColors;   //Want to cycle to the next color
				//whether we have drawn or not.
			    }
			    
			    //We have reached the end of the cycle for this thread.  However, we might be less
			    //than the max length of the bar.  Therefore, fill in the rest of the bar with the
			    //misc. mapping colour.
			    if(barXCoord < (defaultBarLength + barXStart)){
				g2D.setColor(trial.getColorChooser().getMiscMappingsColor());
				g2D.fillRect(barXCoord, (yCoord - barHeight), ((defaultBarLength + barXStart) - barXCoord), barHeight);
				g2D.setColor(Color.black);
				g2D.drawRect(barXCoord, (yCoord - barHeight), ((defaultBarLength + barXStart) - barXCoord), barHeight);
			    }
			    
			    //Reset the drawing color to the text color ... in this case, black.
			    g2D.setColor(Color.black);
			}//End yBeg, yEnd check.
			    
			//We are about to move on to drawing the next thread.  Thus, record the
			//max y draw value for this thread.
			if(instruction==0)
			    sMWThread.setYDrawCoord(yCoord);
		    }
		    //We are about to move on to drawing the next context.  Thus, record the
		    //max y draw value for this context.
		    if(instruction==0)
			sMWContext.setYDrawCoord(yCoord);
		}
		    
		//We are about to move on to drawing the next server.  Thus, record the
		//max y draw value for this server.
		if(instruction==0)
		    sMWServer.setYDrawCoord(yCoord);
	    }
	}
	catch(Exception e){
	    e.printStackTrace();
	    UtilFncs.systemError(e, null, "SMWP07");
	}
    }

    //####################################
    //Interface code.
    //####################################
    
    //######
    //ActionListener.
    //######
    public void actionPerformed(ActionEvent evt){
	try{
	    Object EventSrc = evt.getSource();
      
	    if(EventSrc instanceof JMenuItem){
		String arg = evt.getActionCommand();
		if(arg.equals("Show Function Details")){
		    //Bring up an expanded data window for this mapping, and set this mapping as highlighted.
		    trial.getColorChooser().setHighlightColorID(clickedOnObject.getMappingID());
		    MappingDataWindow tmpRef = new MappingDataWindow(trial, clickedOnObject.getMappingID(), (sMWindow.getSMWData()), this.debug());
		    trial.getSystemEvents().addObserver(tmpRef);
		    tmpRef.show();
		}
		else if(arg.equals("Change Function Color")){ 
		    int mappingID = clickedOnObject.getMappingID();
		    
		    GlobalMapping globalMappingReference = trial.getGlobalMapping();
		    GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
		    
		    Color tmpCol = tmpGME.getColor();
		    
		    JColorChooser tmpJColorChooser = new JColorChooser();
		    tmpCol = tmpJColorChooser.showDialog(this, "Please select a new color", tmpCol);
		    if(tmpCol != null){
			tmpGME.setSpecificColor(tmpCol);
			tmpGME.setColorFlag(true);
			
			trial.getSystemEvents().updateRegisteredObjects("colorEvent");
		    }
		}
		
		else if(arg.equals("Reset to Generic Color")){
		    int mappingID = clickedOnObject.getMappingID();
		    GlobalMapping globalMappingReference = trial.getGlobalMapping();
		    GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
		    tmpGME.setColorFlag(false);
		    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
		}
		else if(arg.equals("Highlight this Function")){   
		    trial.getColorChooser().setHighlightColorID(clickedOnObject.getMappingID());
		}
		else if(arg.equals("Un-Highlight this Function")){   
		    trial.getColorChooser().setHighlightColorID(-1);}
		else if(arg.equals("Show Total Statistics Windows")){
		    StatWindow tmpRef = new StatWindow(trial, node, context,
						       thread, sMWindow.getSMWData(),
						       1, this.debug());
		    trial.getSystemEvents().addObserver(tmpRef);
		    tmpRef.show();
		}
		else if(arg.equals("Show Total User Event Statistics Windows")){
		    StatWindow tmpRef = new StatWindow(trial, node, context,
						       thread, sMWindow.getSMWData(),
						       2, this.debug());
		    trial.getSystemEvents().addObserver(tmpRef);
		    tmpRef.show();
		}
		else if(arg.equals("Show Call Path Thread Relations")){
		    CallPathUtilFuncs.trimCallPathData(trial.getGlobalMapping(),trial.getNCT().getThread(node,context,thread));
		    CallPathTextWindow tmpRef = new CallPathTextWindow(trial, node, context,
								       thread, sMWindow.getSMWData(),
								       false, this.debug());
		    trial.getSystemEvents().addObserver(tmpRef);
		    tmpRef.show();
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP04");}
    }
    //######
    //End - ActionListener.
    //######

    //######
    //MouseListener
    //######
    public void mouseClicked(MouseEvent evt){
	try{
	    //Get the location of the mouse.
	    xCoord = evt.getX();
	    yCoord = evt.getY();
      
	    //Get the number of times clicked.
	    int clickCount = evt.getClickCount();
      
	    //######
	    //Check to see if the click occured in the mean values bar.
	    //######
	    if(!(list[1].isEmpty())){
		sMWThreadDataElement = (SMWThreadDataElement) list[1].elementAt(0);
        
		if((yCoord >= sMWThreadDataElement.getYBeg()) && (yCoord <= sMWThreadDataElement.getYEnd())){
		    //We are inside the mean values bar.  So, cycle through the elements.
		    for(Enumeration gM1 = list[1].elements(); gM1.hasMoreElements() ;){
			sMWThreadDataElement = (SMWThreadDataElement) gM1.nextElement();
			//Now we are going accross in the X direction.
			if(xCoord < barXStart){
			    //Bring up the thread data window for this thread object!
			    if((evt.getModifiers() & InputEvent.BUTTON1_MASK) != 0){
				ThreadDataWindow tmpRef = new ThreadDataWindow(trial, -1, -1, -1, sMWindow.getSMWData(), 0, this.debug());
				trial.getSystemEvents().addObserver(tmpRef);
				tmpRef.show();
			    }
			    else{
				//Bring up the total stat window here!
				StatWindow tmpRef = new StatWindow(trial, -1, -1, -1, sMWindow.getSMWData(), 0, this.debug());
				trial.getSystemEvents().addObserver(tmpRef);
				tmpRef.show();
			    }
			    return;
			}
			else if(xCoord < sMWThreadDataElement.getXEnd()){
			    if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
				//Set the clickedSMWThreadDataElement.
				clickedOnObject = sMWThreadDataElement;
				popup.show(this, evt.getX(), evt.getY());
				return;
			    }
			    else{
				trial.getColorChooser().setHighlightColorID(sMWThreadDataElement.getMappingID());
				MappingDataWindow tmpRef = new MappingDataWindow(trial, sMWThreadDataElement.getMappingID(), (sMWindow.getSMWData()), this.debug());
				trial.getSystemEvents().addObserver(tmpRef);
				tmpRef.show();
			    }
			    return;
			}     
		    }
		}
	    }
	    //######
	    //End - Check to see if the click occured in the mean values bar.
	    //######
      
	    //######
	    //Check for clicking in the rest of the window.
	    //######
	    for(Enumeration e1 = list[0].elements(); e1.hasMoreElements() ;){
		sMWServer = (SMWServer) e1.nextElement();
		if(yCoord <= (sMWServer.getYDrawCoord())){
		    //Enter the context loop for this server.
		    contextList = sMWServer.getContextList();
		    for(Enumeration e2 = contextList.elements(); e2.hasMoreElements() ;){
			sMWContext = (SMWContext) e2.nextElement();
			if(yCoord <= (sMWContext.getYDrawCoord())){
			    //Enter the thread loop for this context.
			    threadList = sMWContext.getThreadList();
			    for(Enumeration e3 = threadList.elements(); e3.hasMoreElements() ;){
				sMWThread = (SMWThread) e3.nextElement();
				if(yCoord <= (sMWThread.getYDrawCoord())){
				    //Now enter the thread loop for this thread.
				    threadDataList = sMWThread.getFunctionList();
				    sMWThreadDataElementCounter = 0;
				    for(Enumeration e4 = threadDataList.elements(); e4.hasMoreElements() ;){
					sMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
					//Now we are going accross in the X direction.
					if(xCoord < barXStart){
					    //Make sure that the mouse is not above or below the bar
					    //for this thread.  The y values from the first thread data
					    //object will indicate this.
					    if((yCoord >= sMWThreadDataElement.getYBeg()) && (yCoord <= sMWThreadDataElement.getYEnd())){
						//Bring up the thread data window for this thread object!
						if((evt.getModifiers() & InputEvent.BUTTON1_MASK) != 0){
						    ThreadDataWindow tmpRef = new ThreadDataWindow(trial, sMWThreadDataElement.getNodeID(),
												   sMWThreadDataElement.getContextID(),
												   sMWThreadDataElement.getThreadID(),
												   sMWindow.getSMWData(),
												   1, this.debug());                  
						    trial.getSystemEvents().addObserver(tmpRef);
						    tmpRef.show();
						}
						else{
						    popup2.show(this, evt.getX(), evt.getY());
						    node = sMWThreadDataElement.getNodeID();
						    context = sMWThreadDataElement.getContextID();
						    thread = sMWThreadDataElement.getThreadID();
						    return;
						}
					    }
					    return;
					}
					else if(xCoord < sMWThreadDataElement.getXEnd()){
					    if((yCoord >= sMWThreadDataElement.getYBeg()) && (yCoord <= sMWThreadDataElement.getYEnd())){
						if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
						    //Set the clickedSMWDataElement.
						    clickedOnObject = sMWThreadDataElement;
						    popup.show(this, evt.getX(), evt.getY());
						    return;
						}
						else{
						    trial.getColorChooser().setHighlightColorID(sMWThreadDataElement.getMappingID());
						    MappingDataWindow tmpRef = new MappingDataWindow(trial, sMWThreadDataElement.getMappingID(),
												     (sMWindow.getSMWData()), this.debug());
						    trial.getSystemEvents().addObserver(tmpRef);
						    tmpRef.show();
						}
						return;
					    }
					}
					else{
					    //Update the counter.
					    sMWThreadDataElementCounter = (sMWThreadDataElementCounter + 1);
					}
				    }
				}
			    }
			}
		    }
		}
	    }
	    //######
	    //End - Check for clicking in the rest of the window.
	    //######
	} 
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP05");
	}
    }
    public void mousePressed(MouseEvent evt){}
    public void mouseReleased(MouseEvent evt){}
    public void mouseEntered(MouseEvent evt){}
    public void mouseExited(MouseEvent evt){}
    //######
    //End - MouseListener
    //######

    //######
    //ParaProfImageInterface
    //######
    public Dimension getImageSize(boolean fullScreen, boolean header){
	Dimension d = null;
	if(fullScreen)
	    d = this.getPreferredSize();
	else
	    d = sMWindow.getSize();
	d.setSize(d.getWidth(),d.getHeight()+lastHeaderEndPosition);
	return d;
    }
    //######
    //End - ParaProfImageInterface
    //######
    
    //######
    //PopupMenuListener code.
    //######
    public void popupMenuWillBecomeVisible(PopupMenuEvent evt){
	try{
	    if(trial.userEventsPresent()){
		tUESWItem.setEnabled(true);
	    }
	    else{
		tUESWItem.setEnabled(false);
	    }
	    
	    if(trial.callPathDataPresent()){
		threadCallpathItem.setEnabled(true);
	    }
	    else{
		threadCallpathItem.setEnabled(true);
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMW03");
	}
    }
    public void popupMenuWillBecomeInvisible(PopupMenuEvent evt){}
    public void popupMenuCanceled(PopupMenuEvent evt){}
    //######
    //PopupMenuListener code.
    //######
  
    public void changeInMultiples(){
	computeDefaultBarLength();
	this.repaint();
    }
  
    public Dimension getPreferredSize(){
	return new Dimension(xPanelSize, yPanelSize);
    }
  
    public void computeDefaultBarLength(){
	try{
	    double sliderValue = (double) sMWindow.getSliderValue();
	    double sliderMultiple = sMWindow.getSliderMultiple();
	    double result = 500*sliderValue*sliderMultiple;
	    
	    defaultBarLength = (int) result;
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP07");
	}
    }
    
    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    StaticMainWindow sMWindow = null;
    private Vector[] list = {new Vector(), new Vector()}; //list[0]:The result of a call to getSMWGeneralData in StaticMainWindowData
    //list[1]:The result of a call to getMeanData in StaticMainWindowData
  
    int xPanelSize = 600;
    int yPanelSize = 300;
  
    //######
    //Popup menu definitions.
    //######
    private JPopupMenu popup = new JPopupMenu();
    private JPopupMenu popup2 = new JPopupMenu();
  
    JMenuItem tUESWItem = null;
    JMenuItem threadCallpathItem = null;
    //######
    //######

    //######
    //Some place holder definitions - used for cycling through data lists.
    //######
    Vector contextList = null;
    Vector threadList = null;
    Vector threadDataList = null;
    SMWServer sMWServer = null;
    SMWContext sMWContext = null;
    SMWThread sMWThread = null;
    SMWThreadDataElement sMWThreadDataElement = null;
    //######
    //End - Place holder definitions.
    //######
  
    //######
    //Convenient counters.
    //######
    int node = 0;
    int context = 0;
    int thread = 0;
    int sMWThreadDataElementCounter = 0;
    int colorCounter = 0;
    //######
    //End - Convenient counters.
    //######
  
    //######
    //Other useful variables for getToolTipText, mouseEvents, and paintComponent.
    //######
    int xCoord = -1;
    int yCoord = -1;
    SMWThreadDataElement clickedOnObject = null;
    //######
    //End - Other useful variables for getToolTipText, mouseEvents, and paintComponent.
    //######
    
    //######
    //Some misc stuff for the paintComponent function.
    //######
    String counterName = null;
    private int defaultBarLength = 500;
    String tmpString = null;
    double tmpSum = -1;
    double tmpDataValue = -1;
    Color tmpColor = null;
    boolean highlighted = false;
    int barXStart = -1;
    int numberOfColors = 0;
    //######
    //End - Some misc stuff for the paintComponent function.
    //######

    private int lastHeaderEndPosition = 0;

    private boolean debug = false; //Off by default.
    //####################################
    //Instance data.
    //####################################
}
