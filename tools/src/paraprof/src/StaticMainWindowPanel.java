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

	    barLength = baseBarLength;

	    //######
	    //Add items to the first popup menu.
	    //######
	    JMenuItem jMenuItem = new JMenuItem("Show Mean Total Statistics Windows");
	    jMenuItem.addActionListener(this);
	    popup1.add(jMenuItem);
      
	    jMenuItem = new JMenuItem("Show Mean Call Path Thread Relations");
	    jMenuItem.addActionListener(this);
	    popup1.add(jMenuItem);
	    //######
	    //End - Add items to the first popup menu.
	    //######

	    //######
	    //Add items to the seccond popup menu.
	    //######
	    jMenuItem = new JMenuItem("Show Total Statistics Windows");
	    jMenuItem.addActionListener(this);
	    popup2.add(jMenuItem);
      
	    jMenuItem = new JMenuItem("Show Total User Event Statistics Windows");
	    jMenuItem.addActionListener(this);
	    popup2.add(jMenuItem);

	    jMenuItem = new JMenuItem("Show Call Path Thread Relations");
	    jMenuItem.addActionListener(this);
	    popup2.add(jMenuItem);
	    //######
	    //End - Add items to the second popup menu.
	    //######

	    //######
	    //Add items to the third popup menu.
	    //######
	    //Add items to the first popup menu.
	    JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
	    mappingDetailsItem.addActionListener(this);
	    popup3.add(mappingDetailsItem);

	    jMenuItem = new JMenuItem("Change Function Color");
	    jMenuItem.addActionListener(this);
	    popup3.add(jMenuItem);
      
	    jMenuItem = new JMenuItem("Reset to Generic Color");
	    jMenuItem.addActionListener(this);
	    popup3.add(jMenuItem);
	    //######
	    //End - Add items to the third popup menu.
	    //######

	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP02");
	}
    }
  
    public String getToolTipText(MouseEvent evt){
	String S = null;
	
	try{
	    //Get the location of the mouse.
	    int xCoord = evt.getX();
	    int yCoord = evt.getY();

	    SMWThread sMWThread = null;
	    
	    //Calculate which SMWThreadDataElement was clicked on.
	    int index = (yCoord)/(trial.getPreferences().getBarSpacing())-1;

	    if(index==-1){
		if(xCoord<barXCoord){
		    if(ParaProf.helpWindow.isShowing()){
			//Clear the window first.
			ParaProf.helpWindow.clearText();
			
			//Now send the help info.
			ParaProf.helpWindow.writeText("You are to the left of the mean bar.");
			ParaProf.helpWindow.writeText("");
			ParaProf.helpWindow.writeText("Using either the right or left mouse buttons, click once" +
						      " to display more options about the" +
						      " mean values for the functions in the system.");
		    }
		    //Return a string indicating that clicking before the display bar
		    //will cause thread data to be displayed.
		    return new String("Left or right click for more options");
		}
		else{
		    ParaProfIterator l = new ParaProfIterator(list[1]);
		    while(l.hasNext()){
			SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) l.next();
			if(xCoord < sMWThreadDataElement.getXEnd()){
			   if(ParaProf.helpWindow.isShowing()){
				//Clear the window first.
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
		    if(xCoord <= (barXCoord + barLength)){
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
	    else if(index<list[0].size()){
		sMWThread = (SMWThread) list[0].elementAt(index);
		if(xCoord<barXCoord){
		    if(ParaProf.helpWindow.isShowing()){
			//Clear the window fisrt.
			ParaProf.helpWindow.clearText();
			
			//Now send the help info.
			ParaProf.helpWindow.writeText("n,c,t stands for: Node, Context and Thread.");
			ParaProf.helpWindow.writeText("");
			ParaProf.helpWindow.writeText("Using either the right or left mouse buttons, click once" +
						      " to display more options for this" +
						      " thread.");
		    }
		    
		    //Return a string indicating that clicking before the display bar
		    //will cause thread data to be displayed.
		    return new String("Left or right click for more options");
		}
		else{
		    ParaProfIterator l = (ParaProfIterator) sMWThread.getFunctionListIterator();
		    while(l.hasNext()){
			SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) l.next();
			if(xCoord < sMWThreadDataElement.getXEnd()){
			    if(ParaProf.helpWindow.isShowing()){
				ParaProf.helpWindow.clearText();
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
		    }
		    //If in here, and at this position, it means that the mouse is not over
		    //a bar. However, we might be over the misc. mapping section.  Check for this.
		    if(xCoord <= (barXCoord + barLength)){
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
	    if(this.debug()){
		System.out.println("####################################");
		System.out.println("StaticMainWindowPanel.renderIt(...)");
		System.out.println("####################################");
	    }
	    
	    list = sMWindow.getData();

	    //######
	    //Some declarations.
	    //######
	    int yCoord = 0;
	    SMWThread sMWThread = null;
	    //######
	    //Some declarations.
	    //######
	    
	    //To make sure the bar details are set, this
	    //method must be called.
	    trial.getPreferences().setBarDetails(g2D);

	    //Now safe to grab spacing and bar heights.
	    barSpacing = trial.getPreferences().getBarSpacing();
	    barHeight = trial.getPreferences().getBarHeight();
		
	    //Create font.
	    Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(), barHeight);
	    g2D.setFont(font);
	    FontMetrics fmFont = g2D.getFontMetrics(font);

	    //######
	    //Calculating the starting positions of drawing.
	    //######
	    int[] maxNCT = trial.getMaxNCTNumbers();
	    barXCoord = (fmFont.stringWidth("n,c,t "+maxNCT[0]+","+maxNCT[1]+","+maxNCT[2])) + 15;
	    //######
	    //End - Calculating the starting positions of drawing.
	    //######
	    
	    //At this point we can determine the size this panel will
	    //require. If we need to resize, don't do any more drawing,
	    //just call revalidate.
	    if(resizePanel(fmFont, barXCoord) && instruction==0){
		this.revalidate();
		return;
	    }

	    int yBeg = 0;
	    int yEnd = 0;
	    int startElement = 0;
	    int endElement = 0;
	    Rectangle clipRect = null;
	    Rectangle viewRect = null;
	    
	    if(instruction==0||instruction==1){
		if(instruction==0){
		    clipRect = g2D.getClipBounds();
		    yBeg = (int) clipRect.getY();
		    yEnd = (int) (yBeg + clipRect.getHeight());
		    /*
		      System.out.println("Clipping Rectangle: xBeg,xEnd: "+clipRect.getX()+","+((clipRect.getX())+(clipRect.getWidth()))+
		      " yBeg,yEnd: "+clipRect.getY()+","+((clipRect.getY())+(clipRect.getHeight())));
		    */ 
		}
		else{
		    viewRect = sMWindow.getViewRect();
		    yBeg = (int) viewRect.getY();
		    yEnd = (int) (yBeg + viewRect.getHeight());
		    /*
		      System.out.println("Viewing Rectangle: xBeg,xEnd: "+viewRect.getX()+","+((viewRect.getX())+(viewRect.getWidth()))+
					   " yBeg,yEnd: "+viewRect.getY()+","+((viewRect.getY())+(viewRect.getHeight())));
		    */
		}
		//Because tooltip redraw can louse things up.  Add an extra one to draw.
		startElement = ((yBeg - yCoord) / barSpacing) - 2;
		endElement  = ((yEnd - yCoord) / barSpacing) + 2;
		
		if(startElement < 0)
		    startElement = 0;
		
		if(endElement < 0)
		    endElement = 0;
		
		if(startElement > (list[0].size() - 1))
		    startElement = (list[0].size() - 1);
		
		if(endElement > (list[0].size() - 1))
		    endElement = (list[0].size() - 1);
		
		if(instruction==0)
		    yCoord = yCoord + (startElement * barSpacing);
	    }
	    else if(instruction==2 || instruction==3){
		startElement = 0;
		endElement = ((list[0].size()) - 1);
	    }
	 
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
	    yCoord = yCoord + (barSpacing); //Still need to update the yCoord even if the mean bar is not drawn.
	    if(startElement==0)
		drawBar(g2D, fmFont, "mean", null, barXCoord, yCoord, barHeight, instruction, true);
	    
	    //######
	    //Draw thread information for this mapping.
	    //######
	    for(int i = startElement; i <= endElement; i++){
		sMWThread = (SMWThread) list[0].elementAt(i);
		yCoord = yCoord + (barSpacing);
		drawBar(g2D, fmFont,
			"n,c,t " + (sMWThread.getNodeID()) +
			"," + (sMWThread.getContextID()) +
			"," + (sMWThread.getThreadID()), sMWThread,
			barXCoord, yCoord, barHeight, instruction, false);
	    } 
	    //######
	    //End - Draw thread information for this mapping.
	    //######
	    if(this.debug()){
		System.out.println("####################################");
		System.out.println("End - StaticMainWindowPanel.renderIt(...)");
		System.out.println("####################################");
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP");
	}
    }
    
    private void drawBar(Graphics2D g2D, FontMetrics fmFont, String text, SMWThread sMWThread,
			 int barXCoord, int yCoord, int barHeight, int instruction, boolean mean){
	ParaProfIterator l = null;
	int selectedGroupID = trial.getColorChooser().getGroupHighlightColorID();
	boolean highlighted = false;

	int barXEnd = barXCoord+barLength;

	g2D.setColor(Color.black);
	g2D.drawString(text, (barXCoord - (fmFont.stringWidth(text)) - 5), yCoord);

	//Calculate the sum for this thread.
	double sum = 0.00;
	if(mean)
	    l = new ParaProfIterator(list[1]);
	else
	    l = (ParaProfIterator) sMWThread.getFunctionListIterator();
	while(l.hasNext()){
	    SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) l.next();
	    if(mean)
		sum += sMWThreadDataElement.getMeanExclusiveValue();
	    else
		sum += sMWThreadDataElement.getExclusiveValue();
	}

	l.reset();
	while(l.hasNext()){
	    SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) l.next();
	    double value =0.00;
	    if(mean)
		value = sMWThreadDataElement.getMeanExclusiveValue();
	    else
		value = sMWThreadDataElement.getExclusiveValue();
	    if(value > 0.0){
		//Now compute the length of the bar for this object.
		int xLength = (int) ((value/sum)*barLength);
		if(xLength > 2){   //Only draw if there is something to draw.
		    if(barHeight > 2){
			g2D.setColor(sMWThreadDataElement.getColor());
			g2D.fillRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
			
			if((sMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorID())){ 
			    highlighted = true;
			    g2D.setColor(trial.getColorChooser().getHighlightColor());
			    g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
			    g2D.drawRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
			}
			else if((sMWThreadDataElement.isGroupMember(trial.getColorChooser().getHighlightColorID()))){
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
				highlighted = false;
			    }
			    else
				g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
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
			else if((sMWThreadDataElement.isGroupMember(trial.getColorChooser().getGroupHighlightColorID())))
			    g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
			else
			    g2D.setColor(sMWThreadDataElement.getColor());
						
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
	}
	//We have reached the end of the cycle for this thread.  However, we might be less
	//than the max length of the bar.  Therefore, fill in the rest of the bar with the
	//misc. mapping colour.
	if(barXCoord < barXEnd){
	    g2D.setColor(trial.getColorChooser().getMiscMappingsColor());
	    g2D.fillRect(barXCoord, (yCoord - barHeight), (barXEnd - barXCoord), barHeight);
	    g2D.setColor(Color.black);
	    g2D.drawRect(barXCoord, (yCoord - barHeight), (barXEnd - barXCoord), barHeight);
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
		if(arg.equals("Show Mean Total Statistics Windows")){
		    StatWindow statWindow = new StatWindow(trial, -1, -1, -1, sMWindow.getSMWData(), 0, this.debug());
		    trial.getSystemEvents().addObserver(statWindow);
		    statWindow.show();
		}
		else if(arg.equals("Show Mean Total User Event Statistics Windows")){
		    if(clickedOnObject instanceof SMWThread){
			SMWThread sMWThread = (SMWThread) clickedOnObject;
			StatWindow statWindow = new StatWindow(trial, sMWThread.getNodeID(),
							       sMWThread.getContextID(),
							       sMWThread.getThreadID(),sMWindow.getSMWData(),
							       2, this.debug());
			trial.getSystemEvents().addObserver(statWindow);
			statWindow.show();
		    }
		}
		else if(arg.equals("Show Mean Call Path Thread Relations")){
		    if(clickedOnObject instanceof SMWThread){
			SMWThread sMWThread = (SMWThread) clickedOnObject;
			CallPathUtilFuncs.trimCallPathData(trial.getGlobalMapping(),trial.getNCT().getThread(sMWThread.getNodeID(),
													     sMWThread.getContextID(),
													     sMWThread.getThreadID()));
			CallPathTextWindow callPathTextWindow = new CallPathTextWindow(trial, sMWThread.getNodeID(),
										       sMWThread.getContextID(),
										       sMWThread.getThreadID(),sMWindow.getSMWData(),
										       false, this.debug());
			trial.getSystemEvents().addObserver(callPathTextWindow);
			callPathTextWindow.show();
		    }
		}
		else if(arg.equals("Show Total Statistics Windows")){
		    if(clickedOnObject instanceof SMWThread){
			SMWThread sMWThread = (SMWThread) clickedOnObject;
			StatWindow statWindow = new StatWindow(trial, sMWThread.getNodeID(),
							       sMWThread.getContextID(),
							       sMWThread.getThreadID(),sMWindow.getSMWData(),
							       1, this.debug());
			trial.getSystemEvents().addObserver(statWindow);
			statWindow.show();
		    }
		}
		else if(arg.equals("Show Total User Event Statistics Windows")){
		    if(clickedOnObject instanceof SMWThread){
			SMWThread sMWThread = (SMWThread) clickedOnObject;
			StatWindow statWindow = new StatWindow(trial, sMWThread.getNodeID(),
							       sMWThread.getContextID(),
							       sMWThread.getThreadID(),sMWindow.getSMWData(),
							       2, this.debug());
			trial.getSystemEvents().addObserver(statWindow);
			statWindow.show();
		    }
		}
		else if(arg.equals("Show Call Path Thread Relations")){
		    if(clickedOnObject instanceof SMWThread){
			SMWThread sMWThread = (SMWThread) clickedOnObject;
			CallPathUtilFuncs.trimCallPathData(trial.getGlobalMapping(),trial.getNCT().getThread(sMWThread.getNodeID(),
													     sMWThread.getContextID(),
													     sMWThread.getThreadID()));
			CallPathTextWindow callPathTextWindow = new CallPathTextWindow(trial, sMWThread.getNodeID(),
										       sMWThread.getContextID(),
										       sMWThread.getThreadID(),sMWindow.getSMWData(),
										       false, this.debug());
			trial.getSystemEvents().addObserver(callPathTextWindow);
			callPathTextWindow.show();
		    }
		}
		else if(arg.equals("Show Function Details")){
		    if(clickedOnObject instanceof SMWThreadDataElement){
			SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) clickedOnObject;
			//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
			trial.getColorChooser().setHighlightColorID(sMWThreadDataElement.getMappingID());
			MappingDataWindow mappingDataWindow = new MappingDataWindow(trial, sMWThreadDataElement.getMappingID(), sMWindow.getSMWData(), this.debug());
			trial.getSystemEvents().addObserver(mappingDataWindow);
			mappingDataWindow.show();
		    }
		}
		else if(arg.equals("Change Function Color")){
		    //Get the clicked on object.
		    if(clickedOnObject instanceof SMWThreadDataElement){
			int mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
			GlobalMapping globalMapping = trial.getGlobalMapping();
			GlobalMappingElement globalMappingElement = (GlobalMappingElement) globalMapping.getGlobalMappingElement(mappingID, 0);
			Color color = globalMappingElement.getColor();
			color = (new JColorChooser()).showDialog(this, "Please select a new color", color);
			if(color != null){
			    globalMappingElement.setSpecificColor(color);
			    globalMappingElement.setColorFlag(true);
			    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
			}
		    }
		}
		else if(arg.equals("Reset to Generic Color")){ 
		   //Get the clicked on object.
		    if(clickedOnObject instanceof SMWThreadDataElement){
			int mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
			GlobalMapping globalMapping = trial.getGlobalMapping();
			GlobalMappingElement globalMappingElement = (GlobalMappingElement) globalMapping.getGlobalMappingElement(mappingID, 0);
			globalMappingElement.setColorFlag(false);
			trial.getSystemEvents().updateRegisteredObjects("colorEvent");
		    }
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP04");
	}
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
	    int xCoord = evt.getX();
	    int yCoord = evt.getY();
	    
	    //Get the number of times clicked.
	    int clickCount = evt.getClickCount();
	    
	    SMWThread sMWThread = null;
	    
	    //Calculate which SMWThreadDataElement was clicked on.
	    int index = (yCoord)/(trial.getPreferences().getBarSpacing())-1;
	    
	    if(index<list[0].size()){
		if(index!=-1)
		    sMWThread = (SMWThread) list[0].elementAt(index);
		if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
		    if(xCoord<barXCoord){
			clickedOnObject = sMWThread;
			if(index==-1)
			    popup1.show(this, evt.getX(), evt.getY());
			else
			    popup2.show(this, evt.getX(), evt.getY());
		    }
		    else{
			if(index==-1){
			    //Find the appropriate SMWThreadDataElement.
			    ParaProfIterator l = new ParaProfIterator(list[1]);
			    while(l.hasNext()){
				SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) l.next();
				if(xCoord < sMWThreadDataElement.getXEnd()){
				    //Set the clickedSMWDataElement.
				    clickedOnObject = sMWThreadDataElement;
				    popup3.show(this, evt.getX(), evt.getY());
				    return;
				}
			    }
			}
			else{
			    //Find the appropriate SMWThreadDataElement.
			    ParaProfIterator l = (ParaProfIterator) sMWThread.getFunctionListIterator();
			    while(l.hasNext()){
				SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) l.next();
				if(xCoord < sMWThreadDataElement.getXEnd()){
				    //Set the clickedSMWDataElement.
				    clickedOnObject = sMWThreadDataElement;
				    popup3.show(this, evt.getX(), evt.getY());
				    return;
				}
			    }
			}
		    }
		}
		else{
		    if(xCoord<barXCoord){
			if(index==-1){
			    ThreadDataWindow threadDataWindow = new ThreadDataWindow(trial, -1, -1, -1, sMWindow.getSMWData(), 0, this.debug());
			    trial.getSystemEvents().addObserver(threadDataWindow);
			    threadDataWindow.show();
			}
			else{
			    ThreadDataWindow  threadDataWindow = new ThreadDataWindow(trial, sMWThread.getNodeID(),
										      sMWThread.getContextID(),
										      sMWThread.getThreadID(),
										      sMWindow.getSMWData(),
										      1, this.debug());
			    trial.getSystemEvents().addObserver(threadDataWindow);
			    threadDataWindow.show();
			}
		    }
		    else{
			if(index==-1){
			    //Find the appropriate SMWThreadDataElement.
			    ParaProfIterator l = new ParaProfIterator(list[1]);
			    while(l.hasNext()){
				SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) l.next();
				if(xCoord < sMWThreadDataElement.getXEnd()){
				    trial.getColorChooser().setHighlightColorID(sMWThreadDataElement.getMappingID());
				    //Now display the MappingDataWindow for this mapping.
				    MappingDataWindow mappingDataWindow = new MappingDataWindow(trial, sMWThreadDataElement.getMappingID(),
												(sMWindow.getSMWData()), this.debug());
				    trial.getSystemEvents().addObserver(mappingDataWindow);
				    mappingDataWindow.show();
				    return;
				}
			    }
			}
			else{
			    //Find the appropriate SMWThreadDataElement.
			    ParaProfIterator l = (ParaProfIterator) sMWThread.getFunctionListIterator();
			    while(l.hasNext()){
				SMWThreadDataElement sMWThreadDataElement = (SMWThreadDataElement) l.next();
				if(xCoord < sMWThreadDataElement.getXEnd()){
				    trial.getColorChooser().setHighlightColorID(sMWThreadDataElement.getMappingID());
				    //Now display the MappingDataWindow for this mapping.
				    MappingDataWindow mappingDataWindow = new MappingDataWindow(trial, sMWThreadDataElement.getMappingID(),
												(sMWindow.getSMWData()), this.debug());
				    trial.getSystemEvents().addObserver(mappingDataWindow);
				    mappingDataWindow.show();
				    return;
				}
			    }
			}
		    }
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP05");
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
	computeBarLength();
	this.repaint();
    }
  
    public void computeBarLength(){
	try{
	    double sliderValue = (double) sMWindow.getSliderValue();
	    double sliderMultiple = sMWindow.getSliderMultiple();
	    barLength = baseBarLength*((int)(sliderValue*sliderMultiple));
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP06");
	}
    }

    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int barXCoord){
	boolean resized = false;
	try{
	    int newYPanelSize = (((sMWindow.getData())[0].size())+2)*barSpacing+10;
	    int newXPanelSize = barXCoord+barLength+5;
	    if((newYPanelSize!=yPanelSize)||(newXPanelSize!=xPanelSize)){
		yPanelSize = newYPanelSize;
		xPanelSize = newXPanelSize;
		this.setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
		resized = false;
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP07");
	}
	return resized;
    }

    public Dimension getPreferredSize(){
	return new Dimension(xPanelSize, yPanelSize);
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

    //######
    //Drawing information.
    //######
    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 500;
    private int barLength = 0;
    private int textOffset = 60;
    private int barXCoord = 100;
    private int lastHeaderEndPosition = 0;
    //######
    //End - Drawing information.
    //######
    
    //######
    //Panel information.
    //######
    int xPanelSize = 0;
    int yPanelSize = 0;
    //######
    //End - Panel information.
    //######
  
    //######
    //Popup menu stuff.
    //######
    private JPopupMenu popup1 = new JPopupMenu();
    private JPopupMenu popup2 = new JPopupMenu();
    private JPopupMenu popup3 = new JPopupMenu();
    private JMenuItem tUESWItem = null;
    private JMenuItem threadCallpathItem = null;
    private Object clickedOnObject = null;
    //######
    //End - Popup menu stuff.
    //######

    private boolean debug = false; //Off by default.
    //####################################
    //Instance data.
    //####################################
}
