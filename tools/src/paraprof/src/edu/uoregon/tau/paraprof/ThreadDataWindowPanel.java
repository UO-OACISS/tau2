/* 
  
ThreadDataWindowPanel.java
  
Title:      ParaProf
Author:     Robert Bell
Description:  
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.geom.*;
import java.text.*;
import java.awt.font.*;
import edu.uoregon.tau.dms.dss.*;

public class ThreadDataWindowPanel extends JPanel implements ActionListener, MouseListener, Printable, ParaProfImageInterface{

    public ThreadDataWindowPanel(){
    
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDWP01");
	}
	
    }

    public ThreadDataWindowPanel(ParaProfTrial trial,
				 int nodeID,
				 int contextID,
				 int threadID,
				 ThreadDataWindow tDWindow,
				 int windowType,
				 boolean debug){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);
	    
	    //Add this object as a mouse listener.
	    addMouseListener(this);
	    
	    this.nodeID = nodeID;
	    this.contextID = contextID;
	    this.threadID = threadID;
	    this.trial = trial;
	    this.tDWindow = tDWindow;
	    this.windowType = windowType;
	    this.debug = debug;
	    barLength = baseBarLength;
	    
	    if(windowType==1)
		thread = trial.getNCT().getThread(nodeID,contextID,threadID);

	    //######
	    //Add items to the popu menu.
	    //######
	    JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
	    mappingDetailsItem.addActionListener(this);
	    popup.add(mappingDetailsItem);
	    
	    JMenuItem changeColorItem = new JMenuItem("Change Function Color");
	    changeColorItem.addActionListener(this);
	    popup.add(changeColorItem);
	    
	    JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
	    maskMappingItem.addActionListener(this);
	    popup.add(maskMappingItem);
	    //######
	    //End - Add items to the popu menu.
	    //######

	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDWP02");
	}
	
    }
    
    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    renderIt((Graphics2D) g, 0, false);
	}
	catch(Exception e){
	    System.out.println(e);
	    UtilFncs.systemError(e, null, "TDWP03");
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
	    list = tDWindow.getData();

	    //######
	    //Some declarations.
	    //######
	    double value = 0.0;
	    double maxValue = 0.0;
	    int stringWidth = 0;
	    int yCoord = 0;
	    int barXCoord = barLength + textOffset;
	    SMWThreadDataElement sMWThreadDataElement = null;
	    //######
	    //End - Some declarations.
	    //######
      	    
	    //With group support present, it is possible that the number of mappings in
	    //our data list is zero.  This can occur when the user's selected groups to display are
	    //not present on this thread ... for example. If so, just return.
	    if((list.size()) == 0)
		return;
	    	    
	    //To make sure the bar details are set, this
	    //method must be called.
	    trial.getPreferences().setBarDetails(g2D);	    
	    
	    //Now safe to grab spacing and bar heights.
	    barSpacing = trial.getPreferences().getBarSpacing();
	    barHeight = trial.getPreferences().getBarHeight();
	    
	    //Obtain the font and its metrics.
	    Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(), barHeight);
	    g2D.setFont(font);
	    FontMetrics fmFont = g2D.getFontMetrics(font);
	    
	    //######
	    //Set max values.
	    //######
	    if(windowType==0){
		switch(tDWindow.getValueType()){
		case 2: 
		    if(tDWindow.isPercent())
			maxValue = trial.getGlobalMapping().getMaxMeanExclusivePercentValue(trial.getSelectedMetricID());
		    else
			maxValue = trial.getGlobalMapping().getMaxMeanExclusiveValue(trial.getSelectedMetricID());
		    break;			    
		case 4:
		    if(tDWindow.isPercent())
			maxValue = trial.getGlobalMapping().getMaxMeanInclusivePercentValue(trial.getSelectedMetricID());
		    else
			maxValue = trial.getGlobalMapping().getMaxMeanInclusiveValue(trial.getSelectedMetricID());
		    break;
		case 6:
		    maxValue = trial.getGlobalMapping().getMaxMeanNumberOfCalls();
		    break;
		case 8:
		    maxValue = trial.getGlobalMapping().getMaxMeanNumberOfSubRoutines();
		    break;
		case 10:
		    maxValue = trial.getGlobalMapping().getMaxMeanUserSecPerCall(trial.getSelectedMetricID());
		    break;
		default:
		    UtilFncs.systemError(null, null, "Unexpected type - MDWP value: " + tDWindow.getValueType());
		}
	    }
	    else{
		switch(tDWindow.getValueType()){
		case 2: 
		    if(tDWindow.isPercent())
			maxValue = thread.getMaxExclusivePercentValue(trial.getSelectedMetricID());
		    else
			maxValue = thread.getMaxExclusiveValue(trial.getSelectedMetricID());
		    break;			    
		case 4:
		    if(tDWindow.isPercent())
			maxValue = thread.getMaxInclusivePercentValue(trial.getSelectedMetricID());
		    else
			maxValue = thread.getMaxInclusiveValue(trial.getSelectedMetricID());
		    break;
		case 6:
		    maxValue = thread.getMaxNumberOfCalls();
		    break;
		case 8:
		    maxValue = thread.getMaxNumberOfSubRoutines();
		    break;
		case 10:
		    maxValue = thread.getMaxUserSecPerCall(trial.getSelectedMetricID());
		    break;
		default:
		    UtilFncs.systemError(null, null, "Unexpected type - MDWP value: " + tDWindow.getValueType());
		}
	    }

	    if(tDWindow.isPercent()){
		stringWidth = fmFont.stringWidth(UtilFncs.adjustDoublePresision(maxValue, ParaProf.defaultNumberPrecision) + "%");
		barXCoord = barXCoord + stringWidth;
	    }
	    else{
		stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(tDWindow.units(),maxValue,ParaProf.defaultNumberPrecision));
		barXCoord = barXCoord + stringWidth;
	    }
	    //######
	    //End - Set max values.
	    //######

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
		    viewRect = tDWindow.getViewRect();
		    yBeg = (int) viewRect.getY();
		    yEnd = (int) (yBeg + viewRect.getHeight());
		    /*
		      System.out.println("Viewing Rectangle: xBeg,xEnd: "+viewRect.getX()+","+((viewRect.getX())+(viewRect.getWidth()))+
					   " yBeg,yEnd: "+viewRect.getY()+","+((viewRect.getY())+(viewRect.getHeight())));
		    */
		}
		startElement = ((yBeg - yCoord) / barSpacing) - 1;
		endElement  = ((yEnd - yCoord) / barSpacing) + 1;
		
		if(startElement < 0)
		    startElement = 0;
		
		if(endElement < 0)
		    endElement = 0;
		
		if(startElement > (list.size() - 1))
		    startElement = (list.size() - 1);
		
		if(endElement > (list.size() - 1))
		    endElement = (list.size() - 1);
		
		if(instruction==0)
		    yCoord = yCoord + (startElement * barSpacing);
	    }
	    else if(instruction==2 || instruction==3){
		startElement = 0;
		endElement = ((list.size()) - 1);
	    }

	    //At this point we can determine the size this panel will
	    //require. If we need to resize, don't do any more drawing,
	    //just call revalidate. Make sure we check the instruction value as we only want to
	    //revalidate if we are drawing to the screen.
	    if(resizePanel(fmFont, barXCoord, list, startElement, endElement) && instruction==0){
		this.revalidate();
		return;
	    }

	    //######
	    //Draw the header if required.
	    //######
	    if(header){
		FontRenderContext frc = g2D.getFontRenderContext();
		Insets insets = this.getInsets();
		yCoord = yCoord + (barSpacing);
		String headerString = tDWindow.getHeaderString();
		//Need to split the string up into its separate lines.
		StringTokenizer st = new StringTokenizer(headerString, "'\n'");
		while(st.hasMoreTokens()){
		    AttributedString as = new AttributedString(st.nextToken());
		    as.addAttribute(TextAttribute.FONT, font);
		    AttributedCharacterIterator aci = as.getIterator();
		    LineBreakMeasurer lbm = new LineBreakMeasurer(aci, frc);
		    float wrappingWidth = this.getSize().width - insets.left - insets.right;
		    float x = insets.left;
		    float y = insets.right;
		    while(lbm.getPosition() < aci.getEndIndex()){
			TextLayout textLayout = lbm.nextLayout(wrappingWidth);
			yCoord+= barSpacing;
			textLayout.draw(g2D, x, yCoord);
			x = insets.left;
		    }
		}
		lastHeaderEndPosition = yCoord;
	    }
	    //######
	    //End - Draw the header if required.
	    //######
	    for(int i = startElement; i <= endElement; i++){   
		sMWThreadDataElement = (SMWThreadDataElement) list.elementAt(i);
		if(windowType==0){
		    switch(tDWindow.getValueType()){
		    case 2:
			if(tDWindow.isPercent())
			    value = sMWThreadDataElement.getMeanExclusivePercentValue();
			else
			    value = sMWThreadDataElement.getMeanExclusiveValue();
			break;			    
		    case 4:
			if(tDWindow.isPercent())
			    value = sMWThreadDataElement.getMeanInclusivePercentValue();
			else
			    value = sMWThreadDataElement.getMeanInclusiveValue();
			break;
		    case 6:
			value = sMWThreadDataElement.getMeanNumberOfCalls();
			break;
		    case 8:
			value = sMWThreadDataElement.getMeanNumberOfSubRoutines();
			break;
		    case 10:
			value = sMWThreadDataElement.getMeanUserSecPerCall();
			break;
		    default:
			UtilFncs.systemError(null, null, "Unexpected type - MDWP value: " + tDWindow.getValueType());
		    }
		}
		else{
		    switch(tDWindow.getValueType()){
		    case 2:
			if(tDWindow.isPercent())
			    value = sMWThreadDataElement.getExclusivePercentValue();
			else
			    value = sMWThreadDataElement.getExclusiveValue();
			break;			    
		    case 4:
			if(tDWindow.isPercent())
			    value = sMWThreadDataElement.getInclusivePercentValue();
			else
			    value = sMWThreadDataElement.getInclusiveValue();
			break;
		    case 6:
			value = sMWThreadDataElement.getNumberOfCalls();
			break;
		    case 8:
			value = sMWThreadDataElement.getNumberOfSubRoutines();
			break;
		    case 10:
			value = sMWThreadDataElement.getUserSecPerCall();
			break;
		    default:
			UtilFncs.systemError(null, null, "Unexpected type - MDWP value: " + tDWindow.getValueType());
		    }
		}

		yCoord = yCoord + (barSpacing);
		drawBar(g2D, fmFont, value, maxValue, barXCoord, yCoord, barHeight, sMWThreadDataElement, instruction);
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDWP04");
	}
    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double value, double maxValue, int barXCoord, 
			 int yCoord, int barHeight, SMWThreadDataElement sMWThreadDataElement, int instruction){
	int xLength = 0;
	double d = 0.0;
	String s = null;
	int stringWidth = 0;
	int stringStart = 0;

	int mappingID = sMWThreadDataElement.getMappingID();
        String mappingName = sMWThreadDataElement.getMappingName();
	boolean groupMember = sMWThreadDataElement.isGroupMember(trial.getColorChooser().getGroupHighlightColorID());

	d = (value / maxValue);
	xLength = (int) (d * barLength);
	if(xLength == 0)
	    xLength = 1;

	if((xLength > 2) && (barHeight > 2)){
	    g2D.setColor(sMWThreadDataElement.getColor());
	    g2D.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
	    
	    if(mappingID == (trial.getColorChooser().getHighlightColorID())){
		g2D.setColor(trial.getColorChooser().getHighlightColor());
		g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
		g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
	    }
	    else if(groupMember){
		g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
		g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
		g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
	    }
	    else{
		g2D.setColor(Color.black);
		g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
	    }
	}
	else{
	    if(mappingID == (trial.getColorChooser().getHighlightColorID()))
		g2D.setColor(trial.getColorChooser().getHighlightColor());
	    else if(groupMember)
		g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
	    else{
		g2D.setColor(sMWThreadDataElement.getColor());
	    }
	    g2D.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
	}
	
	//Draw the value next to the bar.
	g2D.setColor(Color.black);
	//Do not want to put a percent sign after the bar if we are not exclusive or inclusive.
	if((tDWindow.isPercent()) && ((tDWindow.getValueType())<=4))					
	    s = (UtilFncs.adjustDoublePresision(value, ParaProf.defaultNumberPrecision)) + "%";
	else
	    s = UtilFncs.getOutputString(tDWindow.units(),value,ParaProf.defaultNumberPrecision);
	stringWidth = fmFont.stringWidth(s);
	//Now draw the percent value to the left of the bar.
	stringStart = barXCoord - xLength - stringWidth - 5;
	g2D.drawString(s, stringStart, yCoord);

	//Now draw the mapping to the right of the bar.
	g2D.drawString(mappingName, (barXCoord + 5), yCoord);
	
	//Grab the width of the mappingName.
	stringWidth = fmFont.stringWidth(mappingName);	
	//Update the drawing coordinates if we are drawing to the screen.
	if(instruction==0)
	    sMWThreadDataElement.setDrawCoords(stringStart, (barXCoord+5+stringWidth), (yCoord - barHeight), yCoord);
    }

    public void changeInMultiples(){
	computeBarLength();
	this.repaint();
    }
    
    public void computeBarLength(){
	try{
	    double sliderValue = (double) tDWindow.getSliderValue();
	    double sliderMultiple = tDWindow.getSliderMultiple();
	    barLength = baseBarLength*((int)(sliderValue*sliderMultiple));
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP06");
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
	    SMWThreadDataElement sMWThreadDataElement = null;
	    if(EventSrc instanceof JMenuItem){
		String arg = evt.getActionCommand();
		if(arg.equals("Show Function Details")){
		    if(clickedOnObject instanceof SMWThreadDataElement){
			sMWThreadDataElement = (SMWThreadDataElement) clickedOnObject;
			//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
			trial.getColorChooser().setHighlightColorID(sMWThreadDataElement.getMappingID());
			MappingDataWindow mappingDataWindow = new MappingDataWindow(trial, sMWThreadDataElement.getMappingID(), tDWindow.getSMWData(), this.debug());
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
	    UtilFncs.systemError(e, null, "TDWP04");
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

	    SMWThreadDataElement sMWThreadDataElement = null;

	    //Calculate which SMWThreadDataElement was clicked on.
	    int index = (yCoord)/(trial.getPreferences().getBarSpacing());

	    if(index<list.size()){
		sMWThreadDataElement = (SMWThreadDataElement) list.elementAt(index);
		if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
		    //Set the clickedSMWMeanDataElement.
		    clickedOnObject = sMWThreadDataElement;
		    popup.show(this, evt.getX(), evt.getY());
		    return;
		}
		else{
		    //Want to set the clicked on mapping to the current highlight color or, if the one
		    //clicked on is already the current highlighted one, set it back to normal.
		    if((trial.getColorChooser().getHighlightColorID()) == -1){
			trial.getColorChooser().setHighlightColorID(sMWThreadDataElement.getMappingID());
		    }
		    else{
			if(!((trial.getColorChooser().getHighlightColorID()) == (sMWThreadDataElement.getMappingID())))
			    trial.getColorChooser().setHighlightColorID(sMWThreadDataElement.getMappingID());
			else
			    trial.getColorChooser().setHighlightColorID(-1);
		    }
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TDWP05");
	}
    }  
    
    public void mousePressed(MouseEvent evt) {}
    public void mouseReleased(MouseEvent evt) {}
    public void mouseEntered(MouseEvent evt) {}
    public void mouseExited(MouseEvent evt) {}
    //######
    //End - MouseListener
    //######

    //######
    //ParaProfImageInterface
    //######
    public Dimension getImageSize(boolean fullScreen, boolean header){
	Dimension d = null;
	if(fullScreen)
	    d = this.getSize();
	else
	    d = tDWindow.getSize();
	d.setSize(d.getWidth(),d.getHeight()+lastHeaderEndPosition);
	return d;
    }
    //######
    //End - ParaProfImageInterface
    //######

    //####################################
    //End - Interface code.
    //####################################
    
    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int barXCoord, Vector list, int startElement, int endElement){
	boolean resized = false;
	try{
	    int newXPanelSize = 0;
	    int newYPanelSize = 0;
	    int width = 0;
	    int height = 0;
	    SMWThreadDataElement sMWThreadDataElement = null;
	    
	    for(int i = startElement; i <= endElement; i++){   
		sMWThreadDataElement = (SMWThreadDataElement) list.elementAt(i);
		//As a temporary fix, at 500 pixels to the end.
		width = barXCoord+5+(fmFont.stringWidth(sMWThreadDataElement.getMappingName()))+500;
		if(width>newXPanelSize)
		    newXPanelSize=width;
	    }

	    newYPanelSize = barSpacing + ((list.size() + 1) * barSpacing);

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
	return new Dimension(xPanelSize, (yPanelSize + 10));}
  
    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //Instance data.
    //####################################
    private int xPanelSize = 640;
    private int yPanelSize = 480;
  
    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 250;
    private int barLength = 0;
    private int textOffset = 60;

    private int maxXLength = 0;
  
    private ParaProfTrial trial = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private ThreadDataWindow tDWindow = null;
    private int windowType = -1;
    private edu.uoregon.tau.dms.dss.Thread thread = null;  
    private Vector list = null;
  
    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private int lastHeaderEndPosition = 0;

    private boolean debug = false; //Off by default.
    //####################################
    //Instance data.
    //####################################
}
