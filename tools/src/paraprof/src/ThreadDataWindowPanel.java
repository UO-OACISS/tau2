/* 
  
ThreadDataWindowPanel.java
  
Title:      ParaProf
Author:     Robert Bell
Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import java.awt.geom.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import java.text.*;


public class ThreadDataWindowPanel extends JPanel implements ActionListener, MouseListener, Printable{

    public ThreadDataWindowPanel(){
    
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDWP01");
	}
	
    }

    public ThreadDataWindowPanel(Trial trial,
				 int nodeID,
				 int contextID,
				 int threadID,
				 ThreadDataWindow tDWindow,
				 StaticMainWindowData sMWData){
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
	    this.sMWData = sMWData;
	    barLength = baseBarLength;
	    
	    thread = trial.getThread(nodeID,contextID,threadID);

	    //**********
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
	    //End - Add items to the popu menu.
	    //**********
	    
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDWP02");
	}
	
    }
    
    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    drawPage((Graphics2D) g, false);
	}
	catch(Exception e){
	    System.out.println(e);
	    ParaProf.systemError(e, null, "TDWP03");
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
    
	drawPage(g2, true);
    
	return Printable.PAGE_EXISTS;
    }
  
    public void drawPage(Graphics2D g2D, boolean print){
  	try{ 
	    functionList = tDWindow.getStaticMainWindowSystemData();

	    //**********
	    //Some declarations.
	    double value = 0.0;
	    double maxValue = 0.0;
	    int stringWidth = 0;
	    int yCoord = 0;
	    int barXCoord = barLength + textOffset;
	    //End - Some declarations.
	    //**********
      	    
	    //With group support present, it is possible that the number of mappings in
	    //our data list is zero.  This can occur when the user's selected groups to display are
	    //not present on this thread ... for examaple. If so, just return.
	    if((functionList.size()) == 0)
		return;
	    
	    Rectangle clipRect = g2D.getClipBounds();
	    
	    int yBeg = (int) clipRect.getY();
	    int yEnd = (int) (yBeg + clipRect.getHeight());
	    int startThreadElement = 0;
	    int endThreadElement = 0;
	    
	    if(print){
		startThreadElement = 0;
		endThreadElement = 100; //((functionList.size()) - 1);
	    }
	    else{
		
		startThreadElement = ((yBeg - yCoord) / barSpacing) - 1;
		endThreadElement  = ((yEnd - yCoord) / barSpacing) + 1;
		
		if(startThreadElement < 0)
		    startThreadElement = 0;
		
		if(endThreadElement < 0)
		    endThreadElement = 0;
		
		if(startThreadElement > (functionList.size() - 1))
		    startThreadElement = (functionList.size() - 1);
		
		if(endThreadElement > (functionList.size() - 1))
		    endThreadElement = (functionList.size() - 1);
		
		yCoord = yCoord + (startThreadElement * barSpacing);
	    }
	    
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
	    
	    //***
	    //Set max values.
	    //***
	    switch(tDWindow.getMetric()){
	    case 0: 
		if(tDWindow.isPercent())
		    maxValue = thread.getMaxInclusivePercentValue(trial.getCurValLoc());
		else
		    maxValue = thread.getMaxInclusiveValue(trial.getCurValLoc());
		break;			    
	    case 1:
		if(tDWindow.isPercent())
		    maxValue = thread.getMaxExclusivePercentValue(trial.getCurValLoc());
		else
		    maxValue = thread.getMaxExclusiveValue(trial.getCurValLoc());
		break;
	    case 2:
		maxValue = thread.getMaxNumberOfCalls();
		break;
	    case 3:
		maxValue = thread.getMaxNumberOfSubRoutines();
		break;
	    case 4:
		maxValue = thread.getMaxUserSecPerCall(trial.getCurValLoc());
		break;
	    default:
		ParaProf.systemError(null, null, "Unexpected type - MDWP value: " + tDWindow.getMetric());
	    }
	    
	    if(tDWindow.isPercent()){
		stringWidth = fmFont.stringWidth(UtilFncs.getTestString(maxValue, ParaProf.defaultNumberPrecision) + "%");
		barXCoord = barXCoord + stringWidth;
	    }
	    else{
		stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(tDWindow.units(),maxValue));
		barXCoord = barXCoord + stringWidth;
	    }
	    //***
	    //End - Set max values.
	    //***
	    

	    //At this point we can determine the size this panel will
	    //require. If we need to resize, don't do any more drawing,
	    //just call revalidate.
	    if(resizePanel(fmFont, barXCoord, functionList, startThreadElement, endThreadElement)){
		this.revalidate();
		return;
	    }

	    for(int i = startThreadElement; i <= endThreadElement; i++){   
		sMWThreadDataElement = (SMWThreadDataElement) functionList.elementAt(i);
	    
		switch(tDWindow.getMetric()){
		case 0:
		    if(tDWindow.isPercent())
			value = sMWThreadDataElement.getInclusivePercentValue();
		    else
			value = sMWThreadDataElement.getInclusiveValue();
		    break;			    
		case 1:
		    if(tDWindow.isPercent())
			value = sMWThreadDataElement.getExclusivePercentValue();
		    else
			value = sMWThreadDataElement.getExclusiveValue();
		    break;
		case 2:
		    value = sMWThreadDataElement.getNumberOfCalls();
		    break;
		case 3:
		    value = sMWThreadDataElement.getNumberOfSubRoutines();
		    break;
		case 4:
		    value = sMWThreadDataElement.getUserSecPerCall();
		    break;
		default:
		    ParaProf.systemError(null, null, "Unexpected type - MDWP value: " + tDWindow.getMetric());
		}

		yCoord = yCoord + (barSpacing);
		drawBar(g2D, fmFont, value, maxValue, barXCoord, yCoord, barHeight, sMWThreadDataElement);
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDWP04");
	}
    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double value, double maxValue, int barXCoord, 
			 int yCoord, int barHeight, SMWThreadDataElement sMWThreadDataElement){
	int xLength = 0;
	double d = 0.0;
	String s = null;
	int stringWidth = 0;
	int stringStart = 0;

	int mappingID = sMWThreadDataElement.getMappingID();
        String mappingName = sMWThreadDataElement.getMappingName();
	boolean groupMember = sMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID());

	d = (value / maxValue);
	xLength = (int) (d * barLength);
	if(xLength == 0)
	    xLength = 1;

	if((xLength > 2) && (barHeight > 2)){
	    g2D.setColor(sMWThreadDataElement.getMappingColor());
	    g2D.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);
	    
	    if(mappingID == (trial.getColorChooser().getHighlightColorMappingID())){
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
	    if(mappingID == (trial.getColorChooser().getHighlightColorMappingID()))
		g2D.setColor(trial.getColorChooser().getHighlightColor());
	    else if(groupMember)
		g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
	    else{
		g2D.setColor(sMWThreadDataElement.getMappingColor());
	    }
	    g2D.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
	}
	
	//Now print the percentage to the left of the bar.
	g2D.setColor(Color.black);
	//Need to figure out how long the percentage string will be.
	if((tDWindow.isPercent()) && ((tDWindow.getMetric())<2))					
	    s = (UtilFncs.adjustDoublePresision(value, ParaProf.defaultNumberPrecision)) + "%";
	else
	    s = UtilFncs.getOutputString(tDWindow.units(),value);
	stringWidth = fmFont.stringWidth(s);
	//Now draw the percent value to the left of the bar.
	stringStart = barXCoord - xLength - stringWidth - 5;
	g2D.drawString(s, stringStart, yCoord);

	//Now draw the mapping to the right of the bar.
	g2D.drawString(mappingName, (barXCoord + 5), yCoord);
	
	//Grab the width of the mappingName.
	stringWidth = fmFont.stringWidth(mappingName);	
	//Update the drawing coordinates.
	sMWThreadDataElement.setTDWDrawCoords(stringStart, (barXCoord+5+stringWidth), (yCoord - barHeight), yCoord);
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
	    ParaProf.systemError(e, null, "MDWP06");
	}
    }
  
    //************************************
    //Event listener code!!
    //************************************

    //******
    //ActionListener
    //******
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
			trial.getColorChooser().setHighlightColorMappingID(sMWThreadDataElement.getMappingID());
			MappingDataWindow tmpRef = new MappingDataWindow(trial, sMWThreadDataElement.getMappingID(), sMWData);
			trial.getSystemEvents().addObserver(tmpRef);
			tmpRef.show();
		    }
		}
		else if(arg.equals("Change Function Color")){ 
		    int mappingID = -1;
		    
		    //Get the clicked on object.
		    if(clickedOnObject instanceof SMWThreadDataElement)
			mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
		    
		    GlobalMapping globalMappingReference = trial.getGlobalMapping();
		    GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
		    
		    Color tmpCol = tmpGME.getMappingColor();
		    
		    JColorChooser tmpJColorChooser = new JColorChooser();
		    tmpCol = tmpJColorChooser.showDialog(this, "Please select a new color", tmpCol);
		    if(tmpCol != null){
			tmpGME.setSpecificColor(tmpCol);
			tmpGME.setColorFlag(true);
			
			trial.getSystemEvents().updateRegisteredObjects("colorEvent");
		    }
		}
		else if(arg.equals("Reset to Generic Color")){ 
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
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDWP04");
	}
    }
    //******
    //End - ActionListener
    //******
    
    //******
    //MouseListener
    //******
    public void mouseClicked(MouseEvent evt){
	try{
	    //Get the location of the mouse.
	    int xCoord = evt.getX();
	    int yCoord = evt.getY();
	    
	    //Get the number of times clicked.
	    int clickCount = evt.getClickCount();
	    for(Enumeration e1 = functionList.elements(); e1.hasMoreElements() ;){
		sMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
		
		if(yCoord <= (sMWThreadDataElement.getTDWYEnd())){
		    if((yCoord >= (sMWThreadDataElement.getTDWYBeg())) && (xCoord >= (sMWThreadDataElement.getTDWXBeg()))
		       && (xCoord <= (sMWThreadDataElement.getTDWXEnd()))){
			if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
			    //Set the clickedSMWMeanDataElement.
			    clickedOnObject = sMWThreadDataElement;
			    popup.show(this, evt.getX(), evt.getY());
			    
			    //Return from this function.
			    return;
			}
			else{
			    //Want to set the clicked on mapping to the current highlight color or, if the one
			    //clicked on is already the current highlighted one, set it back to normal.
			    if((trial.getColorChooser().getHighlightColorMappingID()) == -1){
				trial.getColorChooser().setHighlightColorMappingID(sMWThreadDataElement.getMappingID());
			    }
			    else{
				if(!((trial.getColorChooser().getHighlightColorMappingID()) == (sMWThreadDataElement.getMappingID())))
				    trial.getColorChooser().setHighlightColorMappingID(sMWThreadDataElement.getMappingID());
				else
				    trial.getColorChooser().setHighlightColorMappingID(-1);
			    }
			}
			//Nothing more to do ... return.
			return;
		    }
		    else{
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
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDWP05");
	}
    }  
    
    public void mousePressed(MouseEvent evt) {}
    public void mouseReleased(MouseEvent evt) {}
    public void mouseEntered(MouseEvent evt) {}
    public void mouseExited(MouseEvent evt) {}
    //******
    //End - MouseListener
    //******

    //************************************
    //End - Event listener code!!
    //************************************
    
    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int barXCoord, Vector functionList, int startThreadElement, int endThreadElement){
	boolean resized = false;
	try{
	    int newXPanelSize = 0;
	    int newYPanelSize = 0;
	    int width = 0;
	    int height = 0;

	    for(int i = startThreadElement; i <= endThreadElement; i++){   
		sMWThreadDataElement = (SMWThreadDataElement) functionList.elementAt(i);
		width = barXCoord+5+(fmFont.stringWidth(sMWThreadDataElement.getMappingName()));
		if(width>newXPanelSize)
		    newXPanelSize=width;
	    }

	    newYPanelSize = barSpacing + ((functionList.size() + 1) * barSpacing);

	    if((newYPanelSize!=yPanelSize)||(newXPanelSize!=xPanelSize)){
		yPanelSize = newYPanelSize;
		xPanelSize = newXPanelSize;
		this.setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
		resized = false;
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "MDWP07");
	}
	return resized;
    }

    public Dimension getPreferredSize(){
	return new Dimension(xPanelSize, (yPanelSize + 10));}
  
    //************************************
    //Instance data.
    //************************************
    private int xPanelSize = 640;
    private int yPanelSize = 480;
  
    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 250;
    private int barLength = 0;
    private int textOffset = 60;

    private int maxXLength = 0;
  
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
  
    private Trial trial = null;
    private StaticMainWindowData sMWData = null;
    private ThreadDataWindow tDWindow = null;
    private Thread thread = null;  
    private Vector functionList = null;
    private SMWThreadDataElement sMWThreadDataElement = null;
  
    private JPopupMenu popup = new JPopupMenu();

    private Object clickedOnObject = null;
  
    //************************************
    //End - Instance data.
    //************************************
}
