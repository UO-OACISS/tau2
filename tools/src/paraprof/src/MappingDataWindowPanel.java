/* 
  
MappingDataWindowPanel.java
  
Title:      ParaProf
Author:     Robert Bell
Description:
  
Things to do:
  
1) Add clipping support to this window. 
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import java.awt.geom.*;


public class MappingDataWindowPanel extends JPanel implements ActionListener, MouseListener{
    int xPanelSize = 550;
    int yPanelSize = 550;
  
    public MappingDataWindowPanel(){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "MDWP01");
	}
    
    }
  
  
    public MappingDataWindowPanel(Trial inTrial, int inMappingID, MappingDataWindow inMDWindow){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);
      
	    //Add this object as a mouse listener.
	    addMouseListener(this);
      
	    trial = inTrial;
	    mDWindow = inMDWindow;
 	    gME = ((GlobalMapping)trial.getGlobalMapping()).getGlobalMappingElement(inMappingID, 0);
 	    mappingName = gME.getMappingName();
	    mappingID = gME.getGlobalID();
      
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
	catch(Exception e){
	    ParaProf.systemError(e, null, "MDWP02");
	}
	
    }
  

    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);	    
	    Graphics2D g2D = (Graphics2D) g;

	    double value = 0.0;
	    double maxValue = 0.0;
	    int stringWidth = 0;
	    int yCoord = 0;
	    
	    Rectangle clipRect = g2D.getClipBounds();
	    
	    int yBeg = (int) clipRect.getY();
	    int yEnd = (int) (yBeg + clipRect.getHeight());
	    yEnd = yEnd + barSpacing;
	    
	    //Do the standard font and spacing stuff.
	    if(!(trial.getPreferences().areBarDetailsSet())){
		
		//Create font.
		Font font = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), 12);
		g2D.setFont(font);
		FontMetrics fmFont = g2D.getFontMetrics(font);
		
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
	    
	    //Reset the font to the correct height.
	    Font nameMetricFont = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), 12);
	    FontMetrics fmFont = g.getFontMetrics(nameMetricFont);
	    //Compute the font metrics for this font.
	    int nameMetricHeight = fmFont.getAscent() + fmFont.getMaxDescent();
	    
	    //Create font.
	    Font font = new Font(trial.getPreferences().getJRacyFont(), trial.getPreferences().getFontStyle(), barHeight);
	    g2D.setFont(font);
	    fmFont = g2D.getFontMetrics(font);
	    
	    int tmpXWidthCalc = 0;
	    //An XCoord used in drawing the bars.
	    int barXCoord = defaultBarLength + 60;
	    
	    //Set the max and mean values for this mapping.
	    switch(mDWindow.getMetric()){
	    case 0:
		if(mDWindow.isPercent()){
		    maxValue = gME.getMaxInclusivePercentValue(trial.getCurValLoc());
		    value = gME.getMeanInclusivePercentValue(trial.getCurValLoc());
		}
		else{
		    maxValue = gME.getMaxInclusiveValue(trial.getCurValLoc());
		    value = gME.getMeanInclusiveValue(trial.getCurValLoc());
		}
		break;			    
	    case 1:
		if(mDWindow.isPercent()){
		    maxValue = gME.getMaxExclusivePercentValue(trial.getCurValLoc());
		    value = gME.getMeanExclusivePercentValue(trial.getCurValLoc());
		}
		else{
		    maxValue = gME.getMaxExclusiveValue(trial.getCurValLoc());
		    value = gME.getMeanExclusiveValue(trial.getCurValLoc());
		}
		break;
	    case 2:
		maxValue = gME.getMaxNumberOfCalls();
		value = gME.getMeanNumberOfCalls();
		break;
	    case 3:
		maxValue = gME.getMaxNumberOfSubRoutines();
		value = gME.getMeanNumberOfSubRoutines();
		break;
	    case 4:
		maxValue = gME.getMaxUserSecPerCall(trial.getCurValLoc());
		value = gME.getMeanUserSecPerCall(trial.getCurValLoc());
		break;
	    default:
		ParaProf.systemError(null, null, "Unexpected type - MDWP value: " + mDWindow.getMetric());
	    }

	    if(mDWindow.isPercent()){
		stringWidth = fmFont.stringWidth(UtilFncs.getTestString(maxValue, ParaProf.defaultNumberPrecision) + "%");
		barXCoord = barXCoord + stringWidth;
	    }
	    else{
		stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(mDWindow.units(),maxValue));
		barXCoord = barXCoord + stringWidth;
	    }
	    
	    //Check for group membership.
	    groupMember = gME.isGroupMember(trial.getColorChooser().getGHCMID());

	    //******************************
	    //Do the mean bar.
	    //******************************
	    if((yCoord >= yBeg) && (yCoord <= yEnd)){
		yCoord = yCoord + (barSpacing);
		drawBar(g2D, fmFont, value, maxValue, "mean", barXCoord, yCoord, barHeight, groupMember);
	    }
	    //******************************
	    //End - Do the mean bar.
	    //******************************
	    
	    //******************************
	    //Now the rest.
	    //******************************
	    serverNumber = 0;
	    for(Enumeration e1 = (mDWindow.getStaticMainWindowSystemData()).elements(); e1.hasMoreElements() ;){
		//Get the name of the server.
		tmpSMWServer = (SMWServer) e1.nextElement();
		contextNumber = 0;
		tmpContextList = tmpSMWServer.getContextList();
		for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;){
		    //Get the next context.
		    tmpSMWContext = (SMWContext) e2.nextElement();
		    tmpThreadList = tmpSMWContext.getThreadList();
		    //Setting the context counter to zero ... this is really required as well. :-)
		    threadNumber = 0;
		    for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;){
			tmpSMWThread = (SMWThread) e3.nextElement();
			tmpThreadDataElementList = tmpSMWThread.getThreadDataList();
			
			for(Enumeration e4 = tmpThreadDataElementList.elements(); e4.hasMoreElements() ;){
			    tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
			    
			    if((tmpSMWThreadDataElement.getMappingID()) == mappingID){
				switch(mDWindow.getMetric()){
				case 0:
				    if(mDWindow.isPercent())
					value = tmpSMWThreadDataElement.getInclusivePercentValue();
				    else
					value = tmpSMWThreadDataElement.getInclusiveValue();
				    break;			    
				case 1:
				    if(mDWindow.isPercent())
					value = tmpSMWThreadDataElement.getExclusivePercentValue();
				    else
					value = tmpSMWThreadDataElement.getExclusiveValue();
				    break;
				case 2:
				    value = tmpSMWThreadDataElement.getNumberOfCalls();
				    break;
				case 3:
				    value = tmpSMWThreadDataElement.getNumberOfSubRoutines();
				    break;
				case 4:
				    value = tmpSMWThreadDataElement.getUserSecPerCall();
				    break;
				default:
				    ParaProf.systemError(null, null, "Unexpected type - MDWP value: " + mDWindow.getMetric());
				}
				
				//For consistancy in drawing, the y coord is updated at the beggining of the loop.
				yCoord = yCoord + (barSpacing);
				
				//Now select whether to draw this thread based on clip rectangle.
				if((yCoord >= yBeg) && (yCoord <= yEnd)){
				    drawBar(g2D, fmFont, value, maxValue, "n,c,t " + serverNumber + "," + contextNumber + "," + threadNumber,
					    barXCoord, yCoord, barHeight, groupMember);
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
	catch(Exception e){
	    ParaProf.systemError(e, null, "MDWP03");
	}
    }

    //Computes the number of threads this mapping exists on.
    //Computation only occurs in the first call.
    private int activeThreads(){
	if(aT==-1){
	    for(Enumeration e1 = (mDWindow.getStaticMainWindowSystemData()).elements(); e1.hasMoreElements() ;){
		//Get the name of the server.
		tmpSMWServer = (SMWServer) e1.nextElement();
		tmpContextList = tmpSMWServer.getContextList();
		for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;){
		    tmpSMWContext = (SMWContext) e2.nextElement();
		    tmpThreadList = tmpSMWContext.getThreadList();
		    for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;){
			tmpSMWThread = (SMWThread) e3.nextElement();
			tmpThreadDataElementList = tmpSMWThread.getThreadDataList();
			for(Enumeration e4 = tmpThreadDataElementList.elements(); e4.hasMoreElements() ;){
			    tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
			    if((tmpSMWThreadDataElement.getMappingID()) == mappingID)
				aT++;
				






	}
	else
	    return aT;
    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double value, double maxValue, String text,
			 int barXCoord, int yCoord, int barHeight, boolean groupMember){
	int xLength = 0;
	double d = 0.0;
	double sum = 0.0;
	String s = null;
	int stringWidth = 0;
	int stringStart = 0;

	d = (value / maxValue);
	xLength = (int) (d * defaultBarLength);
	if(xLength == 0)
	    xLength = 1;

	if((xLength > 2) && (barHeight > 2)){
	    g2D.setColor(gME.getMappingColor());
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
	    else if((gME.isGroupMember(trial.getColorChooser().getGHCMID())))
		g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
	    else{
		g2D.setColor(gME.getMappingColor());
	    }
	    g2D.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
	}
	
	//Now print the percentage to the left of the bar.
	g2D.setColor(Color.black);
	//Need to figure out how long the percentage string will be.
	if((mDWindow.isPercent()) && ((mDWindow.getMetric())<2))					
	    s = (UtilFncs.adjustDoublePresision(value, ParaProf.defaultNumberPrecision)) + "%";
	else
	    s = UtilFncs.getOutputString(mDWindow.units(),value);
	stringWidth = fmFont.stringWidth(s);
	//Now draw the percent value to the left of the bar.
	stringStart = barXCoord - xLength - stringWidth - 5;
	g2D.drawString(s, stringStart, yCoord);
	g2D.drawString(text, (barXCoord + 5), yCoord);
    }

    //******************************
    //Event listener code!!
    //******************************
    //ActionListener code.
    public void actionPerformed(ActionEvent evt){
	try{
	    Object EventSrc = evt.getSource();
	    
	    if(EventSrc instanceof JMenuItem){
		String arg = evt.getActionCommand();
		if(arg.equals("Change Function Color")){ 
		    Color tmpCol = gME.getMappingColor();
		    
		    JColorChooser tmpJColorChooser = new JColorChooser();
		    tmpCol = tmpJColorChooser.showDialog(this, "Please select a new color", tmpCol);
				if(tmpCol != null){
				    gME.setSpecificColor(tmpCol);
				    gME.setColorFlag(true);
				    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
				}
			    }
		else if(arg.equals("Reset to Generic Color")){ 
		    gME.setColorFlag(false);
		    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
		}
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "MDWP04");
	}
    }
    
    //Ok, now the mouse listeners for this panel.
    public void mouseClicked(MouseEvent evt){
	try{
	    //For the moment, I am just showing the popup menu anywhere.
	    //For a future release, there will be more here.
	    if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
		popup.show(this, evt.getX(), evt.getY());
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "MDWP05");
	}
    }
    
    public void mousePressed(MouseEvent evt) {}
    public void mouseReleased(MouseEvent evt) {}
    public void mouseEntered(MouseEvent evt) {}
    public void mouseExited(MouseEvent evt) {}
    
    public void changeInMultiples(){
	computeDefaultBarLength();
	this.repaint();
    }

    public void computeDefaultBarLength(){
	try{
	    double sliderValue = (double) mDWindow.getSliderValue();
	    double sliderMultiple = mDWindow.getSliderMultiple();
	    double result = 250*sliderValue*sliderMultiple;
	    defaultBarLength = (int) result;
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "MDWP06");
	}
    }
    
    public Dimension getPreferredSize(){
	return new Dimension(xPanelSize, (yPanelSize + 10));}

    //******************************
    //Instance data.
    //******************************
    private Vector staticNodeList;
    private int newXPanelSize = 0;
    private int newYPanelSize = 0;
    private String counterName = null;
    private int mappingID = -1;
    private String mappingName;
    GlobalMappingElement gME = null;
    private int barHeight = -1;
    private int barSpacing = -1;
    private int defaultBarLength = 250;
    private int maxXLength = 0;
    private boolean groupMember = false;
    private int serverNumber = -1;
    private int contextNumber = -1;
    private int threadNumber = -1;
    private Trial trial = null;
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

