/* 
  
MappingDataWindowPanel.java
  
Title:      ParaProf
Author:     Robert Bell
Description:
  
Things to do:
  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import java.awt.geom.*;


public class MappingDataWindowPanel extends JPanel implements ActionListener, MouseListener, Printable, ParaProfImageInterface{
    public MappingDataWindowPanel(){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP01");
	}
    }
  
    public MappingDataWindowPanel(ParaProfTrial trial, int mappingID, MappingDataWindow mDWindow){
	try{

	    this.trial = trial;
	    this.mDWindow = mDWindow;
 	    gME = ((GlobalMapping)trial.getGlobalMapping()).getGlobalMappingElement(mappingID, 0);
 	    mappingName = gME.getMappingName();
	    this.mappingID = mappingID;
	    barLength = baseBarLength;

	    //Want the background to be white.
	    setBackground(Color.white);
      
	    //Add this object as a mouse listener.
	    addMouseListener(this);
      
	    //Add items to the popu menu.
	    JMenuItem changeColorItem = new JMenuItem("Change Function Color");
	    changeColorItem.addActionListener(this);
	    popup.add(changeColorItem);
      
	    JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
	    maskMappingItem.addActionListener(this);
	    popup.add(maskMappingItem);

	    
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP02");
	}
	
    }

    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    renderIt((Graphics2D) g, 0);
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
    
	renderIt(g2, 2);
    
	return Printable.PAGE_EXISTS;
    }

    public void renderIt(Graphics2D g2D, int instruction){
	try{
	    double value = 0.0;
	    double maxValue = 0.0;
	    int stringWidth = 0;
	    int yCoord = 0;
	    int barXCoord = barLength + textOffset;
	    
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
	    //Set the max and mean values for this mapping.
	    //***
	    switch(mDWindow.getValueType()){
	    case 2:
		if(mDWindow.isPercent()){
		    maxValue = gME.getMaxExclusivePercentValue(trial.getSelectedMetricID());
		    value = gME.getMeanExclusivePercentValue(trial.getSelectedMetricID());
		}
		else{
		    maxValue = gME.getMaxExclusiveValue(trial.getSelectedMetricID());
		    value = gME.getMeanExclusiveValue(trial.getSelectedMetricID());
		}
		break;
	    case 4:
		if(mDWindow.isPercent()){
		    maxValue = gME.getMaxInclusivePercentValue(trial.getSelectedMetricID());
		    value = gME.getMeanInclusivePercentValue(trial.getSelectedMetricID());
		}
		else{
		    maxValue = gME.getMaxInclusiveValue(trial.getSelectedMetricID());
		    value = gME.getMeanInclusiveValue(trial.getSelectedMetricID());
		}
		break;	
	    case 6:
		maxValue = gME.getMaxNumberOfCalls();
		value = gME.getMeanNumberOfCalls();
		break;
	    case 8:
		maxValue = gME.getMaxNumberOfSubRoutines();
		value = gME.getMeanNumberOfSubRoutines();
		break;
	    case 10:
		maxValue = gME.getMaxUserSecPerCall(trial.getSelectedMetricID());
		value = gME.getMeanUserSecPerCall(trial.getSelectedMetricID());
		break;
	    default:
		UtilFncs.systemError(null, null, "Unexpected type - MDWP value: " + mDWindow.getValueType());
	    }

	    if(mDWindow.isPercent()){
		stringWidth = fmFont.stringWidth(UtilFncs.getTestString(maxValue, ParaProf.defaultNumberPrecision) + "%");
		barXCoord = barXCoord + stringWidth;
	    }
	    else{
		stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(mDWindow.units(),maxValue));
		barXCoord = barXCoord + stringWidth;
	    }
	    //***
	    //End - Set the max and mean values for this mapping.
	    //***
	    
	    //At this point we can determine the size this panel will
	    //require. If we need to resize, don't do any more drawing,
	    //just call revalidate.
	    if(resizePanel(fmFont, barXCoord) && instruction==0){
		this.revalidate();
		return;
	    }

	    int yBeg = 0;
	    int yEnd = 0;
	    Rectangle clipRect = null;

	    if(instruction==1 || instruction==2){
		yBeg = 0;
		yEnd = yPanelSize;
	    }
	    else{
		clipRect = g2D.getClipBounds();
		yBeg = (int) clipRect.getY();
		yEnd = (int) (yBeg + clipRect.getHeight());
		yEnd = yEnd + barSpacing;
	    }

	    //Check for group membership.
	    groupMember = gME.isGroupMember(trial.getColorChooser().getGHCMID());


	    //Some points to note about drawing. When we draw, swing begins at the given y coord,
	    //and draws down towards the bottom of the panel. Given that we use clipping to determine
	    //what to draw, we have to be careful to draw everytime our clipping coords intersect
	    //the object. Otherwise, we run the risk of either not drawing when we need to, or 
	    //clipping out sections that we would like to be redrawn. It is no good just increasing
	    //what is being drawn to something larger than the clip rectangle, because that 
	    //will just be clipped down to the clipping rectangle size.
	    //As an example, change the marked sections below, and observe the effects
	    //when scrolling down ~20 pixels, and the gradually scrolling back up.

	    //Draw mean information.
	    yCoord = yCoord + (barSpacing); //Comment this
	    if((yCoord >= yBeg) && (yCoord <= yEnd)){
		//yCoord = yCoord + (barSpacing);//Uncomment this.
		drawBar(g2D, fmFont, value, maxValue, "mean", barXCoord, yCoord, barHeight, groupMember, instruction);
	    }
	    //else{//Uncomment this.
	    //yCoord = yCoord + (barSpacing);//Uncomment this.
	    //}//Uncomment this.
	    
	    //######
	    //Draw thread information for this mapping.
	    //######
	    for(Enumeration e = (mDWindow.getData()).elements(); e.hasMoreElements() ;){
		tmpSMWThreadDataElement = (SMWThreadDataElement) e.nextElement();
		switch(mDWindow.getValueType()){
		case 2:
		    if(mDWindow.isPercent())
			value = tmpSMWThreadDataElement.getExclusivePercentValue();
		    else
			value = tmpSMWThreadDataElement.getExclusiveValue();
		    break;
		case 4:
		    if(mDWindow.isPercent())
			value = tmpSMWThreadDataElement.getInclusivePercentValue();
		    else
			value = tmpSMWThreadDataElement.getInclusiveValue();
		    break;
		case 6:
		    value = tmpSMWThreadDataElement.getNumberOfCalls();
		    break;
		case 8:
		    value = tmpSMWThreadDataElement.getNumberOfSubRoutines();
		    break;
		case 10:
		    value = tmpSMWThreadDataElement.getUserSecPerCall();
		    break;
		default:
		    UtilFncs.systemError(null, null, "Unexpected type - MDWP value: " + mDWindow.getValueType());
		}
		
		//For consistancy in drawing, the y coord is updated at the beginning of the loop.
		yCoord = yCoord + (barSpacing);
		
		//Now select whether to draw this thread based on clip rectangle.
		if((yCoord >= yBeg) && (yCoord <= yEnd)){
		    drawBar(g2D, fmFont, value, maxValue,
			    "n,c,t " + (tmpSMWThreadDataElement.getNodeID()) +
			    "," + (tmpSMWThreadDataElement.getContextID()) +
			    "," + (tmpSMWThreadDataElement.getThreadID()),
			    barXCoord, yCoord, barHeight, groupMember, instruction);
		}
	    }
	    //######
	    //End - Draw thread information for this mapping.
	    //######
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP03");
	}
    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double value, double maxValue, String text,
			 int barXCoord, int yCoord, int barHeight, boolean groupMember, int instruction){
	int xLength = 0;
	double d = 0.0;
	String s = null;
	int stringWidth = 0;
	int stringStart = 0;
	d = (value / maxValue);
	xLength = (int) (d * barLength);
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
	
	//Draw the value next to the bar.
	g2D.setColor(Color.black);
	//Do not want to put a percent sign after the bar if we are not exclusive or inclusive.
	if((mDWindow.isPercent()) && ((mDWindow.getValueType())<=4))					
	    s = (UtilFncs.adjustDoublePresision(value, ParaProf.defaultNumberPrecision)) + "%";
	else
	    s = UtilFncs.getOutputString(mDWindow.units(),value);
	stringWidth = fmFont.stringWidth(s);
	//Now draw the percent value to the left of the bar.
	stringStart = barXCoord - xLength - stringWidth - 5;
	g2D.drawString(s, stringStart, yCoord);
	g2D.drawString(text, (barXCoord + 5), yCoord);
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
	    //For the moment, I am just showing the popup menu anywhere.
	    //For a future release, there will be more here.
	    if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
		popup.show(this, evt.getX(), evt.getY());
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP05");
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
    public Dimension getImageSize(){
	return this.getPreferredSize();
    }
    //######
    //End - ParaProfImageInterface
    //######

    //####################################
    //End - Interface code.
    //####################################

    public void changeInMultiples(){
	computeBarLength();
	this.repaint();
    }

    public void computeBarLength(){
	try{
	    double sliderValue = (double) mDWindow.getSliderValue();
	    double sliderMultiple = mDWindow.getSliderMultiple();
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
	    int newYPanelSize = ((mDWindow.getData().size())+2)*barSpacing+10;
	    int[] nct = trial.getMaxNCTNumbers();
	    String nctString = "n,c,t " + nct[0] + "," + nct[1] + "," + nct[2];;
	    int newXPanelSize = barXCoord+5+(fmFont.stringWidth(nctString))+ 25;
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
	return new Dimension(xPanelSize, yPanelSize);}

    //####################################
    //Instance data.
    //####################################
    private String counterName = null;
    private int mappingID = -1;
    private String mappingName;
    GlobalMappingElement gME = null;
    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 250;
    private int barLength = 0;
    private int textOffset = 60;
    private int maxXLength = 0;
    private boolean groupMember = false;
    private ParaProfTrial trial = null;
    private MappingDataWindow mDWindow = null;
    private SMWThreadDataElement tmpSMWThreadDataElement = null;
    int xPanelSize = 0;
    int yPanelSize = 0;
  
    private JPopupMenu popup = new JPopupMenu();
    //####################################
    //Instance data.
    //####################################
}
