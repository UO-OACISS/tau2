/* 
  
UserEventWindowPanel.java
  
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


public class UserEventWindowPanel extends JPanel implements ActionListener, MouseListener, Printable, ParaProfImageInterface{
    public UserEventWindowPanel(){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MDWP01");
	}
    }
  
    public UserEventWindowPanel(ParaProfTrial trial, int mappingID, UserEventWindow uEWindow){
	try{

	    this.trial = trial;
	    this.uEWindow = uEWindow;
 	    gME = ((GlobalMapping)trial.getGlobalMapping()).getGlobalMappingElement(mappingID, 2);
 	    mappingName = gME.getMappingName();
	    this.mappingID = mappingID;
	    barLength = baseBarLength;

	    //Want the background to be white.
	    setBackground(Color.white);
      
	    //Add this object as a mouse listener.
	    addMouseListener(this);
      
	    //Add items to the popu menu.
	    JMenuItem changeColorItem = new JMenuItem("Change Userevnet Color");
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
	    //Set the max values for this mapping.
	    //***
	    switch(uEWindow.getValueType()){
	    case 12:
		maxValue = gME.getMaxUserEventNumberValue();
		break;
	    case 14:
		maxValue = gME.getMaxUserEventMinValue();
		break;
	    case 16:
		maxValue = gME.getMaxUserEventMaxValue();
		break;
	    case 18:
		maxValue = gME.getMaxUserEventMeanValue();
		break;
	    default:
		UtilFncs.systemError(null, null, "Unexpected type - UEWP value: " + uEWindow.getValueType());
	    }

	    stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(0,maxValue)); //No units required in this window.  Thus pass in 0 for type.
	    barXCoord = barXCoord + stringWidth;
	    //***
	    //End - Set the max values for this mapping.
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

	    //Some points to note about drawing. When we draw, swing begins at the given y coord,
	    //and draws down towards the bottom of the panel. Given that we use clipping to determine
	    //what to draw, we have to be careful to draw everytime our clipping coords intersect
	    //the object. Otherwise, we run the risk of either not drawing when we need to, or 
	    //clipping out sections that we would like to be redrawn. It is no good just increasing
	    //what is being drawn to something larger than the clip rectangle, because that 
	    //will just be clipped down to the clipping rectangle size.
	    //As an example, change the marked sections below, and observe the effects
	    //when scrolling down ~20 pixels, and the gradually scrolling back up.

	    //######
	    //Draw thread information for this mapping.
	    //######
	    for(Enumeration e = (uEWindow.getData()).elements(); e.hasMoreElements() ;){
		tmpSMWThreadDataElement = (SMWThreadDataElement) e.nextElement();
		switch(uEWindow.getValueType()){
		case 12:
			value = tmpSMWThreadDataElement.getUserEventNumberValue();
		    break;
		case 14:
			value = tmpSMWThreadDataElement.getUserEventMinValue();
		    break;
		case 16:
		    value = tmpSMWThreadDataElement.getUserEventMaxValue();
		    break;
		case 18:
		    value = tmpSMWThreadDataElement.getUserEventMeanValue();
		    break;
		default:
		    UtilFncs.systemError(null, null, "Unexpected type - UEWP value: " + uEWindow.getValueType());
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
	    UtilFncs.systemError(e, null, "UEWP03");
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
	    
	    if(mappingID == (trial.getColorChooser().getUEHCMappingID())){
		g2D.setColor(trial.getColorChooser().getUEHC());
		g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
		g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
	    }
	    else{
		g2D.setColor(Color.black);
		g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
	    }
	}
	else{
	    if(mappingID == (trial.getColorChooser().getUEHCMappingID()))
		g2D.setColor(trial.getColorChooser().getUEHC());
	    else{
		g2D.setColor(gME.getMappingColor());
	    }
	    g2D.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
	}
	
	//Draw the value next to the bar.
	g2D.setColor(Color.black);
	s = UtilFncs.getOutputString(0,value); //Set the unit value (first arg) to 0).
	                                       //This will ensure that UtilFncs.getOutputString
	                                       //Does the right thing. This is of course because
	                                       //we do not have units in this display.
	stringWidth = fmFont.stringWidth(s);
	//Now draw the percent value to the left of the bar.
	stringStart = barXCoord - xLength - stringWidth - 5;
	g2D.drawString(s, stringStart, yCoord);
	g2D.drawString(text, (barXCoord + 5), yCoord);
    }

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
	    UtilFncs.systemError(e, null, "MDWP05");
	}
    }
    //######
    //End - ActionListener.
    //######
    
    //######
    //MouseListener
    //######
    
    public void mousePressed(MouseEvent evt) {}
    public void mouseReleased(MouseEvent evt) {}
    public void mouseEntered(MouseEvent evt) {}
    public void mouseExited(MouseEvent evt) {}

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
	    double sliderValue = (double) uEWindow.getSliderValue();
	    double sliderMultiple = uEWindow.getSliderMultiple();
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
	    int newYPanelSize = ((uEWindow.getData().size())+2)*barSpacing+10;
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

    //******************************
    //Instance data.
    //******************************
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
    private UserEventWindow uEWindow = null;
    private SMWThreadDataElement tmpSMWThreadDataElement = null;
    int xPanelSize = 0;
    int yPanelSize = 0;
  
    //**********
    //Popup menu definitions.
    private JPopupMenu popup = new JPopupMenu();
    //**********
  
    //******************************
    //End - Instance data.
    //******************************
}
