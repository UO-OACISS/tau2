/* 
  CallPathTextWindowPanel.java

  Title:      ParaProf
  Author:     Robert Bell
  Description: 
  Things to do:
  1)Add printing support.
  2)Need to do quite a bit of work in the renderIt function,
    such as adding clipping support, and bringing it more inline
    with the rest of the system.
*/

package paraprof;

import java.util.*;
import java.text.*;
import java.awt.font.TextAttribute;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import javax.swing.*;
import java.awt.geom.*;
import javax.swing.event.*;

public class CallPathTextWindowPanel extends JPanel implements ActionListener, MouseListener, Printable, ParaProfImageInterface{
    
    public CallPathTextWindowPanel(ParaProfTrial trial,
				   int nodeID,
				   int contextID,
				   int threadID,
				   CallPathTextWindow cPTWindow,
				   boolean global,
				   boolean debug){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);
	    
	    this.nodeID = nodeID;
	    this.contextID = contextID;
	    this.threadID = threadID;
	    this.trial = trial;
	    this.cPTWindow = cPTWindow;
	    this.global = global;
	    this.debug = debug;
	    this.repaint();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CPTWP01");
	}
    }

    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    renderIt((Graphics2D) g, 0);
	}
	catch(Exception e){
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
	    int defaultNumberPrecision = ParaProf.defaultNumberPrecision;
	    int yCoord = 0;
	    
	    //In this window, a Monospaced font has to be used.  This will probably not be the same
	    //font as the rest of ParaProf.  As a result, some extra work will have to be done to calculate


	    //spacing.
	    int fontSize = trial.getPreferences().getBarHeight();
	    spacing = trial.getPreferences().getBarSpacing();
      
	    //Create font.
	    MonoFont = new Font("Monospaced", trial.getPreferences().getFontStyle(), fontSize);
	    //Compute the font metrics.
	    fmMonoFont = g2D.getFontMetrics(MonoFont);
	    maxFontAscent = fmMonoFont.getMaxAscent();
	    maxFontDescent = fmMonoFont.getMaxDescent();
	    g2D.setFont(MonoFont);
      
	    if(spacing <= (maxFontAscent + maxFontDescent)){
		spacing = spacing + 1;
	    }

	    if(global){
		ListIterator l1 = null;
		ListIterator l2 = null;
		ListIterator l3 = null;
		GlobalMapping gm = trial.getGlobalMapping();
		GlobalMappingElement gme1 = null;
		GlobalMappingElement gme2 = null;
		Integer listValue = null;
		String s = null;
		
		
		
		
		yCoord = yCoord + (spacing);
		g2D.setColor(Color.black);
		l1 = cPTWindow.getDataIterator();
		while(l1.hasNext()){
		    gme1 = (GlobalMappingElement) l1.next();
		    //Don't draw callpath mapping objects.
		    if(!(gme1.isCallPathObject())){
			l2 = gme1.getParentsIterator();
			while(l2.hasNext()){
			    listValue = (Integer)l2.next();
			    gme2 = gm.getGlobalMappingElement(listValue.intValue(),0);
			    l3 = gme1.getCallPathIDParents(listValue.intValue());
			    s = "        parent callpath(s)";
			    while(l3.hasNext()){
				s=s+":["+(((Integer)l3.next()).toString())+"]";
			    }
			    g2D.drawString("    "+gme2.getMappingName()+"["+gme2.getGlobalID()+"]"+s, 20, yCoord);
			    yCoord = yCoord + (spacing);
			}
			g2D.drawString("--> "+gme1.getMappingName()+"["+gme1.getGlobalID()+"]", 20, yCoord);
			yCoord = yCoord + (spacing);
			l2 = gme1.getChildrenIterator();
			while(l2.hasNext()){
			    listValue = (Integer)l2.next();
			    gme2 = gm.getGlobalMappingElement(listValue.intValue(),0);
			    l3 = gme1.getCallPathIDChildren(listValue.intValue());
			    s = "        child callpath(s)";
			    while(l3.hasNext()){
				s=s+":["+(((Integer)l3.next()).toString())+"]";
			    }
			    g2D.drawString("    "+gme2.getMappingName()+"["+gme2.getGlobalID()+"]"+s, 20, yCoord);
			    yCoord = yCoord + (spacing);
			}
			
			yCoord = yCoord + (spacing);
			yCoord = yCoord + (spacing);
		    }
		}
		
		if(this.debug){
		    g2D.drawString("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 20, yCoord);
		    yCoord = yCoord + (spacing);
		    l1 = cPTWindow.getDataIterator();
		    while(l1.hasNext()){
			gme1 = (GlobalMappingElement) l1.next();
			g2D.drawString("["+gme1.getGlobalID()+"] - "+gme1.getMappingName(), 20, yCoord);
			yCoord = yCoord + (spacing);
		    }
		}
	    }
	    else{
		ListIterator l1 = null;
		ListIterator l2 = null;
		ListIterator l3 = null;
		GlobalMapping gm = trial.getGlobalMapping();
		GlobalMappingElement gme1 = null;
		GlobalMappingElement gme2 = null;
		Integer listValue = null;
		String s = null;
		Thread thread = null;
		Vector functionList = null;
		GlobalThreadDataElement gtde = null;
		SMWThreadDataElement smwtde = null;
		CallPathDrawObject callPathDrawObject = null;
		double d1 = 0.0;
		double d2 = 0.0;
		int d3 = 0;

		thread = (Thread) trial.getNCT().getThread(nodeID,contextID,threadID); 
		functionList = thread.getFunctionList();

		//######
		//Populate drawObjects vector.
		//This should only happen once.
		//######
		if(drawObjects==null){
		    drawObjects = new Vector();
		    //Add five spacer objects representing the column headings.
		    drawObjects.add(new CallPathDrawObject(null, false, true));
		    drawObjects.add(new CallPathDrawObject(null, false, true));
		    drawObjects.add(new CallPathDrawObject(null, false, true));
		    drawObjects.add(new CallPathDrawObject(null, false, true));
		    drawObjects.add(new CallPathDrawObject(null, false, true));

		    l1 = cPTWindow.getDataIterator();
		    while(l1.hasNext()){
			smwtde = (SMWThreadDataElement) l1.next();
			//Don't draw callpath mapping objects.
			if(!(smwtde.isCallPathObject())){
			    l2 = smwtde.getParentsIterator();
			    while(l2.hasNext()){
				listValue = (Integer)l2.next();
				l3 = smwtde.getCallPathIDParents(listValue.intValue());
				d1 = 0.0;
				d2 = 0.0;
				d3 = 0;
				while(l3.hasNext()){
				    int tmpInt = ((Integer)l3.next()).intValue();
				    gtde = (GlobalThreadDataElement) functionList.elementAt(tmpInt);
				    d1=d1+gtde.getExclusiveValue(trial.getSelectedMetricID());
				    d2=d2+gtde.getInclusiveValue(trial.getSelectedMetricID());
				    d3=d3+gtde.getNumberOfCalls();
				}
				callPathDrawObject = new CallPathDrawObject(thread.getFunction(listValue.intValue()), true, false);
				callPathDrawObject.setExclusiveValue(d1);
				callPathDrawObject.setInclusiveValue(d2);
				callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
				callPathDrawObject.setNumberOfCalls(smwtde.getNumberOfCalls());
				drawObjects.add(callPathDrawObject);
			    }
			    callPathDrawObject = new CallPathDrawObject(thread.getFunction(smwtde.getMappingID()), false, false);
			    callPathDrawObject.setExclusiveValue(smwtde.getExclusiveValue());
			    callPathDrawObject.setInclusiveValue(smwtde.getInclusiveValue());
			    callPathDrawObject.setNumberOfCalls(smwtde.getNumberOfCalls());
			    drawObjects.add(callPathDrawObject);
			    l2 = smwtde.getChildrenIterator();
			    while(l2.hasNext()){
				listValue = (Integer)l2.next();
				l3 = smwtde.getCallPathIDChildren(listValue.intValue());
				d1 = 0.0;
				d2 = 0.0;
				d3 = 0;
				while(l3.hasNext()){
				    int tmpInt = ((Integer)l3.next()).intValue();
				    gtde = (GlobalThreadDataElement) functionList.elementAt(tmpInt);
				    d1=d1+gtde.getExclusiveValue(trial.getSelectedMetricID());
				    d2=d2+gtde.getInclusiveValue(trial.getSelectedMetricID());
				    d3=d3+gtde.getNumberOfCalls();
				    s=s+":["+tmpInt+"]";
				}
				callPathDrawObject = new CallPathDrawObject(thread.getFunction(listValue.intValue()), true, false);
				callPathDrawObject.setExclusiveValue(d1);
				callPathDrawObject.setInclusiveValue(d2);
				callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
				callPathDrawObject.setNumberOfCalls(gtde.getNumberOfCalls());
				drawObjects.add(callPathDrawObject);
			    }
			    drawObjects.add(new CallPathDrawObject(null, false, true));
			    drawObjects.add(new CallPathDrawObject(null, false, true));
			}
		    }
		}
		//######
		//End - Populate drawObjects vector.
		//######
		
		//######
		//Set panel size. 
		//######
		if(this.calculatePanelSize()){
		    for(Enumeration e = drawObjects.elements(); e.hasMoreElements() ;){
			callPathDrawObject = (CallPathDrawObject) e.nextElement();
			yHeightNeeded = yHeightNeeded + (spacing);
			max = setMax(max,callPathDrawObject.getExclusiveValue(),callPathDrawObject.getInclusiveValue());
			if(!callPathDrawObject.isSpacer()){
			    length = fmMonoFont.stringWidth(callPathDrawObject.getMappingName()) + 10;
			    if(xWidthNeeded<length)
				xWidthNeeded = length;
			}
		    }

		    base = 20;
		    startPosition = fmMonoFont.stringWidth("--> ") + base;
		    stringWidth = (fmMonoFont.stringWidth(UtilFncs.getOutputString(cPTWindow.units(),max,defaultNumberPrecision)))+10;
		    System.out.println("String: " + UtilFncs.getOutputString(cPTWindow.units(),max,defaultNumberPrecision));
		    check = fmMonoFont.stringWidth("Exclusive");
		    if(stringWidth<check)
			stringWidth = check+5;
		    numCallsWidth = (fmMonoFont.stringWidth(Integer.toString(thread.getMaxNumberOfCalls())))+10;
		    check = fmMonoFont.stringWidth("Calls/Tot.Calls");
		    if(numCallsWidth<check)
			numCallsWidth = check+5;
		    excPos = startPosition;
		    incPos = excPos+stringWidth;
		    callsPos1 = incPos+stringWidth;
		    callsPos2 = callsPos1+numCallsWidth;
		    namePos = callsPos2+numCallsWidth;
		    //Add this to the positon of the name plus a little extra.
		    xWidthNeeded = xWidthNeeded+namePos+20;
		    
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
		    if(sizeChange && instruction==0)
			revalidate();

		    this.setCalculatePanelSize(false);
		}
		//######
		//End - Set panel size. 
		//######

		int yBeg = 0;
		int yEnd = 0;
		int startElement = 0;
		int endElement = 0;
		Rectangle clipRect = null;
		
		
		if(instruction==1 || instruction==2){
		    startElement = 0;
		    endElement = ((drawObjects.size()) - 1);
		}
		else{
		    clipRect = g2D.getClipBounds();
		    yBeg = (int) clipRect.getY();
		    yEnd = (int) (yBeg + clipRect.getHeight());
		    
		    startElement = ((yBeg - yCoord) / spacing) - 1;
		    endElement  = ((yEnd - yCoord) / spacing) + 1;
		    
		    if(startElement < 0)
			startElement = 0;
		    
		    if(endElement < 0)
			endElement = 0;
		    
		    if(startElement > (drawObjects.size() - 1))
			startElement = (drawObjects.size() - 1);
		    
		    if(endElement > (drawObjects.size() - 1))
			endElement = (drawObjects.size() - 1);
		    
		    yCoord = yCoord + (startElement * spacing);
		}
		
		/*
		//At this point we can determine the size this panel will
		//require. If we need to resize, don't do any more drawing,
		//just call revalidate. Make sure we check the instruction value as we only want to
		//revalidate if we are drawing to the screen.
		if(resizePanel(fmFont, barXCoord, list, startElement, endElement) && instruction==0){
		    this.revalidate();
		    return;
		}
		*/
		
		g2D.setColor(Color.black);
		for(int i = startElement; i <= endElement; i++){
		    callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(i);
		    if(i==1){
			g2D.drawString("Exclusive", excPos, yCoord);
			g2D.drawString("Inclusive", incPos, yCoord);
			g2D.drawString("Calls/Tot.Calls", callsPos1, yCoord);
			g2D.drawString("Name[id]", namePos, yCoord);
		    }
		    else if(i==3){
			g2D.drawString("--------------------------------------------------------------------------------", excPos, yCoord);
		    }
		    else if(!callPathDrawObject.isParentChild() && !callPathDrawObject.isSpacer()){
			g2D.drawString("--> "+ (UtilFncs.getOutputString(cPTWindow.units(),callPathDrawObject.getExclusiveValue(),
									 ParaProf.defaultNumberPrecision)), base, yCoord);
			g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),callPathDrawObject.getInclusiveValue(),
								ParaProf.defaultNumberPrecision),incPos, yCoord);
			g2D.drawString(Integer.toString(callPathDrawObject.getNumberOfCalls()), callsPos1, yCoord);
			g2D.drawString(callPathDrawObject.getMappingName()+"["+callPathDrawObject.getMappingID()+"]", namePos, yCoord);
			yCoord = yCoord + (spacing);
		    }
		    else if(callPathDrawObject.isSpacer())
			yCoord = yCoord + spacing;
		    else{
			g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),callPathDrawObject.getExclusiveValue(),
								ParaProf.defaultNumberPrecision), excPos, yCoord);
			g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),callPathDrawObject.getInclusiveValue(),
								ParaProf.defaultNumberPrecision), incPos, yCoord);
			g2D.drawString(callPathDrawObject.getNumberOfCallsFromCallPathObjects()+"/"+callPathDrawObject.getNumberOfCalls(), callsPos1, yCoord);
			g2D.drawString(callPathDrawObject.getMappingName()+"["+callPathDrawObject.getMappingID()+"]", namePos, yCoord);
			yCoord = yCoord + (spacing);
		    }
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, this, "CPTWP02");
	}
    }

    private double setMax(double max, double d1, double d2){
	if(max<d1)
	    max = d1;
	if(max<d2)
	    max = d2;
	return max;
    }
	
    //####################################
    //Interface code.
    //####################################

    //######
    //ActionListener.
    //######
    public void actionPerformed(ActionEvent evt){}
    //######
    //End - ActionListener.
    //######

    //######
    //MouseListener
    //######
    public void mouseClicked(MouseEvent evt){
	try{
	    /*
	    SMWThreadDataElement sMWThreadDataElement = null;
	    //Get the location of the mouse.
	    int xCoord = evt.getX();
	    int yCoord = evt.getY();
	    
	    //Get the number of times clicked.
	    int clickCount = evt.getClickCount();
	    for(Enumeration e1 = list.elements(); e1.hasMoreElements() ;){
		sMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
		
		if(yCoord <= (sMWThreadDataElement.getYEnd())){
		    if((yCoord >= (sMWThreadDataElement.getYBeg())) && (xCoord >= (sMWThreadDataElement.getXBeg()))
		       && (xCoord <= (sMWThreadDataElement.getXEnd()))){
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
	    */
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
    public Dimension getImageSize(boolean fullScreen, boolean prependHeader){
	if(fullScreen)
	    return this.getPreferredSize();
	else
	    return cPTWindow.getSize();
    }
    //######
    //End - ParaProfImageInterface
    //######

    //####################################
    //End - Interface code.
    //####################################
    
    private void setCalculatePanelSize(boolean calculatePanelSize){
	this.calculatePanelSize = calculatePanelSize;}

    private boolean calculatePanelSize(){
	return calculatePanelSize;};

    public Dimension getPreferredSize(){
	return new Dimension(xPanelSize, (yPanelSize + 10));
    }
  
    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //Instance data.
    //####################################
    int xPanelSize = 800;
    int yPanelSize = 600;
    boolean calculatePanelSize = true;
  
    int nodeID = -1;
    int contextID = -1;
    int threadID = -1;
    private ParaProfTrial trial = null;
    CallPathTextWindow cPTWindow = null;
    boolean global = false;
    Font MonoFont = null;
    FontMetrics fmMonoFont = null;

    //Some drawing details.
    Vector drawObjects = null;
    int startLocation = 0;
    int maxFontAscent = 0;
    int maxFontDescent = 0;
    int spacing = 0;

    int check = 0;
    int base = 0;
    int startPosition = 0;
    int stringWidth = 0;
    int numCallsWidth = 0;
    int excPos = 0;
    int incPos = 0;
    int callsPos1 = 0;
    int callsPos2 = 0;
    int namePos = 0;
    double max = 0.0;
    int yHeightNeeded = 0;
    int xWidthNeeded = 0;
    int length = 0;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################
}
