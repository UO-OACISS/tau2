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

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.text.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.geom.*;
import java.text.*;
import java.awt.font.*;
import java.awt.font.TextAttribute;
import edu.uoregon.tau.dms.dss.*;

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

	    //Add this object as a mouse listener.
	    addMouseListener(this);
      
	    //Add items to the popu menu.
	    JMenuItem  jMenuItem = new JMenuItem("Show Function Details");
	    jMenuItem.addActionListener(this);
	    popup.add(jMenuItem);

	    jMenuItem = new JMenuItem("Find Function");
	    jMenuItem.addActionListener(this);
	    popup.add(jMenuItem);
      
	    jMenuItem = new JMenuItem("Change Function Color");
	    jMenuItem.addActionListener(this);
	    popup.add(jMenuItem);
      
	    jMenuItem = new JMenuItem("Reset to Generic Color");
	    jMenuItem.addActionListener(this);
	    popup.add(jMenuItem);

	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "CPTWP01");
	}
    }

    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    renderIt((Graphics2D) g, 0, false);
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
    
	renderIt(g2, 2, false);
    
	return Printable.PAGE_EXISTS;
    }  
    
    public void renderIt(Graphics2D g2D, int instruction, boolean header){
	try{
	    int defaultNumberPrecision = ParaProf.defaultNumberPrecision;
	    int yCoord = 0;
	    
	    //In this window, a Monospaced font has to be used.  This will probably not be the same
	    //font as the rest of ParaProf.  As a result, some extra work will have to be done to calculate


	    //spacing.
	    int fontSize = trial.getPreferences().getBarHeight();
	    spacing = trial.getPreferences().getBarSpacing();
      
	    //Create font.
	    monoFont = new Font("Monospaced", trial.getPreferences().getFontStyle(), fontSize);
	    //Compute the font metrics.
	    fmMonoFont = g2D.getFontMetrics(monoFont);
	    maxFontAscent = fmMonoFont.getMaxAscent();
	    maxFontDescent = fmMonoFont.getMaxDescent();
	    g2D.setFont(monoFont);
      
	    if(spacing <= (maxFontAscent + maxFontDescent)){
		spacing = spacing + 1;
	    }

	    if(global){
		ListIterator l1 = null;
		ListIterator l2 = null;
		ListIterator l3 = null;
		GlobalMapping gm = trial.getGlobalMapping();
		GlobalMappingElement gme = null;
		CallPathDrawObject callPathDrawObject = null;
		Integer listValue = null;
		String s = null;
				
		//######
		//Populate drawObjectsComplete vector.
		//This should only happen once.
		//######
		if(drawObjectsComplete==null){
		    drawObjectsComplete = new Vector();
		    //Add five spacer objects representing the column headings.
		    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
		    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
		    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
		    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));

		    l1 = cPTWindow.getDataIterator();
		    while(l1.hasNext()){
			gme = (GlobalMappingElement) l1.next();
			//Don't draw callpath mapping objects.
			if(!(gme.isCallPathObject())){
			    l2 = gme.getParentsIterator();
			    while(l2.hasNext()){
				listValue = (Integer)l2.next();
				callPathDrawObject = new CallPathDrawObject(gm.getGlobalMappingElement(listValue.intValue(),0), true, false, false);
				drawObjectsComplete.add(callPathDrawObject);
			    }
			    callPathDrawObject = new CallPathDrawObject(gme, false, false, false);
			    drawObjectsComplete.add(callPathDrawObject);
			    l2 = gme.getChildrenIterator();
			    while(l2.hasNext()){
				listValue = (Integer)l2.next();
				callPathDrawObject = new CallPathDrawObject(gm.getGlobalMappingElement(listValue.intValue(),0), false, true, false);
				drawObjectsComplete.add(callPathDrawObject);
			    }
			    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
			    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
			}
		    }
		}
		//######
		//End - Populate drawObjectsComplete vector.
		//######

		//######
		//Populate drawObjects vector.
		//######
		if(drawObjects==null){
		    drawObjects = new Vector();
		    Vector holdingPattern = new Vector();
		    boolean adding = false;
		    int state = -1;
		    int size = -1;
		    if(cPTWindow.showCollapsedView()){
			for(Enumeration e = drawObjectsComplete.elements();e.hasMoreElements() ;){
			    callPathDrawObject = (CallPathDrawObject) e.nextElement();
			    if(callPathDrawObject.isSpacer())
				    state = 0;
			    else if(callPathDrawObject.isParent()){
				if(adding)
				    state = 1;
				else
				    state = 2;
			    }
			    else if(callPathDrawObject.isChild()){
				if(adding)
				    state = 3;
				else
				    state = 4;
			    }
			    else{
				if(adding)
				    state = 5;
				else
				    state = 6;
			    }

			    switch(state){
			    case 0:
				drawObjects.add(callPathDrawObject);
				break;
			    case 1:
				adding = false;
				holdingPattern.add(callPathDrawObject);
				break;
			    case 2:
				holdingPattern.add(callPathDrawObject);
				break;
			    case 3:
				drawObjects.add(callPathDrawObject);
				break;
			    case 5:
				//Transfer holdingPattern elements to drawObjects, then add this function
				//to drawObjects.
				size = holdingPattern.size();
				for(int i=0;i<size;i++)
				    drawObjects.add(holdingPattern.elementAt(i));
				holdingPattern.clear();
				drawObjects.add(callPathDrawObject);
				//Now check to see if this object is expanded.
				if(callPathDrawObject.isExpanded())
				    adding = true;
				else
				    adding = false;
				break;
			    case 6:
				if(callPathDrawObject.isExpanded()){
				    //Transfer holdingPattern elements to drawObjects, then add this function
				    //to drawObjects.
				    size = holdingPattern.size();
				    for(int i=0;i<size;i++)
					drawObjects.add(holdingPattern.elementAt(i));
				    holdingPattern.clear();
				    adding = true;
				}
				else{
				    holdingPattern.clear();
				}
				drawObjects.add(callPathDrawObject);
				break;
			    default:
				if(this.debug())
				    System.out.println("In default state (CPTWP). State is: "+state);
			    }
			}
		    }
		    else
			drawObjects = drawObjectsComplete;
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
			if(!callPathDrawObject.isSpacer()){
			    length = fmMonoFont.stringWidth(callPathDrawObject.getMappingName()) + 10;
			    if(xWidthNeeded<length)
				xWidthNeeded = length;
			}
		    }

		    base = 20;
		    startPosition = fmMonoFont.stringWidth("--> ") + base;
		    
		    xWidthNeeded = xWidthNeeded+20;
		    
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
			viewRect = cPTWindow.getViewRect();
			yBeg = (int) viewRect.getY();
			yEnd = (int) (yBeg + viewRect.getHeight());
			/*
			System.out.println("Viewing Rectangle: xBeg,xEnd: "+viewRect.getX()+","+((viewRect.getX())+(viewRect.getWidth()))+
					   " yBeg,yEnd: "+viewRect.getY()+","+((viewRect.getY())+(viewRect.getHeight())));
			*/
		    }
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
		    
		    if(instruction==0)
			yCoord = yCoord + (startElement * spacing);
		}
		else if(instruction==2 || instruction==3){
		    startElement = 0;
		    endElement = ((drawObjects.size()) - 1);
		}

		g2D.setColor(Color.black);
		//######
		//Draw the header if required.
		//######
		if(header){
		    FontRenderContext frc = g2D.getFontRenderContext();
		    Insets insets = this.getInsets();
		    yCoord = yCoord + (spacing);
		    String headerString = cPTWindow.getHeaderString();
		    //Need to split the string up into its separate lines.
		    StringTokenizer st = new StringTokenizer(headerString, "'\n'");
		    while(st.hasMoreTokens()){
			AttributedString as = new AttributedString(st.nextToken());
			as.addAttribute(TextAttribute.FONT, monoFont);
			AttributedCharacterIterator aci = as.getIterator();
			LineBreakMeasurer lbm = new LineBreakMeasurer(aci, frc);
			float wrappingWidth = this.getSize().width - insets.left - insets.right;
			float x = insets.left;
			float y = insets.right;
			while(lbm.getPosition() < aci.getEndIndex()){
			TextLayout textLayout = lbm.nextLayout(wrappingWidth);
			yCoord+= spacing;
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
		    callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(i);
		    if(i==1){
			g2D.drawString("Name[id]", startPosition, yCoord);
			yCoord = yCoord + spacing;
		    }
		    else if(i==2){
			g2D.drawString("--------------------------------------------------------------------------------", startPosition, yCoord);
			yCoord = yCoord + spacing;
		    }
		    else if(!callPathDrawObject.isParentChild() && !callPathDrawObject.isSpacer()){
			g2D.drawString("--> ",base, yCoord);
			int mappingID = callPathDrawObject.getMappingID();
			if(trial.getColorChooser().getHighlightColorID() == mappingID){
			    g2D.setColor(Color.red);
			    g2D.drawString(callPathDrawObject.getMappingName()+"["+mappingID+"]", startPosition, yCoord);
			    g2D.setColor(Color.black);
			}
			else
			    g2D.drawString(callPathDrawObject.getMappingName()+"["+mappingID+"]", startPosition, yCoord);
			yCoord = yCoord + (spacing);
		    }
		    else if(callPathDrawObject.isSpacer())
			yCoord = yCoord + spacing;
		    else{
			int mappingID = callPathDrawObject.getMappingID();
			if(trial.getColorChooser().getHighlightColorID() == mappingID){
			    g2D.setColor(Color.red);
			    g2D.drawString(callPathDrawObject.getMappingName()+"["+callPathDrawObject.getMappingID()+"]", startPosition, yCoord);
			    g2D.setColor(Color.black);
			}
			else
			   g2D.drawString(callPathDrawObject.getMappingName()+"["+callPathDrawObject.getMappingID()+"]", startPosition, yCoord); 
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
		edu.uoregon.tau.dms.dss.Thread thread = null;
		Vector functionList = null;
		GlobalThreadDataElement gtde = null;
		SMWThreadDataElement smwtde = null;
		CallPathDrawObject callPathDrawObject = null;
		double d1 = 0.0;
		double d2 = 0.0;
		int d3 = 0;

		thread = (edu.uoregon.tau.dms.dss.Thread) trial.getNCT().getThread(nodeID,contextID,threadID); 
		functionList = thread.getFunctionList();

		//######
		//Populate drawObjectsComplete vector.
		//This should only happen once.
		//######
		if(drawObjectsComplete==null){
		    drawObjectsComplete = new Vector();
		    //Add five spacer objects representing the column headings.
		    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
		    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
		    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
		    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));

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
				callPathDrawObject = new CallPathDrawObject(thread.getFunction(listValue.intValue()), true, false, false);
				callPathDrawObject.setExclusiveValue(d1);
				callPathDrawObject.setInclusiveValue(d2);
				callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
				callPathDrawObject.setNumberOfCalls(smwtde.getNumberOfCalls());
				drawObjectsComplete.add(callPathDrawObject);
			    }
			    callPathDrawObject = new CallPathDrawObject(thread.getFunction(smwtde.getMappingID()), false, false, false);
			    callPathDrawObject.setExclusiveValue(smwtde.getExclusiveValue());
			    callPathDrawObject.setInclusiveValue(smwtde.getInclusiveValue());
			    callPathDrawObject.setNumberOfCalls(smwtde.getNumberOfCalls());
			    drawObjectsComplete.add(callPathDrawObject);
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
				}
				callPathDrawObject = new CallPathDrawObject(thread.getFunction(listValue.intValue()), false, true, false);
				callPathDrawObject.setExclusiveValue(d1);
				callPathDrawObject.setInclusiveValue(d2);
				callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
				callPathDrawObject.setNumberOfCalls(thread.getFunction(listValue.intValue()).getNumberOfCalls());
				drawObjectsComplete.add(callPathDrawObject);
			    }
			    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
			    drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
			}
		    }
		}
		//######
		//End - Populate drawObjectsComplete vector.
		//######

		//######
		//Populate drawObjects vector.
		//######
		if(drawObjects==null){
		    drawObjects = new Vector();
		    Vector holdingPattern = new Vector();
		    boolean adding = false;
		    int state = -1;
		    int size = -1;
		    if(cPTWindow.showCollapsedView()){
			for(Enumeration e = drawObjectsComplete.elements();e.hasMoreElements() ;){
			    callPathDrawObject = (CallPathDrawObject) e.nextElement();
			    if(callPathDrawObject.isSpacer())
				    state = 0;
			    else if(callPathDrawObject.isParent()){
				if(adding)
				    state = 1;
				else
				    state = 2;
			    }
			    else if(callPathDrawObject.isChild()){
				if(adding)
				    state = 3;
				else
				    state = 4;
			    }
			    else{
				if(adding)
				    state = 5;
				else
				    state = 6;
			    }

			    switch(state){
			    case 0:
				drawObjects.add(callPathDrawObject);
				break;
			    case 1:
				adding = false;
				holdingPattern.add(callPathDrawObject);
				break;
			    case 2:
				holdingPattern.add(callPathDrawObject);
				break;
			    case 3:
				drawObjects.add(callPathDrawObject);
				break;
			    case 5:
				//Transfer holdingPattern elements to drawObjects, then add this function
				//to drawObjects.
				size = holdingPattern.size();
				for(int i=0;i<size;i++)
				    drawObjects.add(holdingPattern.elementAt(i));
				holdingPattern.clear();
				drawObjects.add(callPathDrawObject);
				//Now check to see if this object is expanded.
				if(callPathDrawObject.isExpanded())
				    adding = true;
				else
				    adding = false;
				break;
			    case 6:
				if(callPathDrawObject.isExpanded()){
				    //Transfer holdingPattern elements to drawObjects, then add this function
				    //to drawObjects.
				    size = holdingPattern.size();
				    for(int i=0;i<size;i++)
					drawObjects.add(holdingPattern.elementAt(i));
				    holdingPattern.clear();
				    adding = true;
				}
				else{
				    holdingPattern.clear();
				}
				drawObjects.add(callPathDrawObject);
				break;
			    default:
				if(this.debug())
				    System.out.println("In default state (CPTWP). State is: "+state);
			    }
			}
		    }
		    else
			drawObjects = drawObjectsComplete;
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
			viewRect = cPTWindow.getViewRect();
			yBeg = (int) viewRect.getY();
			yEnd = (int) (yBeg + viewRect.getHeight());
			/*
			System.out.println("Viewing Rectangle: xBeg,xEnd: "+viewRect.getX()+","+((viewRect.getX())+(viewRect.getWidth()))+
					   " yBeg,yEnd: "+viewRect.getY()+","+((viewRect.getY())+(viewRect.getHeight())));
			*/
		    }
		    
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
		    
		    if(instruction==0)
			yCoord = yCoord + (startElement * spacing);
		}
		else if(instruction==2 || instruction==3){
		    startElement = 0;
		    endElement = ((drawObjects.size()) - 1);
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
		//######
		//Draw the header if required.
		//######
		if(header){
		    yCoord = yCoord + (spacing);
		    String headerString = cPTWindow.getHeaderString();
		    //Need to split the string up into its separate lines.
		    StringTokenizer st = new StringTokenizer(headerString, "'\n'");
		    while(st.hasMoreTokens()){
			g2D.drawString(st.nextToken(), 15, yCoord);
			yCoord = yCoord + (spacing);
		    }
		    lastHeaderEndPosition = yCoord;
		}
		//######
		//End - Draw the header if required.
		//######
		for(int i = startElement; i <= endElement; i++){
		    callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(i);
		    if(i==1){
			g2D.drawString("Exclusive", excPos, yCoord);
			g2D.drawString("Inclusive", incPos, yCoord);
			g2D.drawString("Calls/Tot.Calls", callsPos1, yCoord);
			g2D.drawString("Name[id]", namePos, yCoord);
			yCoord = yCoord + spacing;
		    }
		    else if(i==2){
			g2D.drawString("--------------------------------------------------------------------------------", excPos, yCoord);
			yCoord = yCoord + spacing;
		    }
		    else if(!callPathDrawObject.isParentChild() && !callPathDrawObject.isSpacer()){
			g2D.drawString("--> ", base, yCoord);
			g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),callPathDrawObject.getExclusiveValue(),
									 ParaProf.defaultNumberPrecision), excPos, yCoord);
			g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),callPathDrawObject.getInclusiveValue(),
								ParaProf.defaultNumberPrecision),incPos, yCoord);
			g2D.drawString(Integer.toString(callPathDrawObject.getNumberOfCalls()), callsPos1, yCoord);
			int mappingID = callPathDrawObject.getMappingID();
			if(trial.getColorChooser().getHighlightColorID() == mappingID){
			    g2D.setColor(Color.red);
			    g2D.drawString(callPathDrawObject.getMappingName()+"["+mappingID+"]", namePos, yCoord);
			    g2D.setColor(Color.black);
			}
			else
			    g2D.drawString(callPathDrawObject.getMappingName()+"["+mappingID+"]", namePos, yCoord);
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
			int mappingID = callPathDrawObject.getMappingID();
			if(trial.getColorChooser().getHighlightColorID() == mappingID){
			    g2D.setColor(Color.red);
			    g2D.drawString(callPathDrawObject.getMappingName()+"["+callPathDrawObject.getMappingID()+"]", namePos, yCoord);
			    g2D.setColor(Color.black);
			}
			else
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
    public void actionPerformed(ActionEvent evt){
	try{
	    Object EventSrc = evt.getSource();
	    
	    CallPathDrawObject callPathDrawObject = null;
	    
	    if(EventSrc instanceof JMenuItem){
		String arg = evt.getActionCommand();
		if(arg.equals("Show Function Details")){
		    if(clickedOnObject instanceof CallPathDrawObject){
			callPathDrawObject = (CallPathDrawObject) clickedOnObject;
			//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
			trial.getColorChooser().setHighlightColorID(callPathDrawObject.getMappingID());
			MappingDataWindow  mappingDataWindow = new MappingDataWindow(trial,callPathDrawObject.getMappingID(),
										     trial.getStaticMainWindow().getSMWData(), this.debug());
			trial.getSystemEvents().addObserver(mappingDataWindow);
			mappingDataWindow.show();
		    }
		}
		else if(arg.equals("Find Function")){ 
		    if(clickedOnObject instanceof CallPathDrawObject){
			int mappingID = ((CallPathDrawObject) clickedOnObject).getMappingID();
			int size = drawObjects.size();
			for(int i=0;i<size;i++){
			    callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(i);
			    if((callPathDrawObject.getMappingID()==mappingID)&&(!callPathDrawObject.isParentChild())){
				Dimension dimension = cPTWindow.getViewportSize();
				cPTWindow.setVerticalScrollBarPosition((i*(trial.getPreferences().getBarSpacing()))-((int)dimension.getHeight()/2));
				trial.getColorChooser().setHighlightColorID(mappingID);
				return;
			    }
			}
		    }
		}
		else if(arg.equals("Change Function Color")){ 
		    if(clickedOnObject instanceof CallPathDrawObject){
			int mappingID = ((CallPathDrawObject) clickedOnObject).getMappingID();
			GlobalMapping globalMapping = trial.getGlobalMapping();
			GlobalMappingElement  globalMappingElement = (GlobalMappingElement) globalMapping.getGlobalMappingElement(mappingID, 0);
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
		    if(clickedOnObject instanceof CallPathDrawObject){
			int mappingID = ((CallPathDrawObject) clickedOnObject).getMappingID();
			GlobalMapping globalMapping = trial.getGlobalMapping();
			GlobalMappingElement  globalMappingElement = (GlobalMappingElement) globalMapping.getGlobalMappingElement(mappingID, 0);
			globalMappingElement.setColorFlag(false);
			trial.getSystemEvents().updateRegisteredObjects("colorEvent");
		    }
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TSWP04");
	}
    }
    //######
    //End - ActionListener
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

	    CallPathDrawObject callPathDrawObject = null;

	    //Calculate which CallPathDrawObject was clicked on.
	    int index = (yCoord-1)/(trial.getPreferences().getBarSpacing()) + 1;

	    if(index<drawObjects.size()){
		callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(index);
		if(!callPathDrawObject.isSpacer()){
		    if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
			clickedOnObject = callPathDrawObject;
			popup.show(this, evt.getX(), evt.getY());
			return;
		    }
		    else{
			//Check to see if the click occured to the left of startPosition.
			if(xCoord<startPosition){
			    if(!callPathDrawObject.isParentChild()){
				if(callPathDrawObject.isExpanded())
				   callPathDrawObject.setExpanded(false);
				else
				   callPathDrawObject.setExpanded(true);
			    }
			    drawObjects = null;
			}
			//Want to set the clicked on mapping to the current highlight color or, if the one
			//clicked on is already the current highlighted one, set it back to normal.
			if((trial.getColorChooser().getHighlightColorID()) == -1){
			    trial.getColorChooser().setHighlightColorID(callPathDrawObject.getMappingID());
			}
			else{
			    if(!((trial.getColorChooser().getHighlightColorID()) == (callPathDrawObject.getMappingID())))
				trial.getColorChooser().setHighlightColorID(callPathDrawObject.getMappingID());
			    else
				trial.getColorChooser().setHighlightColorID(-1);
			}
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
	    d = cPTWindow.getSize();
	d.setSize(d.getWidth(),d.getHeight()+lastHeaderEndPosition);
	return d;
    }
    //######
    //End - ParaProfImageInterface
    //######

    //####################################
    //End - Interface code.
    //####################################

    public void resetAllDrawObjects(){
	drawObjectsComplete.clear();
	drawObjectsComplete = null;
	drawObjects.clear();
	drawObjects = null;
    }
    
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
    Font monoFont = null;
    FontMetrics fmMonoFont = null;

    //Some drawing details.
    Vector drawObjectsComplete = null;
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

    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private int lastHeaderEndPosition = 0;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################
}
