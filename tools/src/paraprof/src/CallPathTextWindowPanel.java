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

public class CallPathTextWindowPanel extends JPanel implements ActionListener, Printable, ParaProfImageInterface{
    
    public CallPathTextWindowPanel(ParaProfTrial trial,
				   int nodeID,
				   int contextID,
				   int threadID,
				   CallPathTextWindow inCPTWindow,
				   boolean global){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);
	    
	    this.nodeID = nodeID;
	    this.contextID = contextID;
	    this.threadID = threadID;
	    
	    this.trial = trial;
	    cPTWindow = inCPTWindow;
	    this.global = global;
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
      
	    int tmpXWidthCalc = 0;
      
	    String tmpString = null;
      
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
		
		//**********
		//Set panel size.
		//Now we have reached here, we can calculate the size this panel
		//needs to be.  We might have to call a revalidate to increase
		//its size.
		int yHeightNeeded = 0;
		int xWidthNeeded = 0;
		yHeightNeeded = yHeightNeeded + (spacing);
		l1 = cPTWindow.getDataIterator(); 
		while(l1.hasNext()){
		    gme1 = (GlobalMappingElement) l1.next();
		    //Don't draw callpath mapping objects.
		    if(!(gme1.isCallPathObject())){
			l2 = gme1.getParentsIterator();
			while(l2.hasNext()){
			    listValue = (Integer)l2.next();
			    yHeightNeeded = yHeightNeeded + (spacing);
			}
			if((gme1.getMappingName().length())> xWidthNeeded)
			    xWidthNeeded=gme1.getMappingName().length();
			yHeightNeeded = yHeightNeeded + (spacing);
			l2 = gme1.getChildrenIterator();
			while(l2.hasNext()){
			    listValue = (Integer)l2.next();
			    yHeightNeeded = yHeightNeeded + (spacing);
			}
			yHeightNeeded = yHeightNeeded + (spacing);
			yHeightNeeded = yHeightNeeded + (spacing);
		    }
		}
		
		if(UtilFncs.debug){
		    yHeightNeeded = yHeightNeeded + (spacing);
		    l1 = cPTWindow.getDataIterator();
		    while(l1.hasNext()){
			gme1 = (GlobalMappingElement) l1.next();
			yHeightNeeded = yHeightNeeded + (spacing);
		    }
		}
		
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
		//End - Set panel size. 
		//**********
		
		
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
		
		if(UtilFncs.debug){
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
		double max = 0.0;
		double d1 = 0.0;
		double d2 = 0.0;
		int d3 = 0;

		thread = (Thread) trial.getNCT().getThread(nodeID,contextID,threadID); 
		functionList = thread.getFunctionList();

		//**********
		//Set panel size.
		//Now we have reached here, we can calculate the size this panel
		//needs to be.  We might have to call a revalidate to increase
		//its size.
		int yHeightNeeded = 0;
		int xWidthNeeded = 0;
		yHeightNeeded = yHeightNeeded + (spacing);
		yHeightNeeded = yHeightNeeded + (spacing);
		yHeightNeeded = yHeightNeeded + (spacing);
		l1 = cPTWindow.getDataIterator();
		while(l1.hasNext()){
		    smwtde = (SMWThreadDataElement) l1.next();
		    //Don't draw callpath mapping objects.
		    if(!(smwtde.isCallPathObject())){
			l2 = smwtde.getParentsIterator();
			while(l2.hasNext()){
			    listValue = (Integer)l2.next();
			    yHeightNeeded = yHeightNeeded + (spacing);
			    l3 = smwtde.getCallPathIDParents(listValue.intValue());
			    d1 = 0.0;
			    d2 = 0.0;
			    while(l3.hasNext()){
				gtde = (GlobalThreadDataElement) functionList.elementAt((((Integer)l3.next()).intValue()));
				d1=d1+gtde.getExclusiveValue(trial.getSelectedMetricID());
				d2=d2+gtde.getInclusiveValue(trial.getSelectedMetricID());
			    }
			    max = setMax(max,d1,d2);
			}
			d1=smwtde.getExclusiveValue();
			d2=smwtde.getInclusiveValue();
			max = setMax(max,d1,d2);
			yHeightNeeded = yHeightNeeded + (spacing);
			l2 = smwtde.getChildrenIterator();
			while(l2.hasNext()){
			    listValue = (Integer)l2.next();
			    yHeightNeeded = yHeightNeeded + (spacing);
			    l3 = smwtde.getCallPathIDChildren(listValue.intValue());
			    d1 = 0.0;
			    d2 = 0.0;
			    while(l3.hasNext()){
				gtde = (GlobalThreadDataElement) functionList.elementAt((((Integer)l3.next()).intValue()));
				d1=d1+gtde.getExclusiveValue(trial.getSelectedMetricID());
				d2=d2+gtde.getInclusiveValue(trial.getSelectedMetricID());
			    }
			    max = setMax(max,d1,d2);
			    yHeightNeeded = yHeightNeeded + (spacing);
			    yHeightNeeded = yHeightNeeded + (spacing);
			}
		    }
		}

		int check = 0;
		int base = 20;
		int startPosition = fmMonoFont.stringWidth("--> ") + base;
		int stringWidth = (fmMonoFont.stringWidth(UtilFncs.getTestString(max,defaultNumberPrecision)))+5;
		check = fmMonoFont.stringWidth("Exclusive");
		if(stringWidth<check)
		    stringWidth = check+5;
		int numCallsWidth = (fmMonoFont.stringWidth(Integer.toString(thread.getMaxNumberOfCalls())))+5;
		check = fmMonoFont.stringWidth("Calls/Tot.Calls");
		if(numCallsWidth<check)
		    numCallsWidth = check+5;
		int excPos = startPosition;
		int incPos = excPos+stringWidth;
		int callsPos1 = incPos+stringWidth;
		int callsPos2 = callsPos1+numCallsWidth;
		int namePos = callsPos2+numCallsWidth;

		//Still need to figure out how long the names might be.
		l1 = (ParaProfIterator)gm.getMappingIterator(0);
		while(l1.hasNext()){
		    int length = 0;
		    gme1 = (GlobalMappingElement) l1.next();
		    if(!(gme1.isCallPathObject())){
			length = (gme1.getMappingName()).length();
			if(xWidthNeeded<length)
			    xWidthNeeded = length;
		    }
		}
		//Add this to the positon of the name plus a little extra.
		xWidthNeeded = xWidthNeeded+namePos+5;

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
		//End - Set panel size. 
		//**********

		yCoord = yCoord + (spacing);
		g2D.setColor(Color.black);
		g2D.drawString("Exclusive", excPos, yCoord);
		g2D.drawString("Inclusive", incPos, yCoord);
		g2D.drawString("Calls/Tot.Calls", callsPos1, yCoord);
		g2D.drawString("Name[id]", namePos, yCoord);
		yCoord = yCoord + (spacing);
		g2D.drawString("--------------------------------------------------------------------------------", incPos, yCoord);
		yCoord = yCoord + (spacing);
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
			    g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),d1,ParaProf.defaultNumberPrecision), excPos, yCoord);
			    g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),d2,ParaProf.defaultNumberPrecision), incPos, yCoord);
			    g2D.drawString(d3+"/"+smwtde.getNumberOfCalls(), callsPos1, yCoord);
			    gtde = (GlobalThreadDataElement) functionList.elementAt(listValue.intValue());
			    g2D.drawString(gtde.getMappingName()+"["+gtde.getMappingID()+"]", namePos, yCoord);
			    yCoord = yCoord + (spacing);
			}
			d1 = smwtde.getExclusiveValue(); 
			d2 = smwtde.getInclusiveValue();
			g2D.drawString("--> "+ (UtilFncs.getOutputString(cPTWindow.units(),d1,ParaProf.defaultNumberPrecision)), base, yCoord);
			g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),d2,ParaProf.defaultNumberPrecision), incPos, yCoord);
			g2D.drawString(Integer.toString(smwtde.getNumberOfCalls()), callsPos1, yCoord);
			g2D.drawString(smwtde.getMappingName()+"["+smwtde.getMappingID()+"]", namePos, yCoord);
			yCoord = yCoord + (spacing);
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
			    g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),d1,ParaProf.defaultNumberPrecision), excPos, yCoord);
			    g2D.drawString(UtilFncs.getOutputString(cPTWindow.units(),d2,ParaProf.defaultNumberPrecision), incPos, yCoord);
			    gtde = (GlobalThreadDataElement) functionList.elementAt(listValue.intValue());
			    g2D.drawString(d3+"/"+gtde.getNumberOfCalls(), callsPos1, yCoord);
			    g2D.drawString(gtde.getMappingName()+"["+gtde.getMappingID()+"]", namePos, yCoord);
			    yCoord = yCoord + (spacing);
			}
			
			yCoord = yCoord + (spacing);
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
	
    public void actionPerformed(ActionEvent evt){}

    public Dimension getImageSize(){
	return this.getPreferredSize();
    }
 
    public Dimension getPreferredSize(){
	return new Dimension(xPanelSize, (yPanelSize + 10));
    }
  
    //******************************
    //Instance data.
    //******************************
    int xPanelSize = 800;
    int yPanelSize = 600;
  
    //Some drawing details.
    int startLocation = 0;
    int maxFontAscent = 0;
    int maxFontDescent = 0;
    int spacing = 0;
  
    int nodeID = -1;
    int contextID = -1;
    int threadID = -1;
    private ParaProfTrial trial = null;
    CallPathTextWindow cPTWindow = null;
    boolean global = false;
    Font MonoFont = null;
    FontMetrics fmMonoFont = null;
    //******************************
    //End - Instance data.
    //******************************
}
