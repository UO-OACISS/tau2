/* 
  CallPathTextWindowPanel.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import java.util.*;
import java.text.*;
import java.awt.font.TextAttribute;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

public class CallPathTextWindowPanel extends JPanel implements ActionListener{
    
    public CallPathTextWindowPanel(Trial inTrial,
				   int node,
				   int context,
				   int thread,
				   CallPathTextWindow inCPTWindow,
				   boolean global){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);
	    
	    this.node = node;
	    this.context = context;
	    this.thread = thread;
	    
	    trial = inTrial;
	    cPTWindow = inCPTWindow;
	    this.global = global;
	    this.repaint();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTWP01");
	}
    }
  

    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    
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
	    fmMonoFont = g.getFontMetrics(MonoFont);
	    maxFontAscent = fmMonoFont.getMaxAscent();
	    maxFontDescent = fmMonoFont.getMaxDescent();
	    g.setFont(MonoFont);
      
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
		l1 = gm.getMappingIterator(0);
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
		
		yHeightNeeded = yHeightNeeded + (spacing);
		l1 = gm.getMappingIterator(0);
		while(l1.hasNext()){
		    gme1 = (GlobalMappingElement) l1.next();
		    yHeightNeeded = yHeightNeeded + (spacing);
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
		if(sizeChange)
		    revalidate();
		//End - Set panel size. 
		//**********
		
		
		yCoord = yCoord + (spacing);
		g.setColor(Color.black);
		l1 = gm.getMappingIterator(0);
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
			    g.drawString("    "+gme2.getMappingName()+"["+gme2.getGlobalID()+"]"+s, 20, yCoord);
			    yCoord = yCoord + (spacing);
			}
			g.drawString("--> "+gme1.getMappingName()+"["+gme1.getGlobalID()+"]", 20, yCoord);
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
			    g.drawString("    "+gme2.getMappingName()+"["+gme2.getGlobalID()+"]"+s, 20, yCoord);
			    yCoord = yCoord + (spacing);
			}
			
			yCoord = yCoord + (spacing);
			yCoord = yCoord + (spacing);
		    }
		}
		
		g.drawString("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", 20, yCoord);
		yCoord = yCoord + (spacing);
		l1 = gm.getMappingIterator(0);
		while(l1.hasNext()){
		    gme1 = (GlobalMappingElement) l1.next();
		    g.drawString("["+gme1.getGlobalID()+"] - "+gme1.getMappingName(), 20, yCoord);
		    yCoord = yCoord + (spacing);
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
		Vector staticServerList = null;
		GlobalServer globalServer = null;
		GlobalContext globalContext = null;
		GlobalThread globalThread = null;
		Vector threadDataList = null;
		GlobalThreadDataElement gtde = null;
		SMWThreadDataElement smwtde = null;
		double max = 0.0;
		double d1 = 0.0;
		double d2 = 0.0;
		int d3 = 0;

		//Find the correct global thread data element.
		staticServerList = trial.getNodes();
		globalServer = (GlobalServer) staticServerList.elementAt(node);
		Vector tmpRef = globalServer.getContextList();
		globalContext = (GlobalContext) tmpRef.elementAt(context);
		tmpRef = globalContext.getThreadList();
		globalThread = (GlobalThread) tmpRef.elementAt(thread);
		threadDataList = globalThread.getThreadDataList();

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
				gtde = (GlobalThreadDataElement) threadDataList.elementAt((((Integer)l3.next()).intValue()));
				d1=d1+gtde.getInclusiveValue(trial.getCurValLoc());
				d2=d2+gtde.getExclusiveValue(trial.getCurValLoc());
			    }
			    max = setMax(max,d1,d2);
			}
			d1=smwtde.getInclusiveValue();
			d2=smwtde.getExclusiveValue();
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
				gtde = (GlobalThreadDataElement) threadDataList.elementAt((((Integer)l3.next()).intValue()));
				d1=d1+gtde.getInclusiveValue(trial.getCurValLoc());
				d2=d2+gtde.getExclusiveValue(trial.getCurValLoc());
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
		int numCallsWidth = (fmMonoFont.stringWidth(Integer.toString(globalThread.getMaxNumberOfCalls())))+5;
		check = fmMonoFont.stringWidth("Calls/Tot.Calls");
		if(numCallsWidth<check)
		    numCallsWidth = check+5;
		int incPos = startPosition;
		int excPos = incPos+stringWidth;
		int callsPos1 = excPos+stringWidth;
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
		if(sizeChange)
		    revalidate();
		//End - Set panel size. 
		//**********


		yCoord = yCoord + (spacing);
		g.setColor(Color.black);
		g.drawString("Inclusive", incPos, yCoord);
		g.drawString("Exclusive", excPos, yCoord);
		g.drawString("Calls/Tot.Calls", callsPos1, yCoord);
		g.drawString("Name[id]", namePos, yCoord);
		yCoord = yCoord + (spacing);
		g.drawString("--------------------------------------------------------------------------------", incPos, yCoord);
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
			    s = "        parent callpath(s)";
			    while(l3.hasNext()){
				int tmpInt = ((Integer)l3.next()).intValue();
				gtde = (GlobalThreadDataElement) threadDataList.elementAt(tmpInt);
				d1=d1+gtde.getInclusiveValue(trial.getCurValLoc());
				d2=d2+gtde.getExclusiveValue(trial.getCurValLoc());
				d3=d3+gtde.getNumberOfCalls();
				s=s+":["+tmpInt+"]";
			    }
			    g.drawString(new String(Double.toString(UtilFncs.adjustDoublePresision(d1, defaultNumberPrecision))), incPos, yCoord);
			    g.drawString(new String(Double.toString(UtilFncs.adjustDoublePresision(d2, defaultNumberPrecision))), excPos, yCoord);
			    g.drawString(d3+"/"+smwtde.getNumberOfCalls(), callsPos1, yCoord);
			    gtde = (GlobalThreadDataElement) threadDataList.elementAt(listValue.intValue());
			    //g.drawString(gtde.getMappingName()+"["+gtde.getMappingID()+"]"+s, namePos, yCoord);
			    g.drawString(gtde.getMappingName()+"["+gtde.getMappingID()+"]", namePos, yCoord);
			    yCoord = yCoord + (spacing);
			}
			d1 = smwtde.getInclusiveValue(); 
			d2 = smwtde.getExclusiveValue();
			g.drawString("--> "+ (new String(Double.toString(UtilFncs.adjustDoublePresision(d1, defaultNumberPrecision)))), base, yCoord);
			g.drawString(new String(Double.toString(UtilFncs.adjustDoublePresision(d2, defaultNumberPrecision))), excPos, yCoord);
			g.drawString(Integer.toString(smwtde.getNumberOfCalls()), callsPos1, yCoord);
			g.drawString(smwtde.getMappingName()+"["+smwtde.getMappingID()+"]", namePos, yCoord);
			yCoord = yCoord + (spacing);
			l2 = smwtde.getChildrenIterator();
			while(l2.hasNext()){
			    listValue = (Integer)l2.next();
			    l3 = smwtde.getCallPathIDChildren(listValue.intValue());
			    d1 = 0.0;
			    d2 = 0.0;
			    d3 = 0;
			    s = "        child callpath(s)";
			    while(l3.hasNext()){
				int tmpInt = ((Integer)l3.next()).intValue();
				gtde = (GlobalThreadDataElement) threadDataList.elementAt(tmpInt);
				d1=d1+gtde.getInclusiveValue(trial.getCurValLoc());
				d2=d2+gtde.getExclusiveValue(trial.getCurValLoc());
				d3=d3+gtde.getNumberOfCalls();
				s=s+":["+tmpInt+"]";
			    }
			    g.drawString(new String(Double.toString(UtilFncs.adjustDoublePresision(d1, defaultNumberPrecision))), incPos, yCoord);
			    g.drawString(new String(Double.toString(UtilFncs.adjustDoublePresision(d2, defaultNumberPrecision))), excPos, yCoord);
			    gtde = (GlobalThreadDataElement) threadDataList.elementAt(listValue.intValue());
			    g.drawString(d3+"/"+gtde.getNumberOfCalls(), callsPos1, yCoord);
			    //g.drawString(gtde.getMappingName()+"["+gtde.getMappingID()+"]"+s, namePos, yCoord);
			    g.drawString(gtde.getMappingName()+"["+gtde.getMappingID()+"]", namePos, yCoord);
			    yCoord = yCoord + (spacing);
			}
			
			yCoord = yCoord + (spacing);
			yCoord = yCoord + (spacing);
		    }
		}
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, this, "CPTWP02");
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
  
    int node = -1;
    int context = -1;
    int thread = -1;
    private Trial trial = null;
    CallPathTextWindow cPTWindow = null;
    boolean global = false;
    Font MonoFont = null;
    FontMetrics fmMonoFont = null;
    //******************************
    //End - Instance data.
    //******************************
}
