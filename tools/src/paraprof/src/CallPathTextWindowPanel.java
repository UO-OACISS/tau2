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
				   int inServerNumber,
				   int inContextNumber,
				   int inThreadNumber,
				   CallPathTextWindow inCPTWindow,
				   boolean global){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);
	    
	    serverNumber = inServerNumber;
	    contextNumber = inContextNumber;
	    threadNumber = inThreadNumber;
	    
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
	    
	    int yCoord = 0;


	    //In this window, a Monospaced font has to be used.  This will probably not be the same
	    //font as the rest of ParaProf.  As a result, some extra work will have to be done to calculate
	    //spacing.
	    int fontSize = trial.getPreferences().getBarHeight();
	    spacing = trial.getPreferences().getBarSpacing();
      
	    int tmpXWidthCalc = 0;
      
	    String tmpString = null;
	    String dashString = "";
      
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

		Vector threadDataList = null;
		
		//Create a pruned list from the global list.
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

	    //Grab the appropriate thread.
   	    //tmpThreadDataElementList = tSWindow.getStaticMainWindowSystemData();
    
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTWP02");
	}
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
  
    int serverNumber;
    int contextNumber;
    int threadNumber;
    private Trial trial = null;
    CallPathTextWindow cPTWindow = null;
    boolean global = false;
    Font MonoFont = null;
    FontMetrics fmMonoFont = null;
    //******************************
    //End - Instance data.
    //******************************
}
