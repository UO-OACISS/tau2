package edu.uoregon.tau.viewer.perfcomparison;

import java.util.*;
import java.awt.*;
import java.awt.geom.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*; 
import java.awt.image.*;

import edu.uoregon.tau.dms.dss.*;

/**
 * @author lili
 * This class creates comparison panel by extending JPanel.  
 *
 */
public class ComparisonWindowPanel extends JPanel implements ActionListener, MouseListener, PopupMenuListener{
    int maxBarLength = 550;
    int barSpacing = 6;
    int barHeight = 12;
    int blankSpaceWidth = 50;
    int panelWidth = 750;
    int panelHeight = 500;
    int functionSpacing = 50;

    // height of decoration area for each function, paint gray.
    int heightOfDec;

    // the four variables define the scope of a bar area.
    int xLeft, xRight, yTop, yBottom;

    private PerfComparison env;
    private Hashtable valueHT;

    //Popup menu definition.
    private JPopupMenu popup = new JPopupMenu();
    private JMenuItem popupItem1, popupItem2, popupItem3, popupItem4;
 
    public ComparisonWindowPanel(PerfComparison env){
	super();
	setBackground(Color.white);
	this.env = env;
	
	/*
	if (env.isShowMeanValueEnabled())
	    valueHT = env.getMeanValues();
	else if (env.isShowTotalValueEnabled())
	    valueHT = env.getTotalValues();
	else if (env.isShowGroupMeanValueEnabled())
	    valueHT = env.getMeanGroupValues(env.getSelectedGroupName());
	else 
	    valueHT = env.getTotalGroupValues(env.getSelectedGroupName());
	*/

	setToolTipText("Comparison Window.");
	addMouseListener(this);

	popupItem1 = new JMenuItem("Show exclusive value");
	popupItem1.addActionListener(this);
	popup.add(popupItem1);

	popupItem2 = new JMenuItem("Show inclusive value");
	popupItem2.addActionListener(this);
	popup.add(popupItem2);

	popupItem3 = new JMenuItem("Show number of calls");
	popupItem3.addActionListener(this);	
	popup.add(popupItem3);

	popupItem4 = new JMenuItem("Show inclusive per call");
	popupItem4.addActionListener(this);
	popup.add(popupItem4);

	popup.addPopupMenuListener(this);
    }

    public String getToolTipText(MouseEvent evt){
	
      //Get the location of the mouse.
      int xCoord = evt.getX();
      int yCoord = evt.getY();
      
      if (env.isShowMeanValueEnabled())
	  valueHT = env.getMeanValues();
      else if (env.isShowTotalValueEnabled())
	  valueHT = env.getTotalValues();
      else if (env.isShowGroupMeanValueEnabled())
	  valueHT = env.getMeanGroupValues(env.getSelectedGroupName());
      else 
	  valueHT = env.getTotalGroupValues(env.getSelectedGroupName());
      
      if ((xCoord>xLeft) && (xCoord<xRight) && (yCoord>yTop) && (yCoord<yBottom)){

	  for(Enumeration e1=valueHT.keys(); e1.hasMoreElements();){
	      String strKey = (String) e1.nextElement();
	      Vector value = (Vector) valueHT.get(strKey);

	      for (Enumeration e2=value.elements(); e2.hasMoreElements();){
		  ComparisonWindowIntervalEvent currentFunc = (ComparisonWindowIntervalEvent) e2.nextElement();
		  
		  if (currentFunc.getShape().contains(xCoord, yCoord)){
		      if (env.isShowExclusiveEnabled())
			  return Double.toString(currentFunc.getExclusive()) + " usec";
		      else if (env.isShowInclusiveEnabled())
			  return Double.toString(currentFunc.getInclusive()) + " usec";
		      else if (env.isShowCallsEnabled())
			  return Double.toString(currentFunc.getNumCalls());
		      else
			  return Double.toString(currentFunc.getInclusivePerCall()) + " usec";
		  }
	      }
	  }
      }
      else { return "Right click for other summary values";}

      return null;
    }

    public void paintComponent(Graphics g)
    {	
	super.paintComponent(g);
	
	Graphics2D g2 = (Graphics2D) g;	

	// define two types of font for function-name and trial-legend.
	Font trialFont = new Font(null, Font.PLAIN, barHeight);
	Font functionFont = new Font(null, Font.BOLD, barHeight+5);

	FontMetrics fmFont = g2.getFontMetrics(trialFont);           

	// starting position to write something.
        int xPosition = blankSpaceWidth;
	int yPosition = blankSpaceWidth;

	// write label.

	String label = "COMPARE FUNCTION SUMMARY ";
		
	if (env.isShowMeanValueEnabled() || env.isShowGroupMeanValueEnabled()){
	    label += "(mean ";
	}
	else {
	    label += "(total ";
	}

	if (env.isShowExclusiveEnabled())
	    label += "exclusive " + env.getMetricName() + ")";
	else if (env.isShowInclusiveEnabled())
	    label += "inclusive " + env.getMetricName() + ")";
	else if (env.isShowCallsEnabled())
	    label += "number of calls)";
	else
	    label += "inclusive " + env.getMetricName() + " per call)";

	g2.setPaint(Color.black);
	g2.setFont(functionFont);
	g2.drawString(label, xPosition, yPosition); 

	yPosition += functionFont.getSize() + functionSpacing;	
	
	// set scope of bar area.
	xLeft = blankSpaceWidth-5;
	xRight = xLeft + panelWidth-2*(blankSpaceWidth+5);
	yTop = yPosition;

	//drawing the compared data.

	if (env.isShowMeanValueEnabled())
	    valueHT = env.getMeanValues();
	else if (env.isShowTotalValueEnabled())
	    valueHT = env.getTotalValues();		
	else if (env.isShowGroupMeanValueEnabled())
	    valueHT = env.getMeanGroupValues(env.getSelectedGroupName());
	else 
	    valueHT = env.getTotalGroupValues(env.getSelectedGroupName());

	Enumeration e1;

	if ((e1 = env.getSortedHTKeys())==null) // if not sorted.
	    e1 = valueHT.keys();
	
	// loop through compared functions
	while (e1.hasMoreElements()){

	    // get the function name.
	    String strKey = (String) e1.nextElement();
	    // get thr function values.
	    Vector valueVec = (Vector) valueHT.get(strKey); 

	    // draw decoration area.
	    g2.setPaint(Color.LIGHT_GRAY);
	    g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));	

	    heightOfDec = functionFont.getSize()+(barHeight+barSpacing)*valueVec.size()+2*barSpacing;
	    g2.fill(new Rectangle2D.Double(xPosition-5,yPosition-functionFont.getSize()+1,
					   panelWidth-2*(blankSpaceWidth+5), heightOfDec));
	    
	    // write down the function name.
		
	    g2.setFont(functionFont);
	    g2.setPaint(Color.black);
	    int strLen = g2.getFontMetrics(functionFont).stringWidth(strKey);
	    
	    //g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
	    
	    g2.drawString(strKey, (panelWidth-strLen)/2, yPosition);
	    yPosition += functionFont.getSize()+barSpacing;

	    // prepare to draw bars.		
	    g2.setFont(trialFont);
	    g2.setPaint(Color.red);
	    for(Enumeration e2 = valueVec.elements(); e2.hasMoreElements() ;){
		ComparisonWindowIntervalEvent currentFunc = (ComparisonWindowIntervalEvent) e2.nextElement();
		    
		// write trial name.
		String writtenStr = currentFunc.getTrialName(); 
		int stringWidth = fmFont.stringWidth(writtenStr);

		g2.drawString(writtenStr, xPosition, yPosition);

		//xPosition += stringWidth+10;
		xPosition += 60;
		
		double barLength; 
		    
		if (env.isShowExclusiveEnabled())
		    barLength = (currentFunc.getExclusive()*maxBarLength)/env.getMaxExclusive();
		else if (env.isShowInclusiveEnabled())
		    barLength = (currentFunc.getInclusive()*maxBarLength)/env.getMaxInclusive();
		else if (env.isShowCallsEnabled())
		    barLength = (currentFunc.getNumCalls()*maxBarLength)/env.getMaxCalls();
		else
		    barLength = (currentFunc.getInclusivePerCall()*maxBarLength)/env.getMaxInclusivePerCall();
		    
		Rectangle2D r2 = new Rectangle2D.Double(xPosition,yPosition-barHeight+1,
							barLength, barHeight);
		//System.out.println(currentFunc.getExclusive()+"************"+env.getMaxExclusive());
		currentFunc.setShape(r2);
		//g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		g2.fill(r2); 		

		// update x-position
		xPosition = blankSpaceWidth;
		yPosition += barHeight+barSpacing;
	    }
	    
	    // update y-position after drawing a function.
	    yPosition += functionSpacing;
	}

	// set scope of bar area.
	yBottom = yPosition - functionSpacing; 
	
	// resize panel
	if (yPosition > panelHeight){
	    panelHeight = yPosition + 1;
	    revalidate();
	}
    }

    public Dimension getPreferredSize(){
	return new Dimension(panelWidth, panelHeight);
    }

    public void actionPerformed(ActionEvent evt){
	Object eventSrc = evt.getSource();
                  
	if(eventSrc instanceof JMenuItem){
	    String arg = evt.getActionCommand();
	    if(arg.equals("Show inclusive value")){ // popup menu showing inclusive value
		env.setShowInclusive();
		env.resort();

		popupItem1.setEnabled(true);
		popupItem2.setEnabled(false);
		popupItem3.setEnabled(true);
		popupItem4.setEnabled(true);
		repaint();
	    }
	    else if (arg.equals("Show exclusive value")){ // popup menu showing exclusive value
		env.setShowExclusive();
		env.resort();

		popupItem1.setEnabled(false);
		popupItem2.setEnabled(true);
		popupItem3.setEnabled(true);
		popupItem4.setEnabled(true);
		repaint();
	    }
	    else if (arg.equals("Show number of calls")){ // popup menu showing number of calls
		env.setShowCalls();
		env.resort();

		popupItem1.setEnabled(true);
		popupItem2.setEnabled(true);
		popupItem3.setEnabled(false);
		popupItem4.setEnabled(true);
		repaint();
	    }
	    else if (arg.equals("Show inclusive per call")){// popup menu showing inclusive per call.
		env.setShowInclusivePerCall();
		env.resort();

		popupItem1.setEnabled(true);
		popupItem2.setEnabled(true);
		popupItem3.setEnabled(true);
		popupItem4.setEnabled(false);
		repaint();
	    }
	}
    }

    public void mouseClicked(MouseEvent evt){ 
	int xCoord = evt.getX();
	int yCoord = evt.getY();
	boolean showPopupMenu = false;

	// test whether the mouse is clicked in proper area.
	if (evt.getButton()==MouseEvent.BUTTON3){ // right button is clicked.
	    // check whether in bar area
	    if (!((xCoord>xLeft) && (xCoord<xRight) && (yCoord>yTop) && (yCoord<yBottom)))
		//if not show popup menu. 
		showPopupMenu = true;
	}

	if (showPopupMenu)
	    popup.show(this,xCoord,yCoord);
	
    }

    public void mousePressed(MouseEvent evt){}
    public void mouseReleased(MouseEvent evt){}
    public void mouseEntered(MouseEvent evt){}
    public void mouseExited(MouseEvent evt){}
    
    public void popupMenuWillBecomeVisible(PopupMenuEvent evt){
	if (env.isShowExclusiveEnabled())
	    popupItem1.setEnabled(false);
	else if (env.isShowInclusiveEnabled())
	    popupItem2.setEnabled(false);
	else if (env.isShowCallsEnabled())
	    popupItem3.setEnabled(false);
	else
	    popupItem4.setEnabled(false);
	     
     }
    
    public void popupMenuWillBecomeInvisible(PopupMenuEvent evt){}
    public void popupMenuCanceled(PopupMenuEvent evt){}
}
