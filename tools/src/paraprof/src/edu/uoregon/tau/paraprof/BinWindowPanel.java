/* 
  
BinWindowPanel.java
  
Title:      ParaProf
Author:     Robert Bell
Description:  
*/

/*
  To do: 
  1) Fix this panel.  It does not work at the moment! 
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import java.awt.print.*;
import java.awt.geom.*;
//import javax.print.*;
import edu.uoregon.tau.dms.dss.*;



public class BinWindowPanel extends JPanel implements ActionListener, MouseListener, PopupMenuListener, Printable{
  
    //**********
    //The constructors!!
    public BinWindowPanel(){
	try{
	    //Set the default tool tip for this panel.
	    this.setToolTipText("Incorrect Constructor!!!");
	}
	catch(Exception e)
	    {
		UtilFncs.systemError(e, null, "SMWP01");
	    }
    }
  
    public BinWindowPanel(ParaProfTrial trial, BinWindow bWindow, boolean normalBin, int mappingID, boolean debug){
	try{
	    //Set the default tool tip for this panel.
	    this.setToolTipText("ParaProf bar graph draw window!");
	    setBackground(Color.white);
      
	    this.trial = trial;
	    this.bWindow = bWindow;
	    this.normalBin = normalBin;
	    this.mappingID = mappingID;
	    this.debug = debug;

	    //Add this object as a mouse listener.
	    addMouseListener(this);

	    barXStart = 100;
      
	    //Add items to the first popup menu.
	    JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
	    mappingDetailsItem.addActionListener(this);
	    popup.add(mappingDetailsItem);
      
	    JMenuItem changeColorItem = new JMenuItem("Change Function Color");
	    changeColorItem.addActionListener(this);
	    popup.add(changeColorItem);
      
	    JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
	    maskMappingItem.addActionListener(this);
	    popup.add(maskMappingItem);
      
	    JMenuItem highlightMappingItem = new JMenuItem("Highlight this Function");
	    highlightMappingItem.addActionListener(this);
	    popup.add(highlightMappingItem);
      
	    JMenuItem unHighlightMappingItem = new JMenuItem("Un-Highlight this Function");
	    unHighlightMappingItem.addActionListener(this);
	    popup.add(unHighlightMappingItem);
      
	    //Add items to the second popup menu.
	    popup2.addPopupMenuListener(this);
      
	    JMenuItem tSWItem = new JMenuItem("Show Statistics Window");
	    tSWItem.addActionListener(this);
	    popup2.add(tSWItem);
      
	    if (trial.userEventsPresent()) {
		tUESWItem = new JMenuItem("Show User Event Statistics Window");
		tUESWItem.addActionListener(this);
	    }

	    popup2.add(tUESWItem);
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP02");
	}
    }
    //End - The constructors!!
    //**********
    
    public String getToolTipText(MouseEvent evt){
	return null;  
    }
    
    //******************************
    //Event listener code!!
    //******************************
  
  
    //ActionListener code.
    public void actionPerformed(ActionEvent evt){
    }
  
    //**********
    //Mouse listeners for this panel.
    public void mouseClicked(MouseEvent evt){}
    public void mousePressed(MouseEvent evt){}
    public void mouseReleased(MouseEvent evt){}
    public void mouseEntered(MouseEvent evt){}
    public void mouseExited(MouseEvent evt){}
  
    //End - Mouse listeners for this panel.
    //**********
    
    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    drawPage((Graphics2D) g, false);
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
    
	drawPage(g2, true);
    
	return Printable.PAGE_EXISTS;
    }
  
  
    public void drawPage(Graphics2D g2D, boolean print){/*
	try{
	    
	    list = bWindow.getData();

	    //Check to see if selected groups only are being displayed.
	    GlobalMapping tmpGM = trial.getGlobalMapping();
	    
	    //**********
	    //Other initializations.
	    highlighted = false;
	    xCoord = yCoord = 0;
	    //End - Other initializations.
	    //**********
	    
	    //**********
	    //Do the standard font and spacing stuff.
	    if(!(trial.getPreferences().areBarDetailsSet())){
		
		//Create font.
		Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(), 12);
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
	    //End - Do the standard font and spacing stuff.
	    //**********
      
	    //Set local spacing and bar heights.
	    int barSpacing = trial.getPreferences().getBarSpacing();
	    int barHeight = trial.getPreferences().getBarHeight();
	    
	    //Create font.
	    Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(), barHeight);
	    g2D.setFont(font);
	    FontMetrics fmFont = g2D.getFontMetrics(font);
	    
	    
	    //**********
	    //Calculating the starting positions of drawing.
	    String tmpString2 = new String("n,c,t 99,99,99");
	    int stringWidth = fmFont.stringWidth(tmpString2);
	    barXStart = stringWidth + 15;
	    int tmpXWidthCalc = barXStart + defaultBarLength;
	    int barXCoord = barXStart;
	    yCoord = yCoord + (barSpacing);
	    //End - Calculating the starting positions of drawing.
	    //**********
	    
	    
	    Vector tmpVector = trial.getGlobalMapping().getMapping(0);
	    
	    //**********
	    //Draw the counter name if required.
	    counterName = trial.getMetricName(trial.getSelectedMetricID());
	    if(counterName != null){
		g2D.drawString("COUNTER NAME: " + counterName, 5, yCoord);
		
		GlobalMappingElement tmpGME1 = (GlobalMappingElement) tmpVector.elementAt(mappingID);
		String name = tmpGME1.getMappingName();
		g2D.drawString(name, 360, yCoord);
		
		yCoord = yCoord + (barSpacing);
	    }
	    //End - Draw the counter name if required.
	    //**********
	    
	    
	    Rectangle clipRect = g2D.getClipBounds();
	    
	    int yBeg = (int) clipRect.getY();
	    int yEnd = (int) (yBeg + clipRect.getHeight());
	    //Because tooltip redraw can louse things up.  Add an extra one to draw.
	    yEnd = yEnd + barSpacing;
	    
	    yCoord = yCoord + (barSpacing);
	    
	    
	    //Set the drawing color to the text color ... in this case, black.
	    g2D.setColor(Color.black);
	    
	    g2D.drawLine(35, 430, 35, 30);
	    g2D.drawLine(35,430,585, 430);
	    
	    
	    double maxValue = 0;
	    double minValue = 0;
	    boolean start = true;
	    if(normalBin){
		for(int i=0;i<tmpVector.size();i++){
		    GlobalMappingElement tmpGME = (GlobalMappingElement) tmpVector.elementAt(i);
		    double tmpDouble = tmpGME.getTotalExclusiveValue(trial.getSelectedMetricID());
		    if(tmpDouble > maxValue)
			maxValue = tmpDouble;
		}
	    }
	    else{
		
		SMWServer tmpSMWServer = null;
		SMWContext tmpSMWContext = null;
		SMWThread tmpThread = null;
		SMWThreadDataElement tmpSMWThreadDataElement = null;
		Vector tmpContextList = null;
		Vector tmpThreadList = null;
		Vector tmpThreadDataElementList = null;
		for(Enumeration e1 = list.elements(); e1.hasMoreElements() ;){
		    //Get the name of the server.
		    tmpSMWServer = (SMWServer) e1.nextElement();
		    
		    tmpContextList = tmpSMWServer.getContextList();
		    for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;){
			//Get the next context.
			tmpSMWContext = (SMWContext) e2.nextElement();
			
			//Now draw the thread stuff for this context.
			tmpThreadList = tmpSMWContext.getThreadList();
			
			for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;){
			    tmpSMWThread = (SMWThread) e3.nextElement();
			    tmpThreadDataElementList = tmpSMWThread.getFunctionList();
			    
			    for(Enumeration e4 = tmpThreadDataElementList.elements(); e4.hasMoreElements() ;){
				tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
				if((tmpSMWThreadDataElement.getMappingID()) == mappingID){
				    double tmpDataValue = tmpSMWThreadDataElement.getExclusiveValue();
				    if(start){
					minValue = tmpDataValue;
					start = false;
				    }
				    System.out.println("value is: " + tmpDataValue);
				    if(tmpDataValue > maxValue)
					maxValue = tmpDataValue;
				    if(tmpDataValue < minValue)
					minValue = tmpDataValue;
				}
			    }
			}
		    }
		}
		
        
		//This max is not working.
		//System.out.println("Mapping ID is: " + mappingID);
		//GlobalMappingElement tmpGME = (GlobalMappingElement) tmpVector.elementAt(0);
		//maxValue = tmpGME.getMaxExclusiveValue(trial.getSelectedMetricID());
	    }
	    
	    
	    //maxValue = maxValue/1000;
	    
	    System.out.println("maxValue: " + maxValue);
	    
	    double increment = maxValue / 10;
	    
	    
	    for(int i=0; i<10; i++){
		g2D.drawLine(30,30+i*40,35,30+i*40);
		g2D.drawString(""+(10*(10-i)), 5, 33+i*40);
	    }
	    
	    for(int i=1; i<11; i++){
		g2D.drawLine(35+i*55,430,35+i*55,435);
	    }
	    
	    AffineTransform currentTransform = g2D.getTransform();
	    
	    
	    AffineTransform test = new AffineTransform();
	    test.translate(30,495);
	    test.rotate(Math.toRadians(90));
	    g2D.setTransform(test);
	    
	    AffineTransform first = new AffineTransform();
	    first.translate(27.5,0);
	    test.preConcatenate(first);
	    g2D.setTransform(test);
	    
	    AffineTransform concat = new AffineTransform();
	    concat.translate(55,0);
	    
	    
	    //for(int i=1; i<10; i++){
	      //double rightBound = i*increment;
	      //g2D.drawString(i + "/10th MV", 0, 0);
	      //test.preConcatenate(concat);
	      //g2D.setTransform(test);
	      //}
	    
	    g2D.setTransform(currentTransform);
	    g2D.drawString("Min Value  = " + minValue, 35, 450);              
	    g2D.drawString("Max Value = " + maxValue, 552, 450);
	    
	    int[] intArray = new int[10];
	    
	    for(int i=0;i<10;i++){
		intArray[i]=0;
	    }
	    
	    if(!normalBin){
		
		SMWServer tmpSMWServer = null;
		SMWContext tmpSMWContext = null;
		SMWThread tmpThread = null;
		SMWThreadDataElement tmpSMWThreadDataElement = null;
		Vector tmpContextList = null;
		Vector tmpThreadList = null;
		Vector tmpThreadDataElementList = null;
		for(Enumeration e1 = list.elements(); e1.hasMoreElements() ;){
		    //Get the name of the server.
		    tmpSMWServer = (SMWServer) e1.nextElement();
		    
		    tmpContextList = tmpSMWServer.getContextList();
		    for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;){
			//Get the next context.
			tmpSMWContext = (SMWContext) e2.nextElement();
			
			//Now draw the thread stuff for this context.
			tmpThreadList = tmpSMWContext.getThreadList();
			
			for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;){
			    tmpSMWThread = (SMWThread) e3.nextElement();
			    tmpThreadDataElementList = tmpSMWThread.getFunctionList();
			    
			    for(Enumeration e4 = tmpThreadDataElementList.elements(); e4.hasMoreElements() ;){
				tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
				if((tmpSMWThreadDataElement.getMappingID()) == mappingID){
				    double tmpDataValue = tmpSMWThreadDataElement.getExclusiveValue();
				    for(int j=10;j>0;j--){
					if(tmpDataValue <= (minValue + ((maxValue-minValue)/j))){
					    intArray[10-j]++;
					    break;
					}
				    }
				}
			    }
			}
		    }
		}
	    }
	    else{
		for(int i=0;i<tmpVector.size();i++){    
		    GlobalMappingElement tmpGME = (GlobalMappingElement) tmpVector.elementAt(i);
		    double tmpDouble = tmpGME.getTotalExclusiveValue(trial.getSelectedMetricID());
		    for(int j=10;j>0;j--){
			if(tmpDouble <= (maxValue/j)){
			    intArray[10-j]++;
			    break;
			}
		    }
		}
	    }
	    
	    for(int i=0;i<10;i++){
		System.out.println("Arr. Loc. " + i + " is: " + intArray[i]);
	    }
	    
	    g2D.setColor(Color.red);
	    
	    int num = 512;
	    System.out.println("Number of gm is: " + num);
	    for(int i=0; i<10; i++){
		if(intArray[i] != 0){
		    double tmp1 = intArray[i];
		    
		    double per = (tmp1/num) * 100;
		    int result = (int) per;
		    System.out.println("result is: " + result);
		    g2D.fillRect(38+i*55,430-(result*4),49,result*4);}
	    }
	    
	    
	    boolean sizeChange = false;   
	    //Resize the panel if needed.
	    if(tmpXWidthCalc > 600){
		xPanelSize = tmpXWidthCalc + 1;
		sizeChange = true;
	    }
	    
	    if(yCoord > 300){
		yPanelSize = yCoord + 1;
		sizeChange = true;
	    }
	    
	    if(sizeChange)
		revalidate();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMWP06");
	}*/
    }
  
    //******************************
    //PopupMenuListener code.
    //******************************
    public void popupMenuWillBecomeVisible(PopupMenuEvent evt){
	try{
	    if(trial.userEventsPresent()){
		tUESWItem.setEnabled(true);
	    }
	    else{
		tUESWItem.setEnabled(false);
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SMW03");
	}
    }
    public void popupMenuWillBecomeInvisible(PopupMenuEvent evt){}
    public void popupMenuCanceled(PopupMenuEvent evt){}
    //******************************
    //End - PopupMenuListener code.
    //****************************** 
    
    public Dimension getPreferredSize(){
	return new Dimension(xPanelSize + 10, (yPanelSize + 10));
    }

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    BinWindow bWindow = null;
    boolean normalBin = true;
    int mappingID = -1;
    int xPanelSize = 600;
    int yPanelSize = 400;

    private Vector list = null;
  
    //**********
    //Popup menu definitions.
    private JPopupMenu popup = new JPopupMenu();
    private JPopupMenu popup2 = new JPopupMenu();
  
    JMenuItem tUESWItem = new JMenuItem("Show Total User Event Statistics Windows");
  
    //**********
  
    //**********
    //Some place holder definitions - used for cycling through data lists.
    Vector contextList = null;
    Vector threadList = null;
    Vector threadDataList = null;
    SMWThread sMWThread = null;
    SMWThreadDataElement sMWThreadDataElement = null;
    //End - Place holder definitions.
    //**********
  
    //**********
    //Other useful variables for getToolTipText, mouseEvents, and paintComponent.
    int xCoord = -1;
    int yCoord = -1;
    Object clickedOnObject = null;
    //End - Other useful variables for getToolTipText, mouseEvents, and paintComponent.
    //**********
  
    //**********
    //Some misc stuff for the paintComponent function.
    String counterName = null;
    private int defaultBarLength = 500;
    String tmpString = null;
    double tmpSum = -1;
    double tmpDataValue = -1;
    Color tmpColor = null;
    boolean highlighted = false;
    int barXStart = -1;
    int numberOfColors = 0;
    //End - Some misc stuff for the paintComponent function.
    //**********
    
    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################
}
