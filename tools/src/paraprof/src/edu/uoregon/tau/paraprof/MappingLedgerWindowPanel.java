
//MappingLedgerWindowPanel.

/* 
   Title:      ParaProf
   Author:     Robert Bell
   Description:

   Things to do:
   1) Add clipping support to this window.
   2) Add resize method.
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import java.awt.geom.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import edu.uoregon.tau.dms.dss.*;

public class MappingLedgerWindowPanel extends JPanel implements ActionListener, MouseListener, Printable, ParaProfImageInterface{

    public MappingLedgerWindowPanel(){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MLWP01");
	}
	
    }
  
    public MappingLedgerWindowPanel(ParaProfTrial trial, MappingLedgerWindow mLWindow, int windowType, boolean debug){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);
      
	    this.trial = trial;
	    this.mLWindow = mLWindow;
	    this.windowType = windowType;
	    this.debug = debug;
      
	    //Add this object as a mouse listener.
	    addMouseListener(this);

	    JMenuItem jMenuItem = null;
	    switch(windowType){
	    case 0:
		jMenuItem = new JMenuItem("Show Function Details");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);
		
		jMenuItem = new JMenuItem("Change Function Color");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);

		jMenuItem = new JMenuItem("Reset to Generic Color");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);

		break;
	    case 1:
		jMenuItem = new JMenuItem("Show Function Details");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);
		
		jMenuItem = new JMenuItem("Change Group Color");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);

		jMenuItem = new JMenuItem("Reset to Generic Color");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);
		
		jMenuItem = new JMenuItem("Show This Group Only");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);
		
		jMenuItem = new JMenuItem("Show All Groups Except This One");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);
		
		jMenuItem = new JMenuItem("Show All Groups");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);

		break;
	    case 2:
		jMenuItem = new JMenuItem("Show User Event Details");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);
		
		jMenuItem = new JMenuItem("Change User Event Color");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);

		jMenuItem = new JMenuItem("Reset to Generic Color");
		jMenuItem.addActionListener(this);
		popup.add(jMenuItem);

		break;
	    }
      
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MLWP02");
	}
    }

    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    renderIt((Graphics2D) g, 0, false);
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
    
	renderIt(g2, 2, false);
    
	return Printable.PAGE_EXISTS;
    }
    
    public void renderIt(Graphics2D g2D, int instruction, boolean header){
  	try{
	    list = mLWindow.getData();

	    //######
	    //Some declarations.
	    //######
	    int xCoord = 0;
	    int yCoord = 0;
	    int barXCoord = 0;
	    int tmpXWidthCalc = 0;
	    //######
	    //End - Some declarations.
	    //######

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
    
	    xCoord = 5;
	    yCoord = yCoord + (barSpacing);
      
	    for(Enumeration e1 = list.elements(); e1.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e1.nextElement();
		if((globalMappingElement.getMappingName()) != null){
		    
		    //For consistancy in drawing, the y coord is updated at the beginning of the loop.
		    yCoord = yCoord + (barSpacing);
		    
		    //First draw the mapping color box.
		    g2D.setColor(globalMappingElement.getColor());
		    g2D.fillRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
		    
		    if(windowType == 2){
			if((globalMappingElement.getMappingID()) == (trial.getColorChooser().getUserEventHightlightColorID())){
			    g2D.setColor(trial.getColorChooser().getUserEventHightlightColor());
			    g2D.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
			    g2D.drawRect(xCoord + 1, (yCoord - barHeight) + 1, barHeight - 2, barHeight - 2);
			}
			else{
			    g2D.setColor(Color.black);
			    g2D.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
			}
		    }
		    else if(windowType == 1){
			if((globalMappingElement.getMappingID()) == (trial.getColorChooser().getGroupHighlightColorID())){
			    g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
			    g2D.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
			    g2D.drawRect(xCoord + 1, (yCoord - barHeight) + 1, barHeight - 2, barHeight - 2);
			}
			else{
			    g2D.setColor(Color.black);
			    g2D.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
			}
		    }
		    else{
			if((globalMappingElement.getMappingID()) == (trial.getColorChooser().getHighlightColorID())){
			    g2D.setColor(trial.getColorChooser().getHighlightColor());
			    g2D.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
			    g2D.drawRect(xCoord + 1, (yCoord - barHeight) + 1, barHeight - 2, barHeight - 2);
			}
			else{
			    g2D.setColor(Color.black);
			    g2D.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
			}
		    }
		    
		    //Update the xCoord to draw the mapping name.
		    xCoord = xCoord + (barHeight + 10);
		    //Reset the drawing color to the text color ... in this case, black.
		    g2D.setColor(Color.black);
		    
		    //Draw the mapping name.
		    String s = globalMappingElement.getMappingName();
		    
		    g2D.drawString(s, xCoord, yCoord);
		    
		    //Figure out how wide that string was for x coord reasons.
		    int tmpWidth = 5 + barHeight + (fmFont.stringWidth(s));
		    
		    //Figure out how wide that string was for x coord reasons.
		    if(tmpXWidthCalc < tmpWidth){
			tmpXWidthCalc = (tmpWidth + 11);
		    }
		    
		    if(instruction==0)
			globalMappingElement.setDrawCoords(0, tmpWidth, (yCoord - barHeight), yCoord);
		    
		    //Reset the xCoord.
		    xCoord = xCoord - (barHeight + 10);
		}
		
	    }
	    
	    //Resize the panel if needed.
	    if(((yCoord >= yPanelSize) || (tmpXWidthCalc >= xPanelSize)) && instruction==0){
		yPanelSize = yCoord + 1;
		xPanelSize = tmpXWidthCalc + 1;
		
		revalidate();
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MLWP03");
	}
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
		if(arg.equals("Show Function Details")){
		    if(clickedOnObject instanceof GlobalMappingElement){
			GlobalMappingElement globalMapppingElement = (GlobalMappingElement) clickedOnObject;
			//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
			trial.getColorChooser().setHighlightColorID(globalMapppingElement.getMappingID());
			MappingDataWindow tmpRef = new MappingDataWindow(trial, globalMapppingElement.getMappingID(), trial.getStaticMainWindow().getSMWData(), this.debug());
			trial.getSystemEvents().addObserver(tmpRef);
			tmpRef.show();
		    }
		}
		else if(arg.equals("Show User Event Details")){
		    if(clickedOnObject instanceof GlobalMappingElement){
			GlobalMappingElement globalMapppingElement = (GlobalMappingElement) clickedOnObject;
			//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
			trial.getColorChooser().setUserEventHighlightColorID(globalMapppingElement.getMappingID());
			UserEventWindow tmpRef = new UserEventWindow(trial, globalMapppingElement.getMappingID(), trial.getStaticMainWindow().getSMWData(), this.debug());
			trial.getSystemEvents().addObserver(tmpRef);
			tmpRef.show();
		    }
		}
		else if((arg.equals("Change Function Color")) || (arg.equals("Change User Event Color")) ||
			(arg.equals("Change Group Color"))){ 
		    if(clickedOnObject instanceof GlobalMappingElement){
			GlobalMappingElement globalMapppingElement = (GlobalMappingElement) clickedOnObject;
			Color color = globalMapppingElement.getColor();
			JColorChooser tmpJColorChooser = new JColorChooser();
			color = tmpJColorChooser.showDialog(this, "Please select a new color", color);
			if(color != null){
			    globalMapppingElement.setSpecificColor(color);
			    globalMapppingElement.setColorFlag(true);
			    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
			}
		    }
		}
		else if(arg.equals("Reset to Generic Color")){ 
		    if(clickedOnObject instanceof GlobalMappingElement){
			GlobalMappingElement globalMapppingElement = (GlobalMappingElement) clickedOnObject;
			globalMapppingElement.setColorFlag(false);
			trial.getSystemEvents().updateRegisteredObjects("colorEvent");
		    }
		}
		else if(arg.equals("Show This Group Only")){ 
		    if(clickedOnObject instanceof GlobalMappingElement){
			GlobalMappingElement globalMapppingElement = (GlobalMappingElement) clickedOnObject;
			GlobalMapping globalMapping = trial.getGlobalMapping();
			globalMapping.setSelectedGroupID(globalMapppingElement.getMappingID());
			globalMapping.setGroupFilter(1);
			trial.getSystemEvents().updateRegisteredObjects("dataEvent");
		    }
		}
		else if(arg.equals("Show All Groups Except This One")){ 
		    if(clickedOnObject instanceof GlobalMappingElement){
			GlobalMappingElement globalMapppingElement = (GlobalMappingElement) clickedOnObject;
			GlobalMapping globalMapping = trial.getGlobalMapping();
			globalMapping.setSelectedGroupID(globalMapppingElement.getMappingID());
			globalMapping.setGroupFilter(2);
			trial.getSystemEvents().updateRegisteredObjects("dataEvent");
		    }
		}
		else if(arg.equals("Show All Groups")){ 
		    if(clickedOnObject instanceof GlobalMappingElement){
			GlobalMappingElement globalMapppingElement = (GlobalMappingElement) clickedOnObject;
			GlobalMapping globalMapping = trial.getGlobalMapping();
			globalMapping.setSelectedGroupID(-1);
			globalMapping.setGroupFilter(0);
			trial.getSystemEvents().updateRegisteredObjects("dataEvent");
		    }
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MLWP04");
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
	    //Get the location of the mouse.
	    int xCoord = evt.getX();
	    int yCoord = evt.getY();
	    
	    //Get the number of times clicked.
	    int clickCount = evt.getClickCount();
	    
	    //Cycle through the id mapping list.
	    GlobalMappingElement globalMappingElement = null;
	    
	    for(Enumeration e1 = list.elements(); e1.hasMoreElements() ;){
		globalMappingElement = (GlobalMappingElement) e1.nextElement();
		
		if(yCoord <= (globalMappingElement.getYEnd())){
		    if((yCoord >= (globalMappingElement.getYBeg())) && (xCoord >= (globalMappingElement.getXBeg()))
		       && (xCoord <= (globalMappingElement.getXEnd()))){
			if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
			    //Set the clickedSMWMeanDataElement.
			    clickedOnObject = globalMappingElement;
			    popup.show(this, evt.getX(), evt.getY());
			    
			    //Return from this mapping.
			    return;
			}
			else{
			    if(windowType == 2){
				//Want to set the clicked on mapping to the current highlight color or, if the one
				//clicked on is already the current highlighted one, set it back to normal.
				if((trial.getColorChooser().getUserEventHightlightColorID()) == -1){
				    trial.getColorChooser().setUserEventHighlightColorID(globalMappingElement.getMappingID());
				}
				else{
				    if(!((trial.getColorChooser().getUserEventHightlightColorID()) == (globalMappingElement.getMappingID())))
					trial.getColorChooser().setUserEventHighlightColorID(globalMappingElement.getMappingID());
				    else
					trial.getColorChooser().setUserEventHighlightColorID(-1);
				}
			    }
			    else if(windowType == 1){
				//Want to set the clicked on mapping to the current highlight color or, if the one
				//clicked on is already the current highlighted one, set it back to normal.
				if((trial.getColorChooser().getGroupHighlightColorID()) == -1)
				    trial.getColorChooser().setGroupHighlightColorID(globalMappingElement.getMappingID());
				else{
				    if((trial.getColorChooser().getGroupHighlightColorID()) == (globalMappingElement.getMappingID()))
					trial.getColorChooser().setGroupHighlightColorID(-1);
				    else
					trial.getColorChooser().setGroupHighlightColorID(globalMappingElement.getMappingID());
				}
			    }
			    else{
				//Want to set the clicked on mapping to the current highlight color or, if the one
				//clicked on is already the current highlighted one, set it back to normal.
				if((trial.getColorChooser().getHighlightColorID()) == -1)
				    trial.getColorChooser().setHighlightColorID(globalMappingElement.getMappingID());
				else{
				    if((trial.getColorChooser().getHighlightColorID()) == (globalMappingElement.getMappingID()))
					trial.getColorChooser().setHighlightColorID(-1);
				    else
					trial.getColorChooser().setHighlightColorID(globalMappingElement.getMappingID());
				}
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
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "MLWP05");
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
	    return mLWindow.getSize();
    }
    //######
    //End - ParaProfImageInterface
    //######

    //####################################
    //End - Interface code.
    //####################################
  
  
    public Dimension getPreferredSize(){
	return new Dimension((xPanelSize + 10), (yPanelSize + 10));
    }
    
    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //Instance data.
    //####################################
    private int xPanelSize = 300;
    private int yPanelSize = 400;
  
    private int barHeight = -1;
    private int barSpacing = -1;
  
    private ParaProfTrial trial = null;
    private MappingLedgerWindow mLWindow = null;
    private int windowType = -1;
    private Vector list = null;
  
    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################
  
}
