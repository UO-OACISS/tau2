
/* 
   
StatWindowPanel.java

Title:      ParaProf
Author:     Robert Bell
Description:  
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.text.*;
import java.awt.font.*;
import java.awt.font.TextAttribute;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.geom.*;
import java.text.*;
import java.awt.font.*;
import java.awt.print.*;
import edu.uoregon.tau.dms.dss.*;

public class StatWindowPanel extends JPanel implements ActionListener, MouseListener, Printable, ParaProfImageInterface{
    public StatWindowPanel(){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    
	    //Schedule a repaint of this panel.
	    this.repaint();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SWP01");
	}
  
    }
  
  
    public StatWindowPanel(ParaProfTrial inParaProfTrial, int nodeID,
			   int contextID, int threadID, StatWindow sWindow,
			   int windowType, boolean debug){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);

	    trial = inParaProfTrial;
	    this.nodeID = nodeID;
	    this.contextID = contextID;
	    this.threadID = threadID;
	    this.sWindow = sWindow;
	    this.windowType = windowType;
	    this.debug = debug;
      
	    //Add this object as a mouse listener.
	    addMouseListener(this);
      
	    //Add items to the popu menu.
	    if(windowType == 2){
		JMenuItem mappingDetailsItem = new JMenuItem("Show Userevent Details");
		mappingDetailsItem.addActionListener(this);
		popup.add(mappingDetailsItem);
		
		JMenuItem changeColorItem = new JMenuItem("Change Userevent Color");
		changeColorItem.addActionListener(this);
		popup.add(changeColorItem);
	    }
	    else{
		JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
		mappingDetailsItem.addActionListener(this);
		popup.add(mappingDetailsItem);
		
		JMenuItem changeColorItem = new JMenuItem("Change Function Color");
		changeColorItem.addActionListener(this);
		popup.add(changeColorItem);
	    }
      
	    JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
	    maskMappingItem.addActionListener(this);
	    popup.add(maskMappingItem);
      
	    this.repaint();
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SWP02");
	}
  
    }

    public void paintComponent(Graphics g){
	try{
	    super.paintComponent(g);
	    renderIt((Graphics2D) g, 0, false);
	}
	catch(Exception e){
	    System.out.println(e);
	    UtilFncs.systemError(e, null, "SWP03");
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

    public void renderIt(Graphics2D g2D, int instruction, boolean header){ //Got to here!!!!!
	try{
	    if(this.debug()){
		System.out.println("####################################");
		System.out.println("StatWindowPanel.renderIt(...)");
		System.out.println("####################################");
	    }

	    list = sWindow.getData();

	    //With group support present, it is possible that the number of mappings in
	    //our data list is zero.  If so, just return.
	    if((list.size()) == 0)
		return;

	    //######
	    //Some declarations.
	    //######
	    SMWThreadDataElement sMWThreadDataElement = null;
	    Color tmpColor;
	    int yCoord = 0;
	    String tmpString = null;
	    String dashString = "";
	    int tmpXWidthCalc = 0;
	    //######
	    //Some declarations.
	    //######
	    
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
	    FontRenderContext frc = g2D.getFontRenderContext();
      
	    if(spacing <= (maxFontAscent + maxFontDescent)){
		spacing = spacing + 1;
	    }

	    //######
	    //Draw the header if required.
	    //######
	    if(header){
		//FontRenderContext frc2 = g2D.getFontRenderContext();
		Insets insets = this.getInsets();
		yCoord = yCoord + (spacing);
		String headerString = sWindow.getHeaderString();
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

	    switch(windowType){
	    case 0:
		if(trial.isTimeMetric())
		    tmpString = GlobalThreadDataElement.getTStatStringHeading("Time");
		else
		    tmpString = GlobalThreadDataElement.getTStatStringHeading("Counts");
		break;
	    case 1:
		if(trial.isTimeMetric())
		    tmpString = GlobalThreadDataElement.getTStatStringHeading("Time");
		else
		    tmpString = GlobalThreadDataElement.getTStatStringHeading("Counts");
		break;
	    case 2:
		tmpString = GlobalThreadDataElement.getUserEventStatStringHeading();
		break;
	    default:
		UtilFncs.systemError(null, null, "Unexpected window type - SWP value: " + (windowType));
	    }
	    
	    //Calculate the name position.
	    int namePosition = fmMonoFont.stringWidth(tmpString)+20; //Note that 20 is the begin draw position.

	    //Now append "name" to the end of the string.
	    tmpString = tmpString+"name";
	    int tmpInt = tmpString.length();
	    
	    for(int i=0; i<tmpInt; i++){
		dashString = dashString + "-";
	    }
	    
	    g2D.setColor(Color.black);
	    
	    //Draw the first dashed string.
	    yCoord = yCoord + spacing;
	    g2D.drawString(dashString, 20, yCoord);
	    yCoord = yCoord + spacing + 10;

	    //Draw the heading.
	    g2D.drawString(tmpString, 20, yCoord);
	    yCoord = yCoord + spacing + 10;
	    
	    //Draw the second dashed string.
	    g2D.drawString(dashString, 20, yCoord);
	    
	    if(instruction==0)
		startLocation = yCoord;
	    
	    //Set up some panel dimensions.
	    newYPanelSize = yCoord + ((list.size() + 1) * spacing);
	    

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
		    viewRect = sWindow.getViewRect();
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
		
		if(startElement > (list.size() - 1))
		    startElement = (list.size() - 1);
		
		if(endElement > (list.size() - 1))
		    endElement = (list.size() - 1);
		
		if(instruction==0)
		    yCoord = yCoord + (startElement * spacing);
	    }
	    else if(instruction==2 || instruction==3){
		startElement = 0;
		endElement = ((list.size()) - 1);
	    }

	    for(int i = startElement; i <= endElement; i++){ 
		sMWThreadDataElement = (SMWThreadDataElement) list.elementAt(i);
		switch(windowType){
		case 0:
		    tmpString = sMWThreadDataElement.getMeanTotalStatString(sWindow.units());
		    break;
		case 1:
		    tmpString = sMWThreadDataElement.getTStatString(sWindow.units());
		    break;
		case 2:
		    tmpString = sMWThreadDataElement.getUserEventStatString(ParaProf.defaultNumberPrecision);
		    break;
		default:
		    UtilFncs.systemError(null, null, "Unexpected window type - SWP value: " + (windowType));
		}

		yCoord = yCoord + spacing;
		
		g2D.setColor(Color.black);
		int highLightColor = -1;
		if(windowType==2)
		   highLightColor  = trial.getColorChooser().getUserEventHightlightColorID();
		else
		   highLightColor  = trial.getColorChooser().getHighlightColorID();
		
		if((sMWThreadDataElement.getMappingID()) == highLightColor){
		    g2D.setColor(trial.getColorChooser().getHighlightColor());
		    (new TextLayout(tmpString, monoFont, frc)).draw(g2D, 20, yCoord);
		    //g2D.drawString(tmpString, 20, yCoord);
		    g2D.setColor(sMWThreadDataElement.getColor());
		    (new TextLayout(sMWThreadDataElement.getMappingName(), monoFont, frc)).draw(g2D, namePosition, yCoord);
		    //g2D.drawString(sMWThreadDataElement.getMappingName(), namePosition, yCoord);
		}
		else if((windowType!=2)&&(sMWThreadDataElement.isGroupMember(trial.getColorChooser().getGroupHighlightColorID()))){
		    g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
		    (new TextLayout(tmpString, monoFont, frc)).draw(g2D, 20, yCoord);
		    //g2D.drawString(tmpString, 20, yCoord);
		    g2D.setColor(sMWThreadDataElement.getColor());
		    (new TextLayout(sMWThreadDataElement.getMappingName(), monoFont, frc)).draw(g2D, namePosition, yCoord);
		    //g2D.drawString(sMWThreadDataElement.getMappingName(), namePosition, yCoord);
		}
		else{
		    (new TextLayout(tmpString, monoFont, frc)).draw(g2D, 20, yCoord);
		    //g2D.drawString(tmpString, 20, yCoord);
		    g2D.setColor(sMWThreadDataElement.getColor());
		    (new TextLayout(sMWThreadDataElement.getMappingName(), monoFont, frc)).draw(g2D, namePosition, yCoord);
		    //g2D.drawString(sMWThreadDataElement.getMappingName(), namePosition, yCoord);
		}
		
		
		//Figure out how wide that string was for x coord reasons.
		if (tmpXWidthCalc < (20 + namePosition + fmMonoFont.stringWidth(sMWThreadDataElement.getMappingName()))) {
		    tmpXWidthCalc = (20 + namePosition + fmMonoFont.stringWidth(sMWThreadDataElement.getMappingName()));
		}
	    }
	    
	    //Resize the panel if needed.
	    if((newYPanelSize >= yPanelSize) || (tmpXWidthCalc  >= xPanelSize)){
		yPanelSize = newYPanelSize + 1;
		xPanelSize = tmpXWidthCalc + 1;
		revalidate();
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TSWP03");
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
	    
	    SMWThreadDataElement sMWThreadDataElement = null;
	    
	    if(EventSrc instanceof JMenuItem){
		String arg = evt.getActionCommand();
		if(arg.equals("Show Function Details")){
		    
		    if(clickedOnObject instanceof SMWThreadDataElement){
			sMWThreadDataElement = (SMWThreadDataElement) clickedOnObject;
			//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
			trial.getColorChooser().setHighlightColorID(sMWThreadDataElement.getMappingID());
			MappingDataWindow tmpRef = new MappingDataWindow(trial, sMWThreadDataElement.getMappingID(), trial.getStaticMainWindow().getSMWData(), this.debug());
			trial.getSystemEvents().addObserver(tmpRef);
			tmpRef.show();
		    }
		}
		if(arg.equals("Show Userevent Details")){
		    
		    if(clickedOnObject instanceof SMWThreadDataElement){
			sMWThreadDataElement = (SMWThreadDataElement) clickedOnObject;
			//Bring up an expanded data window for this mapping, and set this mapping as highlighted.
			trial.getColorChooser().setUserEventHighlightColorID(sMWThreadDataElement.getMappingID());
			UserEventWindow tmpRef = new UserEventWindow(trial, sMWThreadDataElement.getMappingID(), trial.getStaticMainWindow().getSMWData(), this.debug());
			trial.getSystemEvents().addObserver(tmpRef);
			tmpRef.show();
		    }
		}
		else if(arg.equals("Change Function Color")){ 
		    int mappingID = -1;
		    GlobalMappingElement tmpGME = null;
		    //Get the clicked on object.
		    if(clickedOnObject instanceof SMWThreadDataElement)
			mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
		    
		    GlobalMapping globalMappingReference = trial.getGlobalMapping();
		    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
		     Color tmpCol = tmpGME.getColor();
		     JColorChooser tmpJColorChooser = new JColorChooser();
		     tmpCol = tmpJColorChooser.showDialog(this, "Please select a new color", tmpCol);
		     if(tmpCol != null){
			 tmpGME.setSpecificColor(tmpCol);
			 tmpGME.setColorFlag(true);
			 
			 trial.getSystemEvents().updateRegisteredObjects("colorEvent");
		     }
		}
		else if(arg.equals("Change Userevent Color")){ 
		    int mappingID = -1;
		    GlobalMappingElement tmpGME = null;
		    //Get the clicked on object.
		    if(clickedOnObject instanceof SMWThreadDataElement)
			mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
		    
		    GlobalMapping globalMappingReference = trial.getGlobalMapping();
		    tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
		    Color tmpCol = tmpGME.getColor();
		    JColorChooser tmpJColorChooser = new JColorChooser();
		    tmpCol = tmpJColorChooser.showDialog(this, "Please select a new color", tmpCol);
		    if(tmpCol != null){
			tmpGME.setSpecificColor(tmpCol);
			tmpGME.setColorFlag(true);
			
			trial.getSystemEvents().updateRegisteredObjects("colorEvent");
		    }
		}
		else if(arg.equals("Reset to Generic Color")){ 
		    int mappingID = -1;
		    GlobalMappingElement tmpGME = null;
		    
		    //Get the clicked on object.
		    if(clickedOnObject instanceof SMWThreadDataElement)
			mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
		    
		    GlobalMapping globalMappingReference = trial.getGlobalMapping();
		    if(windowType==2)
			tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 2);
		    else
			tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
		    
		    tmpGME.setColorFlag(false);
		    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
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
    //MouseListener.
    //######
    public void mouseClicked(MouseEvent evt){
	try{
	    SMWThreadDataElement sMWThreadDataElement = null;
	    String tmpString = null;

	    //Get the location of the mouse.
	    //Get the location of the mouse.
	    int xCoord = evt.getX();
	    int yCoord = evt.getY();

	    int fontSize = trial.getPreferences().getBarHeight();
	    
	    //Get the number of times clicked.
	    int clickCount = evt.getClickCount();
	    
	    int tmpInt1 = yCoord - startLocation;
	    int tmpInt2 = tmpInt1 / spacing;
	    int tmpInt3 = (tmpInt2 + 1) * spacing;
	    int tmpInt4 = tmpInt3 - maxFontAscent;

	    if((tmpInt1 >= tmpInt4) && (tmpInt1 <= tmpInt3)){
		if(tmpInt2 < (list.size())){
		    sMWThreadDataElement = (SMWThreadDataElement) list.elementAt(tmpInt2);
		    
		    if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
			//Set the clickedSMWDataElement.
			clickedOnObject = sMWThreadDataElement;
			popup.show(this, evt.getX(), evt.getY());
		    }
		    else{
			//Want to set the clicked on mapping to the current highlight color or, if the one
			//clicked on is already the current highlighted one, set it back to normal.
			if(windowType==2){
			    if((trial.getColorChooser().getUserEventHightlightColorID()) == -1){
				trial.getColorChooser().setUserEventHighlightColorID(sMWThreadDataElement.getMappingID());
			    }
			    else{
				if(!((trial.getColorChooser().getHighlightColorID()) == (sMWThreadDataElement.getMappingID())))
				    trial.getColorChooser().setUserEventHighlightColorID(sMWThreadDataElement.getMappingID());
				else
				    trial.getColorChooser().setUserEventHighlightColorID(-1);
			    }
			}
			else{
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
		    }
		}
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "TSWP05");
	}
    }
    
    public void mousePressed(MouseEvent evt){}
    public void mouseReleased(MouseEvent evt){}
    public void mouseEntered(MouseEvent evt){}
    public void mouseExited(MouseEvent evt){}
    //######
    //End - MouseListener.
    //######

    //######
    //ParaProfImageInterface
    //######
    public Dimension getImageSize(boolean fullScreen, boolean header){
	Dimension d = null;
	if(fullScreen)
	    d = this.getSize();
	else
	    d = sWindow.getSize();
	d.setSize(d.getWidth(),d.getHeight()+lastHeaderEndPosition);
	return d;
    }
    //######
    //End - ParaProfImageInterface
    //######

    //####################################
    //End - Interface code.
    //####################################
    
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
    private int xPanelSize = 800;
    private int yPanelSize = 600;
    private int newXPanelSize = 0;
    private int newYPanelSize = 0;
    
    
    //Some drawing details.
    private int startLocation = 0;
    private int maxFontAscent = 0;
    private int maxFontDescent = 0;
    private int spacing = 0;
  
    private ParaProfTrial trial = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private StatWindow sWindow = null;
    private int windowType = -1;
    private Vector list = null;

    private Font monoFont = null;
    private FontMetrics fmMonoFont = null;
  
    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private int lastHeaderEndPosition = 0;

    private boolean debug = false; //Off by default.private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################
}
