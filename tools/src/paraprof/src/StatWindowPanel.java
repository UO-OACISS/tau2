/* 
   
StatWindowPanel.java

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
import javax.swing.event.*;
import java.awt.geom.*;
import java.awt.print.*;

public class StatWindowPanel extends JPanel implements ActionListener, MouseListener, Printable{   
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
  
  
    public StatWindowPanel(ParaProfTrial inParaProfTrial, int nodeID, int contextID, int threadID, StatWindow sWindow, int windowType){
	try{
	    setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
	    setBackground(Color.white);

	    trial = inParaProfTrial;
	    this.nodeID = nodeID;
	    this.contextID = contextID;
	    this.threadID = threadID;
	    this.sWindow = sWindow;
	    this.windowType = windowType;
      
	    //Add this object as a mouse listener.
	    addMouseListener(this);
      
	    //Add items to the popu menu.
	    JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
	    mappingDetailsItem.addActionListener(this);
	    popup.add(mappingDetailsItem);
      
	    JMenuItem changeColorItem = new JMenuItem("Change Function Color");
	    changeColorItem.addActionListener(this);
	    popup.add(changeColorItem);
      
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
	    drawPage((Graphics2D) g, false);
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
    
	drawPage(g2, true);
    
	return Printable.PAGE_EXISTS;
    }  

    public void drawPage(Graphics2D g2D, boolean print){
	try{
	    SMWThreadDataElement sMWThreadDataElement = null;
	    Color tmpColor;
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
	    fmMonoFont = g2D.getFontMetrics(MonoFont);
	    maxFontAscent = fmMonoFont.getMaxAscent();
	    maxFontDescent = fmMonoFont.getMaxDescent();
	    g2D.setFont(MonoFont);
      
	    if(spacing <= (maxFontAscent + maxFontDescent)){
		spacing = spacing + 1;
	    }
	    
	    list = sWindow.getData();
	    //With group support present, it is possible that the number of mappings in
	    //our data list is zero.  If so, just return.
	    if((list.size()) == 0)
		return;
   
	    Rectangle clipRect = g2D.getClipBounds();
	    int yBeg = (int) clipRect.getY();
	    int yEnd = (int) (yBeg + clipRect.getHeight());
	    int startThreadElement = 0;
	    int endThreadElement = 0;
      
	    //Draw the heading!
	    switch(windowType){
	    case 0:
		tmpString = GlobalThreadDataElement.getTStatStringHeading(trial);
	    case 1:
		tmpString = GlobalThreadDataElement.getTStatStringHeading(trial);
		break;
	    case 2:
		tmpString = GlobalThreadDataElement.getUserEventStatStringHeading();
		break;
	    default:
		UtilFncs.systemError(null, null, "Unexpected window type - SWP value: " + (windowType));
	    }
	    
	    int tmpInt = tmpString.length();
	    
	    for(int i=0; i<tmpInt; i++){
		dashString = dashString + "-";
	    }
	    
	    g2D.setColor(Color.black);
	    
	    AttributedString dashStringAS = new AttributedString(dashString);
	    dashStringAS.addAttribute(TextAttribute.FONT, MonoFont);
	    
	    //Draw the first dashed string.
	    yCoord = yCoord + spacing;
	    g2D.drawString(dashStringAS.getIterator(), 20, yCoord);
	    yCoord = yCoord + spacing + 10;
	    
	    //Draw the heading.
	    AttributedString headingStringAS = new AttributedString(tmpString);
	    headingStringAS.addAttribute(TextAttribute.FONT, MonoFont);
	    g2D.drawString(headingStringAS.getIterator(), 20, yCoord);
	    yCoord = yCoord + spacing + 10;
	    
	    //Draw the second dashed string.
	    g2D.drawString(dashStringAS.getIterator(), 20, yCoord);
	    
	    startLocation = yCoord;
	    
	    //Set up some panel dimensions.
	    newYPanelSize = yCoord + ((list.size() + 1) * spacing);
	    
	    startThreadElement = ((yBeg - yCoord) / spacing) - 1;
	    endThreadElement  = ((yEnd - yCoord) / spacing) + 1;
	    
	    if((yCoord > yBeg) || (yCoord > yEnd)){
		if(yCoord > yBeg){
		    startThreadElement = 0;
		}
		
		if(yCoord > yEnd){
		    endThreadElement = 0;
		}
	    }
	    
	    if(startThreadElement < 0)
		startThreadElement = 0;
	    
	    if(endThreadElement < 0)
		endThreadElement = 0;
	    
	    if(startThreadElement > (list.size() - 1))
		startThreadElement = (list.size() - 1);
            
	    if(endThreadElement > (list.size() - 1))
		endThreadElement = (list.size() - 1);
	    
	    yCoord = yCoord + (startThreadElement * spacing);
	    
	    for(int i = startThreadElement; i <= endThreadElement; i++){ 
		sMWThreadDataElement = (SMWThreadDataElement) list.elementAt(i);
		switch(windowType){
		case 0:
		    tmpString = sMWThreadDataElement.getMeanTotalStatString(sWindow.units());
		    break;
		case 1:
		    tmpString = sMWThreadDataElement.getTStatString(sWindow.units());
		    break;
		case 2:
		    tmpString = sMWThreadDataElement.getUserEventStatString();
		    break;
		default:
		    UtilFncs.systemError(null, null, "Unexpected window type - SWP value: " + (windowType));
		}

		yCoord = yCoord + spacing;
		
		g2D.setColor(Color.black);
		
		AttributedString as = new AttributedString(tmpString);
		as.addAttribute(TextAttribute.FONT, MonoFont);
		
		if((sMWThreadDataElement.getMappingID()) == (trial.getColorChooser().getHighlightColorMappingID()))
		    as.addAttribute(TextAttribute.FOREGROUND, 
				    (trial.getColorChooser().getHighlightColor()),
				    GlobalThreadDataElement.getPositionOfName(), tmpString.length());
		else if((sMWThreadDataElement.isGroupMember(trial.getColorChooser().getGHCMID())))
		    as.addAttribute(TextAttribute.FOREGROUND, 
				    (trial.getColorChooser().getGroupHighlightColor()),
				    GlobalThreadDataElement.getPositionOfName(), tmpString.length());
		else
		    as.addAttribute(TextAttribute.FOREGROUND, 
				    (sMWThreadDataElement.getMappingColor()),
				    GlobalThreadDataElement.getPositionOfName(), tmpString.length());
		
		g2D.drawString(as.getIterator(), 20, yCoord);
		
		//Figure out how wide that string was for x coord reasons.
		if(tmpXWidthCalc < 2*fmMonoFont.stringWidth(tmpString)){
		    tmpXWidthCalc = (20 + 2*fmMonoFont.stringWidth(tmpString));
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
			trial.getColorChooser().setHighlightColorMappingID(sMWThreadDataElement.getMappingID());
			MappingDataWindow tmpRef = new MappingDataWindow(trial, sMWThreadDataElement.getMappingID(), trial.getStaticMainWindow().getSMWData());
			trial.getSystemEvents().addObserver(tmpRef);
			tmpRef.show();
		    }
		}
		else if(arg.equals("Change Function Color")){ 
		    int mappingID = -1;
		    
		    //Get the clicked on object.
		    if(clickedOnObject instanceof SMWThreadDataElement)
			mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
		    
		    GlobalMapping globalMappingReference = trial.getGlobalMapping();
		    GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
		    
		    Color tmpCol = tmpGME.getMappingColor();
		    
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
		    
		    //Get the clicked on object.
		    if(clickedOnObject instanceof SMWThreadDataElement)
			mappingID = ((SMWThreadDataElement) clickedOnObject).getMappingID();
		    
		    GlobalMapping globalMappingReference = trial.getGlobalMapping();
		    GlobalMappingElement tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(mappingID, 0);
		    
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
		    
		    if(fmMonoFont != null){
			switch(windowType){
			case 0:
			    tmpString = sMWThreadDataElement.getMeanTotalStatString(sWindow.units());
			    break;
			case 1:
			    tmpString = sMWThreadDataElement.getTStatString(sWindow.units());
			    break;
			case 2:
			    tmpString = sMWThreadDataElement.getUserEventStatString();
			    break;
			default:
			    UtilFncs.systemError(null, null, "Unexpected window type - SWP value: " + (windowType));
			}
			
			int stringWidth = fmMonoFont.stringWidth(tmpString) + 20;
			
			if(xCoord <= stringWidth){
			    if((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0){
				//Set the clickedSMWDataElement.
				clickedOnObject = sMWThreadDataElement;
				popup.show(this, evt.getX(), evt.getY());
			    }
			    else{
				//Want to set the clicked on mapping to the current highlight color or, if the one
				//clicked on is already the current highlighted one, set it back to normal.
				if((trial.getColorChooser().getHighlightColorMappingID()) == -1){
				    trial.getColorChooser().setHighlightColorMappingID(sMWThreadDataElement.getMappingID());
				}
				else{
				    if(!((trial.getColorChooser().getHighlightColorMappingID()) == (sMWThreadDataElement.getMappingID())))
					trial.getColorChooser().setHighlightColorMappingID(sMWThreadDataElement.getMappingID());
				    else
					trial.getColorChooser().setHighlightColorMappingID(-1);
				}
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

    //####################################
    //End - Interface code.
    //####################################
    
    public Dimension getPreferredSize(){
	return new Dimension(xPanelSize, (yPanelSize + 10));
    }
    
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

    private Font MonoFont = null;
    private FontMetrics fmMonoFont = null;
  
    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;
    //####################################
    //End - Instance data.
    //####################################
}
