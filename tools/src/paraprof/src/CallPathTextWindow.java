/* 
   CallPathTextWindow.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import java.awt.print.*;

public class CallPathTextWindow extends JFrame implements ActionListener, MenuListener, Observer{

    public CallPathTextWindow(){
	try{
	    setLocation(new java.awt.Point(0, 0));
	    setSize(new java.awt.Dimension(800, 600));
      
	    //Set the title indicating that there was a problem.
	    this.setTitle("Wrong constructor used!");
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW01");
	}
    }
  
    public CallPathTextWindow(ParaProfTrial trial, int nodeID, int contextID, int threadID, 
			      StaticMainWindowData sMWData, 
			      boolean global){
	try{
	    this.trial = trial;
	    this.nodeID = nodeID;
	    this.contextID = contextID;
	    this.threadID = threadID;
	    this.sMWData = sMWData;
	    this.global = global;
      
	    setLocation(new java.awt.Point(0, 0));
	    setSize(new java.awt.Dimension(800, 600));
      
	    //Now set the title.
	    if(global)
		this.setTitle("Call Path Data Relations - " + trial.getTrialIdentifier(true));
	    else
		this.setTitle("Call Path Data " + "n,c,t, " + nodeID + "," + contextID + "," + threadID + " - " + trial.getTrialIdentifier(true));
	    
	    //Add some window listener code
	    addWindowListener(new java.awt.event.WindowAdapter() {
		    public void windowClosing(java.awt.event.WindowEvent evt) {
			thisWindowClosing(evt);
		    }
		});
	    
	    //Set the help window text if required.
	    if(ParaProf.helpWindow.isVisible()){
		this.help(false);
	    }

	    //Sort the local data.
	    sortLocalData();

	    //####################################
	    //Code to generate the menus.
	    //####################################
	    JMenuBar mainMenu = new JMenuBar();

	    //File menu.
	    JMenu fileMenu = new JMenu("File");
	    UtilFncs.fileMenuItems(fileMenu, this);

	    //Options menu.
	    optionsMenu = new JMenu("Options");
	    optionsMenu.addMenuListener(this);
	    UtilFncs.optionMenuItems(optionsMenu,this);

	    //Windows menu
	    windowsMenu = new JMenu("Windows");
	    windowsMenu.addMenuListener(this);
	    UtilFncs.windowMenuItems(windowsMenu,this);

	    //Help menu.
	    JMenu helpMenu = new JMenu("Help");
	    UtilFncs.helpMenuItems(helpMenu, this);
	    
	    //Now, add all the menus to the main menu.
	    mainMenu.add(fileMenu);
	    mainMenu.add(optionsMenu);
	    mainMenu.add(windowsMenu);
	    mainMenu.add(helpMenu);
	    
	    setJMenuBar(mainMenu);
	    //####################################
	    //End - Code to generate the menus.
	    //####################################

      	    //####################################
	    //Create and add the components.
	    //####################################
	    //Setting up the layout system for the main window.
	    Container contentPane = getContentPane();
	    GridBagLayout gbl = new GridBagLayout();
	    contentPane.setLayout(gbl);
	    GridBagConstraints gbc = new GridBagConstraints();
	    gbc.insets = new Insets(5, 5, 5, 5);
      
	    //######
	    //Panel and ScrollPane definition.
	    //######
	    panel = new CallPathTextWindowPanel(trial, nodeID, contextID, threadID, this, global);
	    //The scroll panes into which the list shall be placed.
	    JScrollPane sp = new JScrollPane(panel);
	    JScrollBar vScollBar = sp.getVerticalScrollBar();
	    vScollBar.setUnitIncrement(35);
	    sp.setPreferredSize(new Dimension(500, 450));
	    //######
	    //End - Panel and ScrollPane definition.
	    //######
      
	    //Now add the componants to the main screen.
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 1;
	    gbc.weighty = 1;
	    addCompItem(sp, gbc, 0, 0, 1, 1);
	    //####################################
	    //End - Create and add the components.
	    //####################################
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW02");
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
		if(arg.equals("Print")){
		    PrinterJob job = PrinterJob.getPrinterJob();
		    PageFormat defaultFormat = job.defaultPage();
		    PageFormat selectedFormat = job.pageDialog(defaultFormat);
		    job.setPrintable(panel, selectedFormat);
		    if(job.printDialog()){
			job.print();
		    }
		}
		else if(arg.equals("Edit ParaProf Preferences!")){
		    trial.getPreferences().showPreferencesWindow();
		}
		else if(arg.equals("Save Image")){
		    ParaProfImageOutput imageOutput = new ParaProfImageOutput();
		    imageOutput.saveImage((ParaProfImageInterface) panel);
		}
		else if(arg.equals("Close This Window")){
		    closeThisWindow();
		}
		else if(arg.equals("Exit ParaProf!")){
		    setVisible(false);
		    dispose();
		    System.exit(0);
		}
		else if(arg.equals("name")){
		    if(((JCheckBoxMenuItem)optionsMenu.getItem(0)).isSelected())
			name = true;
		    else
			name = false;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Decending Order")){
		    if(((JCheckBoxMenuItem)optionsMenu.getItem(1)).isSelected())
			order = 0;
		    else
			order = 1;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Show Values as Percent")){
		    if(((JCheckBoxMenuItem)optionsMenu.getItem(2)).isSelected())
			percent = true;
		    else
			percent = false;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Exclusive")){
		    valueType = 2;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Inclusive")){
		    valueType = 4;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Number of Calls")){
		    valueType = 6;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Number of Subroutines")){
		    valueType = 8;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Per Call Value")){
		    valueType = 10;
		    sortLocalData();
		    panel.repaint();
		}
		else if(arg.equals("Microseconds")){
		    units = 0;
		    panel.repaint();
		}
		else if(arg.equals("Milliseconds")){
		    units = 1;
		    panel.repaint();
		}
		else if(arg.equals("Seconds")){
		    units = 2;
		    panel.repaint();
		}
		else if(arg.equals("hr:min:sec")){
		    units = 3;
		    panel.repaint();
		}
		else if(arg.equals("Show Function Ledger")){
		    (new MappingLedgerWindow(trial, 0)).show();
		}
		else if(arg.equals("Show Group Ledger")){
		    (new MappingLedgerWindow(trial, 1)).show();
		}
		else if(arg.equals("Show User Event Ledger")){
		    (new MappingLedgerWindow(trial, 2)).show();
		}
		else if(arg.equals("Close All Sub-Windows")){
		    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
		}
		else if(arg.equals("About ParaProf")){
		    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
		}
		else if(arg.equals("Show Help Window")){
		    this.help(true);
		}
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDW03");
	}
    }
    //######
    //End - ActionListener
    //######

    //######
    //MenuListener.
    //######
   public void menuSelected(MenuEvent evt){
	try{
	    if(valueType > 4){
		((JCheckBoxMenuItem)optionsMenu.getItem(2)).setEnabled(false);
		((JMenu)optionsMenu.getItem(3)).setEnabled(false);}
	    else if(percent){
		((JCheckBoxMenuItem)optionsMenu.getItem(2)).setEnabled(true);
		((JMenu)optionsMenu.getItem(3)).setEnabled(false);}
	    else if(trial.isTimeMetric()){
		((JCheckBoxMenuItem)optionsMenu.getItem(2)).setEnabled(true);
		((JMenu)optionsMenu.getItem(3)).setEnabled(true);}
	    else{
		((JCheckBoxMenuItem)optionsMenu.getItem(2)).setEnabled(true);
		((JMenuItem)optionsMenu.getItem(3)).setEnabled(false);}
	    
	    if(trial.groupNamesPresent())
		((JMenuItem)windowsMenu.getItem(1)).setEnabled(true);
	    else
		((JMenuItem)windowsMenu.getItem(1)).setEnabled(false);
	    
	    if(trial.userEventsPresent())
		((JMenuItem)windowsMenu.getItem(2)).setEnabled(true);
	    else
		((JMenuItem)windowsMenu.getItem(1)).setEnabled(false);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDW04");
	}
    }
    
    public void menuDeselected(MenuEvent evt){}
    public void menuCanceled(MenuEvent evt){}
    //######
    //End - MenuListener.
    //######

        //######
    //Observer.
    //######
    public void update(Observable o, Object arg){
	try{
	    String tmpString = (String) arg;
	    if(tmpString.equals("prefEvent")){
		panel.repaint();
	    }
	    else if(tmpString.equals("colorEvent")){
		panel.repaint();
	    }
	    else if(tmpString.equals("dataEvent")){
		sortLocalData();
		if(!(trial.isTimeMetric()))
		    units = 0;
		panel.repaint();
	    }
	    else if(tmpString.equals("subWindowCloseEvent")){
		closeThisWindow();
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDW05");
	}
    }
    //######
    //End - Observer.
    //######

    //####################################
    //End - Interface code.
    //####################################

    private void help(boolean display){
	//Show the ParaProf help window.
	ParaProf.helpWindow.clearText();
	if(display)
	    ParaProf.helpWindow.show();
	ParaProf.helpWindow.writeText("Call path text window.");
	ParaProf.helpWindow.writeText("");
	ParaProf.helpWindow.writeText("This window displays call path relationships in two ways:");
	ParaProf.helpWindow.writeText("1- If this window has been invoked from the \"windows\" menu of");
	ParaProf.helpWindow.writeText("ParaProf, the information displayed is all call path relations found.");
	ParaProf.helpWindow.writeText("That is, all the parent/child relationships.");
	ParaProf.helpWindow.writeText("Thus, in this case, given the parallel nature of ParaProf, this information");
	ParaProf.helpWindow.writeText("might not be valid for a particular thread. It is however useful to observe");
	ParaProf.helpWindow.writeText("all the realtionships that exist in the data.");
	ParaProf.helpWindow.writeText("");
	ParaProf.helpWindow.writeText("2- If this window has been invoked from the popup menu to the left of a thread bar");
	ParaProf.helpWindow.writeText("in the main ParaProf window, the information dispayed will be specific to this thread,");
	ParaProf.helpWindow.writeText("and will thus contain both parent/child relations and the data relating to those");
	ParaProf.helpWindow.writeText("relationships.");
    }

    //Updates this window's data copy.
    private void sortLocalData(){ 
	try{
	    if(global){
		list = (trial.getGlobalMapping()).getMapping(0);
	    }
	    else{
		//The name selection behaves slightly differently. Thus the check for it.
		if(name){
		    list = sMWData.getThreadData(nodeID, contextID, threadID, windowType, order);
		}
		else{
		    list = sMWData.getThreadData(nodeID, contextID, threadID, windowType, valueType+order);
		}
	    }
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDW06");
	}
    }

    public Vector getData(){
	return list;}

    public ListIterator getDataIterator(){
	return new ParaProfIterator(this.getData());
    }

    public int getWindowType(){
	return windowType;}

    public boolean isPercent(){
	return percent;}
    
    public int getValueType(){
	return valueType;}
    
    public int units(){
	return units;}

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	try{
	    gbc.gridx = x;
	    gbc.gridy = y;
	    gbc.gridwidth = w;
	    gbc.gridheight = h;
	    
	    getContentPane().add(c, gbc);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW05");
	}
    }
  
    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e){
	closeThisWindow();
    }
  
    void closeThisWindow(){ 
	try{
	    if(ParaProf.debugIsOn){
		System.out.println("------------------------");
		System.out.println("A total stat window for: \"" + "n,c,t, " + nodeID + "," + contextID + "," + threadID + "\" is closing");
		System.out.println("Clearing resourses for this window.");
	    }
      
	    setVisible(false);
	    trial.getSystemEvents().deleteObserver(this);
	    dispose();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "CPTW06");
	}
    }
  
    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private CallPathTextWindowPanel panel;
    private StaticMainWindowData sMWData;
    private boolean global = false;
    private int windowType = 1; //0: mean data,1: function data.
                                //Note that in this window, windowType
                                //will always be 1.

    private JMenu optionsMenu = null;
    private JMenu windowsMenu = null;

    private Vector list = null;

    private boolean name = false; //true: sort by name,false: sort by value.
    private int order = 0; //0: descending order,1: ascending order.
    private boolean percent = true; //true: show values as percent,false: show actual values.
    private int valueType = 2; //2-exclusive,4-inclusive,6-number of calls,8-number of subroutines,10-per call value.
    private int units = 0; //0-microseconds,1-milliseconds,2-seconds.
    //####################################
    //Instance data.
    //####################################
}
