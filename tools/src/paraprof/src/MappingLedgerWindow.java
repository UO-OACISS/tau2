/* 
  MappingLedgerWindow.java

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

public class MappingLedgerWindow extends JFrame implements ActionListener,  MenuListener, Observer{
  
    public MappingLedgerWindow(){
	try{
	    setLocation(new java.awt.Point(300, 200));
	    setSize(new java.awt.Dimension(350, 450));
	    
	    //Set the title indicating that there was a problem.
	    this.setTitle("Wrong constructor used");
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "MLW01");
	}
    }
  
    public MappingLedgerWindow(ParaProfTrial trial, int windowType){
	try{
	    
	    this.trial = trial;
	    this.windowType = windowType;
	    
	    setLocation(new java.awt.Point(300, 200));
	    setSize(new java.awt.Dimension(350, 450));
	    
	    //Now set the title.
	    switch(windowType){
	    case 0:
		this.setTitle("Function Ledger Window: " + trial.getProfilePathName());
		break;
	    case 1:
		this.setTitle("Group Ledger Window: " + trial.getProfilePathName());
		break;
	    case 2:
		this.setTitle("User Event Window: " + trial.getProfilePathName());
		break;
	    default:
		ParaProf.systemError(null, null, "Unexpected window type - MLW02 value: " + windowType);
	    }
	    
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
	    
	    //Windows menu
	    windowsMenu = new JMenu("Windows");
	    windowsMenu.addMenuListener(this);
	    UtilFncs.windowMenuItems(windowsMenu,this);
	    
	    //Help menu.
	    JMenu helpMenu = new JMenu("Help");
	    UtilFncs.helpMenuItems(helpMenu, this);
	    
	    //Now, add all the menus to the main menu.
	    mainMenu.add(fileMenu);
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
	    contentPane = getContentPane();
	    gbl = new GridBagLayout();
	    contentPane.setLayout(gbl);
	    gbc = new GridBagConstraints();
	    gbc.insets = new Insets(5, 5, 5, 5);
	    
	    //######
	    //Panel and ScrollPane definition.
	    //######
	    panel = new MappingLedgerWindowPanel(trial, this, windowType);
	    sp = new JScrollPane(panel);
	    //######
	    //End - Panel and ScrollPane definition.
	    //######
	    
	    gbc.fill = GridBagConstraints.BOTH;
	    gbc.anchor = GridBagConstraints.CENTER;
	    gbc.weightx = 1;
	    gbc.weighty = 1;
	    addCompItem(sp, gbc, 0, 0, 1, 1);
	    //####################################
	    //End - Create and add the components.
	    //####################################
	    
	    trial.getSystemEvents().addObserver(this);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "MLW02");
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
	    if(trial.groupNamesPresent())
		((JMenuItem)windowsMenu.getItem(1)).setEnabled(true);
	    else
		((JMenuItem)windowsMenu.getItem(1)).setEnabled(false);
	    
	    if(trial.userEventsPresent())
		((JMenuItem)windowsMenu.getItem(2)).setEnabled(true);
	    else
		((JMenuItem)windowsMenu.getItem(2)).setEnabled(false);
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
	ParaProf.helpWindow.clearText();
	if(display)
	    ParaProf.helpWindow.show();
	if(windowType == 0){
	    ParaProf.helpWindow.writeText("This is the function ledger window.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("This window shows all the functions tracked in this profile.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("To see more information about any of the mappings shown here,");
	    ParaProf.helpWindow.writeText("right click on that function, and select from the popup menu.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("You can also left click any function to hightlight it in the system.");
	}
	else if(windowType == 1){
	    ParaProf.helpWindow.writeText("This is the group ledger window.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("This window shows all the groups tracked in this profile.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("Left click any group to hightlight it in the system.");
	    ParaProf.helpWindow.writeText("Right click on any group, and select from the popup menu"
					  + " to display more options for masking or displaying functions in a particular group.");
	}
	else{
	    ParaProf.helpWindow.writeText("This is the user event ledger window.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("This window shows all the user events tracked in this profile.");
	    ParaProf.helpWindow.writeText("");
	    ParaProf.helpWindow.writeText("Left click any mapping to hightlight it in the system.");
	    ParaProf.helpWindow.writeText("Right click on any user event, and select from the popup menu.");
	}
    }

    //Updates this window's data copy.
    private void sortLocalData(){ 
	    list = trial.getGlobalMapping().getMapping(windowType);}
    
    public Vector getData(){
	return list;}

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h){
	try{
	    gbc.gridx = x;
	    gbc.gridy = y;
	    gbc.gridwidth = w;
	    gbc.gridheight = h;
	    
	    getContentPane().add(c, gbc);
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "MLW03");
	}
    }

    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e){
	closeThisWindow();}
  
    void closeThisWindow(){ 
	try{
	    if(ParaProf.debugIsOn){
		System.out.println("------------------------");
		System.out.println("A Mapping Ledger Window for window type: " + windowType + " is closing");
		System.out.println("Clearing resourses for that window.");
	    }
	    
	    setVisible(false);
	    trial.getSystemEvents().deleteObserver(this);
	    dispose();
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "TDW10");
	}
    }

    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    private int windowType = -1; //0:function, 1:group, 2:userevent.

    private JMenu windowsMenu = null;
  
    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;
  
    private JScrollPane sp = null;
    private MappingLedgerWindowPanel panel = null;
 
    private Vector list = null;
    //####################################
    //End - Instance data.
    //####################################
}
