package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;

import javax.swing.*;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import edu.uoregon.tau.dms.dss.Function;
import edu.uoregon.tau.dms.dss.Group;
import edu.uoregon.tau.dms.dss.UserEvent;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;

/**
 * LedgerWindow
 * This object represents the ledger window.
 *  
 * <P>CVS $Id: LedgerWindow.java,v 1.19 2005/09/08 22:40:44 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.19 $
 * @see		LedgerDataElement
 * @see		LedgerWindowPanel
 */
public class LedgerWindow extends JFrame implements Observer, ParaProfWindow {

    public static final int FUNCTION_LEDGER = 0;
    public static final int GROUP_LEDGER = 1;
    public static final int USEREVENT_LEDGER = 2;
    public static final int PHASE_LEDGER = 3;
    private int windowType = -1; //0:function, 1:group, 2:userevent, 3:phase

    private ParaProfTrial ppTrial = null;
    private JScrollPane sp = null;
    private LedgerWindowPanel panel = null;
    private Vector list = new Vector();
    
    public void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));
        //mainMenu.add(ParaProfUtils.createTrialMenu(trial, this));
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
    }

    public LedgerWindow(ParaProfTrial ppTrial, int windowType, Component parent) {
        this.ppTrial = ppTrial;
        ppTrial.addObserver(this);

        this.windowType = windowType;

        setSize(new java.awt.Dimension(350, 450));

        //setLocation(new java.awt.Point(300, 200));
        setLocation(WindowPlacer.getNewLedgerLocation(this, parent));
        
        //Now set the title.
        switch (windowType) {
        case FUNCTION_LEDGER:
            this.setTitle("Function Ledger: " + ppTrial.getTrialIdentifier(true));
            break;
        case GROUP_LEDGER:
            this.setTitle("Group Ledger: " + ppTrial.getTrialIdentifier(true));
            break;
        case USEREVENT_LEDGER:
            this.setTitle("User Event Ledger: " + ppTrial.getTrialIdentifier(true));
            break;
        case PHASE_LEDGER:
            this.setTitle("Phase Ledger: " + ppTrial.getTrialIdentifier(true));
            break;
        default:
            throw new ParaProfException("Invalid Ledger Window Type: " + windowType);
        }

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Set the help window text if required.
        if (ParaProf.helpWindow.isVisible()) {
            this.help(false);
        }

       

        //Sort the local data.
        sortLocalData();
        
        getContentPane().setLayout(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //Panel and ScrollPane definition.
        panel = new LedgerWindowPanel(ppTrial, this, windowType);
        sp = new JScrollPane(panel);
        JScrollBar vScrollBar = sp.getVerticalScrollBar();
        vScrollBar.setUnitIncrement(35);

        setupMenus();
        
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        addCompItem(sp, gbc, 0, 0, 1, 1);

        ParaProf.incrementNumWindows();
    }


    public void update(Observable o, Object arg) {
            String tmpString = (String) arg;
            if (tmpString.equals("prefEvent")) {
                panel.repaint();
            } else if (tmpString.equals("colorEvent")) {
                panel.repaint();
            } else if (tmpString.equals("dataEvent")) {
                sortLocalData();
                panel.repaint();
            } else if (tmpString.equals("subWindowCloseEvent")) {
                closeThisWindow();
            }
    }

    public void help(boolean display) {
        ParaProf.helpWindow.clearText();
        if (display)
            ParaProf.helpWindow.show();
        if (windowType == 0) {
            ParaProf.helpWindow.writeText("This is the function ledger window.\n");
            ParaProf.helpWindow.writeText("This window shows all the functions tracked in this profile.\n");
            ParaProf.helpWindow.writeText("To see more information about any of the functions shown here,");
            ParaProf.helpWindow.writeText("right click on that function, and select from the popup menu.\n");
            ParaProf.helpWindow.writeText("You can also left click any function to highlight it in the system.");
        } else if (windowType == 1) {
            ParaProf.helpWindow.writeText("This is the group ledger window.\n");
            ParaProf.helpWindow.writeText("This window shows all the groups tracked in this profile.\n");
            ParaProf.helpWindow.writeText("Left click any group to highlight it in the system.");
            ParaProf.helpWindow.writeText("Right click on any group, and select from the popup menu"
                    + " to display more options for masking or displaying functions in a particular group.");
        } else {
            ParaProf.helpWindow.writeText("This is the user event ledger window.\n");
            ParaProf.helpWindow.writeText("This window shows all the user events tracked in this profile.\n");
            ParaProf.helpWindow.writeText("Left click any user event to highlight it in the system.");
            ParaProf.helpWindow.writeText("Right click on any user event, and select from the popup menu.");
        }
    }

    //Updates this window's data copy.
    private void sortLocalData() {
        list = new Vector();
        if (this.windowType == FUNCTION_LEDGER) {
            for (Iterator it = ppTrial.getDataSource().getFunctions(); it.hasNext();) {
                list.addElement(new LedgerDataElement((Function) it.next()));
            }
        } else if (this.windowType == GROUP_LEDGER) {
            for (Iterator it = ppTrial.getDataSource().getGroups(); it.hasNext();) {
                list.addElement(new LedgerDataElement((Group) it.next()));
            }
        } else if (this.windowType == USEREVENT_LEDGER) {
            for (Iterator it = ppTrial.getDataSource().getUserEvents(); it.hasNext();) {
                list.addElement(new LedgerDataElement((UserEvent) it.next()));
            }
        } else if (this.windowType == PHASE_LEDGER) {
            Group group = ppTrial.getDataSource().getGroup("TAU_PHASE");
            for (Iterator it = ppTrial.getDataSource().getFunctions(); it.hasNext();) {
                Function function = (Function) it.next();
                if (function.isGroupMember(group)) {
                    list.addElement(new LedgerDataElement(function));
                }
            }
        }
    }

    public Vector getData() {
        return list;
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
            gbc.gridx = x;
            gbc.gridy = y;
            gbc.gridwidth = w;
            gbc.gridheight = h;
            getContentPane().add(c, gbc);
    }

    public Rectangle getViewRect() {
        return sp.getViewport().getViewRect();
    }

    //Respond correctly when this window is closed.
    private void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    public void closeThisWindow() {
        try {
            setVisible(false);
            ppTrial.deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

   
}