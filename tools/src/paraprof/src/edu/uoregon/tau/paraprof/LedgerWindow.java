package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.print.*;
import edu.uoregon.tau.dms.dss.*;

/**
 * LedgerWindow
 * This object represents the ledger window.
 *  
 * <P>CVS $Id: LedgerWindow.java,v 1.10 2005/01/31 23:11:08 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.10 $
 * @see		LedgerDataElement
 * @see		LedgerWindowPanel
 */
public class LedgerWindow extends JFrame implements ActionListener, MenuListener, Observer {

    public static final int FUNCTION_LEDGER = 0;
    public static final int GROUP_LEDGER = 1;
    public static final int USEREVENT_LEDGER = 2;
    private int windowType = -1; //0:function, 1:group, 2:userevent.

    private ParaProfTrial trial = null;
    private JMenu windowsMenu = null;
    private JScrollPane sp = null;
    private LedgerWindowPanel panel = null;
    private Vector list = null;
    
    public void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        JMenu subMenu = null;
        JMenuItem menuItem = null;

        //File menu.
        JMenu fileMenu = new JMenu("File");

        //Save menu.
        subMenu = new JMenu("Save ...");

        menuItem = new JMenuItem("Save Image");
        menuItem.addActionListener(this);
        subMenu.add(menuItem);

        fileMenu.add(subMenu);
        //End - Save menu.

        menuItem = new JMenuItem("Preferences...");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Print");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Close This Window");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Exit ParaProf!");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        fileMenu.addMenuListener(this);

        //Windows menu
        windowsMenu = new JMenu("Windows");

        menuItem = new JMenuItem("Show ParaProf Manager");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Function Ledger");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Group Ledger");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show User Event Ledger");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Call Path Relations");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Close All Sub-Windows");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        windowsMenu.addMenuListener(this);

        //Help menu.
        JMenu helpMenu = new JMenu("Help");

        menuItem = new JMenuItem("Show Help Window");
        menuItem.addActionListener(this);
        helpMenu.add(menuItem);

        menuItem = new JMenuItem("About ParaProf");
        menuItem.addActionListener(this);
        helpMenu.add(menuItem);

        helpMenu.addMenuListener(this);

        //Now, add all the menus to the main menu.
        mainMenu.add(fileMenu);
        mainMenu.add(windowsMenu);
        mainMenu.add(helpMenu);

        setJMenuBar(mainMenu);
    }

    public LedgerWindow(ParaProfTrial trial, int windowType) {
        this.trial = trial;
        this.windowType = windowType;

        setLocation(new java.awt.Point(300, 200));
        setSize(new java.awt.Dimension(350, 450));

        //Now set the title.
        switch (windowType) {
        case FUNCTION_LEDGER:
            this.setTitle("Function Ledger Window: " + trial.getTrialIdentifier(true));
            break;
        case GROUP_LEDGER:
            this.setTitle("Group Ledger Window: " + trial.getTrialIdentifier(true));
            break;
        case USEREVENT_LEDGER:
            this.setTitle("User Event Window: " + trial.getTrialIdentifier(true));
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

        setupMenus();

        //Sort the local data.
        sortLocalData();
        
        getContentPane().setLayout(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //Panel and ScrollPane definition.
        panel = new LedgerWindowPanel(trial, this, windowType);
        sp = new JScrollPane(panel);
        JScrollBar vScollBar = sp.getVerticalScrollBar();
        vScollBar.setUnitIncrement(35);


        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        addCompItem(sp, gbc, 0, 0, 1, 1);

        
        trial.getSystemEvents().addObserver(this);
        ParaProf.incrementNumWindows();
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Print")) {
                    ParaProfUtils.print(panel);
                } else if (arg.equals("Preferences...")) {
                    trial.getPreferences().showPreferencesWindow();
                } else if (arg.equals("Save Image")) {
                    ParaProfImageOutput imageOutput = new ParaProfImageOutput();
                    imageOutput.saveImage((ParaProfImageInterface) panel);
                } else if (arg.equals("Close This Window")) {
                    closeThisWindow();
                } else if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);
                } else if (arg.equals("Show ParaProf Manager")) {
                    (new ParaProfManagerWindow()).show();
                } else if (arg.equals("Show Function Ledger")) {
                    (new LedgerWindow(trial, 0)).show();
                } else if (arg.equals("Show Group Ledger")) {
                    (new LedgerWindow(trial, 1)).show();
                } else if (arg.equals("Show User Event Ledger")) {
                    (new LedgerWindow(trial, 2)).show();
                } else if (arg.equals("Show Call Path Relations")) {
                    CallPathTextWindow tmpRef = new CallPathTextWindow(trial, -1, -1, -1, new DataSorter(trial), 2);
                    trial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();
                } else if (arg.equals("Close All Sub-Windows")) {
                    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
                } else if (arg.equals("About ParaProf")) {
                    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
                } else if (arg.equals("Show Help Window")) {
                    this.help(true);
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuSelected(MenuEvent evt) {
        try {
            if (trial.groupNamesPresent())
                ((JMenuItem) windowsMenu.getItem(2)).setEnabled(true);
            else
                ((JMenuItem) windowsMenu.getItem(2)).setEnabled(false);

            if (trial.userEventsPresent())
                ((JMenuItem) windowsMenu.getItem(3)).setEnabled(true);
            else
                ((JMenuItem) windowsMenu.getItem(3)).setEnabled(false);
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuDeselected(MenuEvent evt) {
    }

    public void menuCanceled(MenuEvent evt) {
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

    private void help(boolean display) {
        ParaProf.helpWindow.clearText();
        if (display)
            ParaProf.helpWindow.show();
        if (windowType == 0) {
            ParaProf.helpWindow.writeText("This is the function ledger window.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("This window shows all the functionProfiles tracked in this profile.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("To see more information about any of the functionProfiles shown here,");
            ParaProf.helpWindow.writeText("right click on that function, and select from the popup menu.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("You can also left click any function to highlight it in the system.");
        } else if (windowType == 1) {
            ParaProf.helpWindow.writeText("This is the group ledger window.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("This window shows all the groups tracked in this profile.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("Left click any group to highlight it in the system.");
            ParaProf.helpWindow.writeText("Right click on any group, and select from the popup menu"
                    + " to display more options for masking or displaying functionProfiles in a particular group.");
        } else {
            ParaProf.helpWindow.writeText("This is the user event ledger window.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("This window shows all the user events tracked in this profile.");
            ParaProf.helpWindow.writeText("");
            ParaProf.helpWindow.writeText("Left click any user event to highlight it in the system.");
            ParaProf.helpWindow.writeText("Right click on any user event, and select from the popup menu.");
        }
    }

    //Updates this window's data copy.
    private void sortLocalData() {
        list = new Vector();
        if (this.windowType == FUNCTION_LEDGER) {
            for (Iterator it = trial.getDataSource().getFunctions(); it.hasNext();) {
                list.addElement(new LedgerDataElement((Function) it.next()));
            }
        } else if (this.windowType == GROUP_LEDGER) {
            for (Iterator it = trial.getDataSource().getGroups(); it.hasNext();) {
                list.addElement(new LedgerDataElement((Group) it.next()));
            }
        } else if (this.windowType == USEREVENT_LEDGER) {
            for (Iterator it = trial.getDataSource().getUserEvents(); it.hasNext();) {
                list.addElement(new LedgerDataElement((UserEvent) it.next()));
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
    void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    void closeThisWindow() {
        try {
            setVisible(false);
            trial.getSystemEvents().deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

   
}