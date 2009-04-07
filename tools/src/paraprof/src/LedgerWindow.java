package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;

import javax.swing.*;

import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Group;
import edu.uoregon.tau.perfdmf.UserEvent;

/**
 * LedgerWindow
 * This object represents the ledger window.
 *  
 * <P>CVS $Id: LedgerWindow.java,v 1.7 2009/04/07 20:31:44 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.7 $
 * @see		LedgerDataElement
 * @see		LedgerWindowPanel
 */
public class LedgerWindow extends JFrame implements Observer, ParaProfWindow {

    public static final int FUNCTION_LEGEND = 0;
    public static final int GROUP_LEGEND = 1;
    public static final int USEREVENT_LEGEND = 2;
    public static final int PHASE_LEGEND = 3;
    private int windowType = -1; //0:function, 1:group, 2:userevent, 3:phase

    private ParaProfTrial ppTrial = null;
    private JScrollPane sp = null;
    private LedgerWindowPanel panel = null;
    private Vector list = new Vector();

    public LedgerWindow(ParaProfTrial ppTrial, int windowType, Component parent) {
        this.ppTrial = ppTrial;
        ppTrial.addObserver(this);

        this.windowType = windowType;

        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(350, 450)));

        //setLocation(new java.awt.Point(300, 200));
        setLocation(WindowPlacer.getNewLedgerLocation(this, parent));

        //Now set the title.
        switch (windowType) {
        case FUNCTION_LEGEND:
            this.setTitle("TAU: ParaProf: Function Legend: " + ppTrial.getTrialIdentifier(true));
            break;
        case GROUP_LEGEND:
            this.setTitle("TAU: ParaProf: Group Legend: " + ppTrial.getTrialIdentifier(true));
            break;
        case USEREVENT_LEGEND:
            this.setTitle("TAU: ParaProf: User Event Legend: " + ppTrial.getTrialIdentifier(true));
            break;
        case PHASE_LEGEND:
            this.setTitle("TAU: ParaProf: Phase Legend: " + ppTrial.getTrialIdentifier(true));
            break;
        default:
            throw new ParaProfException("Invalid Legend Window Type: " + windowType);
        }
        ParaProfUtils.setFrameIcon(this);

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Set the help window text if required.
        if (ParaProf.getHelpWindow().isVisible()) {
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

    public void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        mainMenu.add(ParaProfUtils.createFileMenu(this, panel, panel));
        //mainMenu.add(ParaProfUtils.createTrialMenu(trial, this));

        if (this.windowType == FUNCTION_LEGEND) {
            JMenu filter = new JMenu("Filter");
            JMenuItem advanced = new JMenuItem("Advanced Filtering...");
            advanced.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    (new FunctionFilterDialog(LedgerWindow.this, ppTrial)).setVisible(true);
                }
            });
            filter.add(advanced);
            mainMenu.add(filter);
        }
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
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
        ParaProf.getHelpWindow().clearText();
        if (display) {
            ParaProf.getHelpWindow().setVisible(true);
        }
        if (windowType == 0) {
            ParaProf.getHelpWindow().writeText("This is the function ledger window.\n");
            ParaProf.getHelpWindow().writeText("This window shows all the functions tracked in this profile.\n");
            ParaProf.getHelpWindow().writeText("To see more information about any of the functions shown here,");
            ParaProf.getHelpWindow().writeText("right click on that function, and select from the popup menu.\n");
            ParaProf.getHelpWindow().writeText("You can also left click any function to highlight it in the system.");
        } else if (windowType == 1) {
            ParaProf.getHelpWindow().writeText("This is the group ledger window.\n");
            ParaProf.getHelpWindow().writeText("This window shows all the groups tracked in this profile.\n");
            ParaProf.getHelpWindow().writeText("Left click any group to highlight it in the system.");
            ParaProf.getHelpWindow().writeText(
                    "Right click on any group, and select from the popup menu"
                            + " to display more options for masking or displaying functions in a particular group.");
        } else {
            ParaProf.getHelpWindow().writeText("This is the user event ledger window.\n");
            ParaProf.getHelpWindow().writeText("This window shows all the user events tracked in this profile.\n");
            ParaProf.getHelpWindow().writeText("Left click any user event to highlight it in the system.");
            ParaProf.getHelpWindow().writeText("Right click on any user event, and select from the popup menu.");
        }
    }

    //Updates this window's data copy.
    private void sortLocalData() {
        list = new Vector();
        if (this.windowType == FUNCTION_LEGEND) {
            for (Iterator it = ppTrial.getDataSource().getFunctions(); it.hasNext();) {
                list.addElement(new LedgerDataElement((Function) it.next()));
            }
        } else if (this.windowType == GROUP_LEGEND) {
            for (Iterator it = ppTrial.getDataSource().getGroups(); it.hasNext();) {
                list.addElement(new LedgerDataElement((Group) it.next()));
            }
        } else if (this.windowType == USEREVENT_LEGEND) {
            for (Iterator it = ppTrial.getDataSource().getUserEvents(); it.hasNext();) {
                list.addElement(new LedgerDataElement((UserEvent) it.next()));
            }
        } else if (this.windowType == PHASE_LEGEND) {
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

    public int getWindowType() {
        return windowType;
    }

    public JFrame getFrame() {
        return this;
    }

}