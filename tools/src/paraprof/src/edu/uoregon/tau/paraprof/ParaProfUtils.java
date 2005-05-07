package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.ClipboardOwner;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.print.*;

import javax.swing.*;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import edu.uoregon.tau.dms.dss.Function;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;

public class ParaProfUtils {

    static boolean verbose;
    static boolean verboseSet;

    // Suppress default constructor for noninstantiability
    private ParaProfUtils() {
        // This constructor will never be invoked
    }

    private static void checkVerbose() {
        if (!verboseSet) {
            if (System.getProperty("paraprof.verbose") != null) {
                verbose = true;
            }
            verboseSet = true;
        }
    }

    public static void verr(String string) {
        checkVerbose();

        if (verbose) {
            System.err.println(string);
        }
    }

    public static void vout(String string) {
        checkVerbose();

        if (verbose) {
            System.out.println(string);
        }
    }

    public static void vout(Object obj, String string) {
        checkVerbose();

        if (verbose) {

            String className = obj.getClass().getName();
            int lastDot = className.lastIndexOf('.');
            if (lastDot != -1) {
                className = className.substring(lastDot + 1);
            }

            System.out.println(className + ": " + string);
        }
    }

    public static void verr(Object obj, String string) {
        checkVerbose();

        if (verbose) {

            String className = obj.getClass().getName();
            int lastDot = className.lastIndexOf('.');
            if (lastDot != -1) {
                className = className.substring(lastDot + 1);
            }

            System.err.println(className + ": " + string);
        }
    }

    public static void helperAddRadioMenuItem(String name, String command, boolean on, ButtonGroup group, JMenu menu,
            ActionListener act) {
        JRadioButtonMenuItem item = new JRadioButtonMenuItem(name, on);
        item.addActionListener(act);
        item.setActionCommand(command);
        group.add(item);
        menu.add(item);
    }

    public static void addCompItem(Container jPanel, Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        jPanel.add(c, gbc);
    }

    public static void print(Printable printable) {
        PrinterJob job = PrinterJob.getPrinterJob();
        PageFormat defaultFormat = job.defaultPage();
        PageFormat selectedFormat = job.pageDialog(defaultFormat);
        if (defaultFormat != selectedFormat) { // only proceed if the user did not select cancel
            job.setPrintable(printable, selectedFormat);
            //if (job.getPrintService() != null) {
            if (job.printDialog()) { // only proceed if the user did not select cancel
                try {
                    job.print();
                } catch (PrinterException e) {
                    ParaProfUtils.handleException(e);
                }
            }
            //}
        }

    }

    public static JMenu createTrialMenu(final ParaProfTrial ppTrial, final JFrame owner) {

        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {

                try {
                    Object EventSrc = evt.getSource();

                    if (EventSrc instanceof JMenuItem) {
                        String arg = evt.getActionCommand();

                        if (arg.equals("Show 3D Window")) {

                            //Gears.main(null);
                            //(new Gears()).show();

                            try {
                                (new ThreeDeeWindow(ppTrial)).show();
                                //(new ThreeDeeWindow()).show();
                            } catch (UnsatisfiedLinkError e) {
                                JOptionPane.showMessageDialog(
                                        owner,
                                        "Unable to load jogl library.  Possible reasons:\nlibjogl.so is not in your LD_LIBRARY_PATH.\nJogl is not built for this platform.\nOpenGL is not installed\n\nJogl is available at jogl.dev.java.net");
                            } catch (UnsupportedClassVersionError e) {
                                JOptionPane.showMessageDialog(owner,
                                        "Unsupported class version.  Are you using Java 1.4 or above?");
                            }
                        } else if (arg.equals("Show Call Path Relations")) {
                            CallPathTextWindow tmpRef = new CallPathTextWindow(ppTrial, -1, -1, -1, null, 2);
                            ppTrial.getSystemEvents().addObserver(tmpRef);
                            tmpRef.show();
                        } else if (arg.equals("Show Mean Call Graph")) {
                            CallGraphWindow tmpRef = new CallGraphWindow(ppTrial, ppTrial.getDataSource().getMeanData());
                            ppTrial.getSystemEvents().addObserver(tmpRef);
                            tmpRef.show();
                        } else if (arg.equals("Close All Sub-Windows")) {
                            ppTrial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
                        }
                    }
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JMenu trialMenu = new JMenu("Trial");

        JMenuItem menuItem;

        menuItem = new JMenuItem("Show 3D Window");
        menuItem.addActionListener(actionListener);
        trialMenu.add(menuItem);

        menuItem = new JMenuItem("Show Call Path Relations");
        menuItem.addActionListener(actionListener);
        trialMenu.add(menuItem);

        menuItem = new JMenuItem("Show Mean Call Graph");
        menuItem.addActionListener(actionListener);
        trialMenu.add(menuItem);

        menuItem = new JMenuItem("Close All Sub-Windows");
        menuItem.addActionListener(actionListener);
        trialMenu.add(menuItem);

        return trialMenu;
    }

    public static JMenu createHelpMenu(final JFrame owner, final ParaProfWindow ppWindow) {

        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {

                try {
                    Object EventSrc = evt.getSource();

                    if (EventSrc instanceof JMenuItem) {
                        String arg = evt.getActionCommand();

                        if (arg.equals("About ParaProf")) {
                            JOptionPane.showMessageDialog(owner, ParaProf.getInfoString());
                        } else if (arg.equals("Show Help Window")) {
                            ppWindow.help(true);
                        }
                    }
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JMenu helpMenu = new JMenu("Help");

        JMenuItem menuItem = new JMenuItem("Show Help Window");
        menuItem.addActionListener(actionListener);
        helpMenu.add(menuItem);

        menuItem = new JMenuItem("About ParaProf");
        menuItem.addActionListener(actionListener);
        helpMenu.add(menuItem);

        return helpMenu;
    }

    public static JMenu createFileMenu(final ParaProfWindow window, final Printable printable, final Object panel) {

        if (printable == null) {
            throw new ParaProfException("File menu created with null panel!");
        }

        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {

                try {
                    Object EventSrc = evt.getSource();

                    String arg = evt.getActionCommand();

                    if (arg.equals("Print")) {
                        ParaProfUtils.print(printable);
                    } else if (arg.equals("Preferences...")) {
                        ParaProf.preferencesWindow.showPreferencesWindow();
                    } else if (arg.equals("Save Image")) {

                        if (panel instanceof ParaProfImageInterface) {
                            ParaProfImageInterface ppImageInterface = (ParaProfImageInterface) panel;
                            ParaProfImageOutput.saveImage(ppImageInterface);
                        }

                        if (panel instanceof ThreeDeeWindow) {
                            ThreeDeeWindow threeDeeWindow = (ThreeDeeWindow) panel;
                            ParaProfImageOutput.save3dImage(threeDeeWindow);
                        }

                    } else if (arg.equals("Close This Window")) {
                        window.closeThisWindow();
                    } else if (arg.equals("Exit ParaProf!")) {
                        ParaProf.exitParaProf(0);
                    }
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JMenu fileMenu = new JMenu("File");

        JMenu subMenu = new JMenu("Save ...");

        JMenuItem menuItem = new JMenuItem("Save Image");
        menuItem.addActionListener(actionListener);
        subMenu.add(menuItem);

        fileMenu.add(subMenu);

        menuItem = new JMenuItem("Preferences...");
        menuItem.addActionListener(actionListener);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Print");
        menuItem.addActionListener(actionListener);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Close This Window");
        menuItem.addActionListener(actionListener);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Exit ParaProf!");
        menuItem.addActionListener(actionListener);
        fileMenu.add(menuItem);

        return fileMenu;
    }

    public static JMenu createWindowsMenu(final ParaProfTrial ppTrial, final JFrame owner) {

        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {

                try {
                    Object EventSrc = evt.getSource();

                    String arg = evt.getActionCommand();

                    if (arg.equals("Show ParaProf Manager")) {
                        (new ParaProfManagerWindow()).show();
                    } else if (arg.equals("Show Function Ledger")) {
                        (new LedgerWindow(ppTrial, 0)).show();
                    } else if (arg.equals("Show Group Ledger")) {
                        (new LedgerWindow(ppTrial, 1)).show();
                    } else if (arg.equals("Show User Event Ledger")) {
                        (new LedgerWindow(ppTrial, 2)).show();
                    }
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JMenu windowsMenu = new JMenu("Windows");

        JMenuItem menuItem;

        menuItem = new JMenuItem("Show ParaProf Manager");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Function Ledger");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        final JMenuItem groupLedger = new JMenuItem("Show Group Ledger");
        groupLedger.addActionListener(actionListener);
        windowsMenu.add(groupLedger);

        final JMenuItem userEventLedger = new JMenuItem("Show User Event Ledger");
        userEventLedger.addActionListener(actionListener);
        windowsMenu.add(userEventLedger);

        MenuListener menuListener = new MenuListener() {
            public void menuSelected(MenuEvent evt) {
                try {
                    groupLedger.setEnabled(ppTrial.groupNamesPresent());
                    userEventLedger.setEnabled(ppTrial.userEventsPresent());
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

            public void menuCanceled(MenuEvent e) {
            }

            public void menuDeselected(MenuEvent e) {
            }
        };

        windowsMenu.addMenuListener(menuListener);

        return windowsMenu;
    }

    public static void scaleForPrint(Graphics g, PageFormat pageFormat, int width, int height) {
        double pageWidth = pageFormat.getImageableWidth();
        double pageHeight = pageFormat.getImageableHeight();
        int cols = (int) (width / pageWidth) + 1;
        int rows = (int) (height / pageHeight) + 1;
        double xScale = pageWidth / width;
        double yScale = pageHeight / height;
        double scale = Math.min(xScale, yScale);

        double tx = 0.0;
        double ty = 0.0;
        if (xScale > scale) {
            tx = 0.5 * (xScale - scale) * width;
        } else {
            ty = 0.5 * (yScale - scale) * height;
        }

        Graphics2D g2 = (Graphics2D) g;

        g2.translate((int) pageFormat.getImageableX(), (int) pageFormat.getImageableY());
        g2.translate(tx, ty);
        g2.scale(scale, scale);
    }

    public static JPopupMenu createFunctionClickPopUp(final ParaProfTrial ppTrial, final Function function,
            final JComponent owner) {
        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    Object EventSrc = evt.getSource();

                    String arg = evt.getActionCommand();

                    if (arg.equals("Show Function Details")) {
                        FunctionDataWindow functionDataWindow = new FunctionDataWindow(ppTrial, function);
                        ppTrial.getSystemEvents().addObserver(functionDataWindow);
                        functionDataWindow.show();
                    } else if (arg.equals("Show Function Histogram")) {

                        HistogramWindow hw = new HistogramWindow(ppTrial, function);
                        ppTrial.getSystemEvents().addObserver(hw);
                        hw.show();
                    } else if (arg.equals("Assign Function Color")) {
                        ParaProf.colorMap.assignColor(owner, function);
                    } else if (arg.equals("Reset to Default Color")) {
                        ParaProf.colorMap.removeColor(function);
                        ParaProf.colorMap.reassignColors();
                    }

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JPopupMenu functionPopup = new JPopupMenu();

        //Add items to the third popup menu.
        JMenuItem functionDetailsItem = new JMenuItem("Show Function Details");
        functionDetailsItem.addActionListener(actionListener);
        functionPopup.add(functionDetailsItem);

        JMenuItem functionHistogramItem = new JMenuItem("Show Function Histogram");
        functionHistogramItem.addActionListener(actionListener);
        functionPopup.add(functionHistogramItem);

        JMenuItem jMenuItem = new JMenuItem("Assign Function Color");
        jMenuItem.addActionListener(actionListener);
        functionPopup.add(jMenuItem);

        jMenuItem = new JMenuItem("Reset to Default Color");
        jMenuItem.addActionListener(actionListener);
        functionPopup.add(jMenuItem);

        return functionPopup;

    }

    public static void handleThreadClick(final ParaProfTrial ppTrial, final edu.uoregon.tau.dms.dss.Thread thread,
            JPanel owner, MouseEvent evt) {
        if (thread.getNodeID() == -1) { // mean
            JPopupMenu meanThreadPopup = new JPopupMenu();
            ActionListener actionListener = new ActionListener() {
                public void actionPerformed(ActionEvent evt) {
                    try {
                        Object EventSrc = evt.getSource();

                        String arg = evt.getActionCommand();
                        if (arg.equals("Show Mean Statistics Window")) {
                            StatWindow statWindow = new StatWindow(ppTrial, -1, -1, -1, false);
                            ppTrial.getSystemEvents().addObserver(statWindow);
                            statWindow.show();
                        } else if (arg.equals("Show Mean Call Graph")) {
                            CallGraphWindow tmpRef = new CallGraphWindow(ppTrial, ppTrial.getDataSource().getMeanData());
                            ppTrial.getSystemEvents().addObserver(tmpRef);
                            tmpRef.show();
                        } else if (arg.equals("Show Mean Call Path Thread Relations")) {
                            CallPathTextWindow callPathTextWindow = new CallPathTextWindow(ppTrial, -1, -1, -1, null, 0);
                            ppTrial.getSystemEvents().addObserver(callPathTextWindow);
                            callPathTextWindow.show();
                        }
                    } catch (Exception e) {
                        ParaProfUtils.handleException(e);
                    }
                }
            };

            meanThreadPopup = new JPopupMenu();

            JMenuItem jMenuItem = new JMenuItem("Show Mean Statistics Window");
            jMenuItem.addActionListener(actionListener);
            meanThreadPopup.add(jMenuItem);

            jMenuItem = new JMenuItem("Show Mean Call Path Thread Relations");
            jMenuItem.addActionListener(actionListener);
            meanThreadPopup.add(jMenuItem);

            jMenuItem = new JMenuItem("Show Mean Call Graph");
            jMenuItem.addActionListener(actionListener);
            meanThreadPopup.add(jMenuItem);

            meanThreadPopup.show(owner, evt.getX(), evt.getY());
        } else {
            JPopupMenu threadPopup = new JPopupMenu();
            ActionListener actionListener = new ActionListener() {
                public void actionPerformed(ActionEvent evt) {
                    try {
                        Object EventSrc = evt.getSource();

                        String arg = evt.getActionCommand();
                        if (arg.equals("Show Thread Statistics Window")) {
                            StatWindow statWindow = new StatWindow(ppTrial, thread.getNodeID(), thread.getContextID(),
                                    thread.getThreadID(), false);
                            ppTrial.getSystemEvents().addObserver(statWindow);
                            statWindow.show();
                        } else if (arg.equals("Show Thread Call Graph")) {
                            CallGraphWindow tmpRef = new CallGraphWindow(ppTrial, thread);
                            ppTrial.getSystemEvents().addObserver(tmpRef);
                            tmpRef.show();
                        } else if (arg.equals("Show Call Path Thread Relations")) {
                            CallPathTextWindow callPathTextWindow = new CallPathTextWindow(ppTrial, thread.getNodeID(),
                                    thread.getContextID(), thread.getThreadID(), null, 1);
                            ppTrial.getSystemEvents().addObserver(callPathTextWindow);
                            callPathTextWindow.show();
                        } else if (arg.equals("Show User Event Statistics Window")) {
                            StatWindow statWindow = new StatWindow(ppTrial, thread.getNodeID(), thread.getContextID(),
                                    thread.getThreadID(), true);
                            ppTrial.getSystemEvents().addObserver(statWindow);
                            statWindow.show();
                        }

                    } catch (Exception e) {
                        ParaProfUtils.handleException(e);
                    }
                }

            };

            threadPopup = new JPopupMenu();

            JMenuItem jMenuItem = new JMenuItem("Show Thread Statistics Window");
            jMenuItem.addActionListener(actionListener);
            threadPopup.add(jMenuItem);

            if (ppTrial.userEventsPresent()) {

                jMenuItem = new JMenuItem("Show User Event Statistics Window");
                jMenuItem.addActionListener(actionListener);
                threadPopup.add(jMenuItem);
            }

            jMenuItem = new JMenuItem("Show Call Path Thread Relations");
            jMenuItem.addActionListener(actionListener);
            threadPopup.add(jMenuItem);

            jMenuItem = new JMenuItem("Show Thread Call Graph");
            jMenuItem.addActionListener(actionListener);
            threadPopup.add(jMenuItem);

            threadPopup.show(owner, evt.getX(), evt.getY());

        }
    }

    public static int[] computeClipping(Rectangle clipRect, Rectangle viewRect, boolean toScreen, boolean fullWindow,
            int size, int barSpacing, int yCoord) {

        int startElement, endElement;
        if (!fullWindow) {
            int yBeg = 0;
            int yEnd = 0;

            if (toScreen) {
                yBeg = (int) clipRect.getY();
                yEnd = (int) (yBeg + clipRect.getHeight());
            } else {
                yBeg = (int) viewRect.getY();
                yEnd = (int) (yBeg + viewRect.getHeight());
            }
            startElement = ((yBeg - yCoord) / barSpacing) - 1;
            endElement = ((yEnd - yCoord) / barSpacing) + 1;

            if (startElement < 0)
                startElement = 0;

            if (endElement < 0)
                endElement = 0;

            if (startElement > (size - 1))
                startElement = (size - 1);

            if (endElement > (size - 1))
                endElement = (size - 1);

            if (toScreen)
                yCoord = yCoord + (startElement * barSpacing);
        } else {
            startElement = 0;
            endElement = (size - 1);
        }

        int[] clips = new int[3];
        clips[0] = startElement;
        clips[1] = endElement;
        clips[2] = yCoord;
        return clips;
    }

    public static JMenu createUnitsMenu(final UnitListener unitListener, int initialUnits) {

        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    Object EventSrc = evt.getSource();

                    String arg = evt.getActionCommand();

                    if (arg.equals("Microseconds")) {
                        unitListener.setUnits(0);
                    } else if (arg.equals("Milliseconds")) {
                        unitListener.setUnits(1);
                    } else if (arg.equals("Seconds")) {
                        unitListener.setUnits(2);
                    } else if (arg.equals("hr:min:sec")) {
                        unitListener.setUnits(3);
                    }

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JMenu unitsSubMenu = new JMenu("Select Units");
        ButtonGroup group = new ButtonGroup();

        JRadioButtonMenuItem button = new JRadioButtonMenuItem("Microseconds", initialUnits == 0);
        button.addActionListener(actionListener);
        group.add(button);
        unitsSubMenu.add(button);

        button = new JRadioButtonMenuItem("Milliseconds", initialUnits == 1);
        button.addActionListener(actionListener);
        group.add(button);
        unitsSubMenu.add(button);

        button = new JRadioButtonMenuItem("Seconds", initialUnits == 2);
        button.addActionListener(actionListener);
        group.add(button);
        unitsSubMenu.add(button);

        button = new JRadioButtonMenuItem("hr:min:sec", initialUnits == 3);
        button.addActionListener(actionListener);
        group.add(button);
        unitsSubMenu.add(button);

        return unitsSubMenu;
    }

    public static void handleException(Exception e) {
        new ParaProfErrorDialog(e);
    }

}
