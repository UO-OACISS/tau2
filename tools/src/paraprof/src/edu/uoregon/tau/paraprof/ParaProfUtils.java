package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.print.*;
import java.io.*;
import java.util.Iterator;
import java.util.Vector;
import java.util.zip.GZIPOutputStream;

import javax.swing.*;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import edu.uoregon.tau.dms.dss.*;
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

    //
    //    public static JMenu createTrialMenu(final ParaProfTrial ppTrial, final JFrame owner) {
    //
    //        ActionListener actionListener = new ActionListener() {
    //            public void actionPerformed(ActionEvent evt) {
    //
    //                try {
    //                    Object EventSrc = evt.getSource();
    //
    //                    if (EventSrc instanceof JMenuItem) {
    //                        String arg = evt.getActionCommand();
    //
    //                        if (arg.equals("Show 3D Window")) {
    //
    //                            if (JVMDependent.version.equals("1.3")) {
    //                                JOptionPane.showMessageDialog(
    //                                        owner,
    //                                        "3D Visualization requires Java 1.4 or above\nPlease make sure Java 1.4 is in your path, then reconfigure TAU and re-run ParaProf");
    //                                return;
    //                            }
    //
    //                            //Gears.main(null);
    //                            //(new Gears()).show();
    //
    //                            try {
    //                                (new ThreeDeeWindow(ppTrial)).show();
    //                                //(new ThreeDeeWindow()).show();
    //                            } catch (UnsatisfiedLinkError e) {
    //                                JOptionPane.showMessageDialog(
    //                                        owner,
    //                                        "Unable to load jogl library.  Possible reasons:\nlibjogl.so is not in your LD_LIBRARY_PATH.\nJogl is not built for this platform.\nOpenGL is not installed\n\nJogl is available at jogl.dev.java.net");
    //                            } catch (UnsupportedClassVersionError e) {
    //                                JOptionPane.showMessageDialog(owner,
    //                                        "Unsupported class version.  Are you using Java 1.4 or above?");
    //                            }
    //                        } else if (arg.equals("Show Call Path Relations")) {
    //                            CallPathTextWindow tmpRef = new CallPathTextWindow(ppTrial, -1, -1, -1, null, 2);
    //                            ppTrial.getSystemEvents().addObserver(tmpRef);
    //                            tmpRef.show();
    //                        } else if (arg.equals("Show Mean Call Graph")) {
    //                            CallGraphWindow tmpRef = new CallGraphWindow(ppTrial, ppTrial.getDataSource().getMeanData());
    //                            ppTrial.getSystemEvents().addObserver(tmpRef);
    //                            tmpRef.show();
    //                        } else if (arg.equals("Close All Sub-Windows")) {
    //                            ppTrial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
    //                        }
    //                    }
    //                } catch (Exception e) {
    //                    ParaProfUtils.handleException(e);
    //                }
    //            }
    //
    //        };
    //
    //        JMenu trialMenu = new JMenu("Trial");
    //
    //        JMenuItem menuItem;
    //
    //        menuItem = new JMenuItem("Show 3D Window");
    //        menuItem.addActionListener(actionListener);
    //        trialMenu.add(menuItem);
    //
    //        menuItem = new JMenuItem("Show Call Path Relations");
    //        menuItem.addActionListener(actionListener);
    //        trialMenu.add(menuItem);
    //
    //        menuItem = new JMenuItem("Show Mean Call Graph");
    //        menuItem.addActionListener(actionListener);
    //        trialMenu.add(menuItem);
    //
    //        menuItem = new JMenuItem("Close All Sub-Windows");
    //        menuItem.addActionListener(actionListener);
    //        trialMenu.add(menuItem);
    //
    //        return trialMenu;
    //    }

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

    public static JMenu createThreadMenu(final ParaProfTrial ppTrial, final JFrame owner,
            final edu.uoregon.tau.dms.dss.Thread thread) {

        JMenu threadMenu = new JMenu("Thread");

        JMenuItem menuItem;

        menuItem = new JMenuItem("Function Graph");
        threadMenu.add(menuItem);

        menuItem = new JMenuItem("Callpath Relations");
        threadMenu.add(menuItem);

        menuItem = new JMenuItem("Call Graph");
        threadMenu.add(menuItem);

        menuItem = new JMenuItem("Function Statistics");
        threadMenu.add(menuItem);

        menuItem = new JMenuItem("User Event Statistics");
        threadMenu.add(menuItem);

        return threadMenu;

    }

    public static JMenu createFunctionMenu(final ParaProfTrial ppTrial, final JFrame owner,
            final edu.uoregon.tau.dms.dss.Thread thread) {

        JMenu menu = new JMenu("Function");

        JMenuItem menuItem;

        menuItem = new JMenuItem("Thread Graph");
        menu.add(menuItem);

        menuItem = new JMenuItem("Histogram");
        menu.add(menuItem);

        return menu;

    }

    public static JMenu createWindowsMenu(final ParaProfTrial ppTrial, final JFrame owner) {

        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {

                try {
                    Object EventSrc = evt.getSource();

                    String arg = evt.getActionCommand();

                    if (arg.equals("ParaProf Manager")) {
                        (new ParaProfManagerWindow()).show();
                    } else if (arg.equals("Function Ledger")) {
                        (new LedgerWindow(ppTrial, 0)).show();
                    } else if (arg.equals("Group Ledger")) {
                        (new LedgerWindow(ppTrial, 1)).show();
                    } else if (arg.equals("User Event Ledger")) {
                        (new LedgerWindow(ppTrial, 2)).show();
                    } else if (arg.equals("3D Visualization")) {

                        if (JVMDependent.version.equals("1.3")) {
                            JOptionPane.showMessageDialog(
                                    owner,
                                    "3D Visualization requires Java 1.4 or above\nPlease make sure Java 1.4 is in your path, then reconfigure TAU and re-run ParaProf");
                            return;
                        }

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
                    } else if (arg.equals("Call Path Relations")) {
                        CallPathTextWindow tmpRef = new CallPathTextWindow(ppTrial, -1, -1, -1, null, 2);
                        ppTrial.getSystemEvents().addObserver(tmpRef);
                        tmpRef.show();
                    } else if (arg.equals("Close All Sub-Windows")) {
                        ppTrial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
                    }

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JMenu windowsMenu = new JMenu("Windows");

        JMenuItem menuItem;

        menuItem = new JMenuItem("ParaProf Manager");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        windowsMenu.add(new JSeparator());

        menuItem = new JMenuItem("3D Visualization");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Call Path Relations");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        windowsMenu.add(new JSeparator());

        menuItem = new JMenuItem("Function Ledger");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        final JMenuItem groupLedger = new JMenuItem("Group Ledger");
        groupLedger.addActionListener(actionListener);
        windowsMenu.add(groupLedger);

        final JMenuItem userEventLedger = new JMenuItem("User Event Ledger");
        userEventLedger.addActionListener(actionListener);
        windowsMenu.add(userEventLedger);

        windowsMenu.add(new JSeparator());

        menuItem = new JMenuItem("Close All Sub-Windows");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

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

    private static int findGroupID(Group groups[], Group group) {
        for (int i = 0; i < groups.length; i++) {
            if (groups[i] == group) {
                return i;
            }
        }
        throw new ParaProfException("Couldn't find group: " + group.getName());
    }

    public static void writePacked(DataSource dataSource, File file) throws FileNotFoundException, IOException {
        //File file = new File("/home/amorris/test.ppk");
        FileOutputStream ostream = new FileOutputStream(file);
        GZIPOutputStream gzip = new GZIPOutputStream(ostream);
        BufferedOutputStream bw = new BufferedOutputStream(gzip);
        DataOutputStream p = new DataOutputStream(bw);

        int numFunctions = dataSource.getNumFunctions();
        int numMetrics = dataSource.getNumberOfMetrics();
        int numUserEvents = dataSource.getNumUserEvents();
        int numGroups = dataSource.getNumGroups();

        // write out magic cookie
        p.writeChar('P');
        p.writeChar('P');
        p.writeChar('K');

        // write out version
        p.writeInt(1);

        // write out lowest compatibility version
        p.writeInt(1);

        // write out size of header in bytes
        p.writeInt(0);

        // write out metric names
        p.writeInt(numMetrics);
        for (int i = 0; i < numMetrics; i++) {
            String metricName = dataSource.getMetricName(i);
            p.writeUTF(metricName);
        }

        int idx = 0;

        // write out group names
        p.writeInt(numGroups);
        Group groups[] = new Group[numGroups];
        for (Iterator it = dataSource.getGroups(); it.hasNext();) {
            Group group = (Group) it.next();
            String groupName = group.getName();
            p.writeUTF(groupName);
            groups[idx++] = group;
        }

        Function functions[] = new Function[numFunctions];
        idx = 0;
        // write out function names
        p.writeInt(numFunctions);
        for (Iterator it = dataSource.getFunctions(); it.hasNext();) {
            Function function = (Function) it.next();
            functions[idx++] = function;
            p.writeUTF(function.getName());

            Vector thisGroups = function.getGroups();
            if (thisGroups == null) {
                p.writeInt(0);
            } else {
                p.writeInt(thisGroups.size());
                for (int i = 0; i < thisGroups.size(); i++) {
                    Group group = (Group) thisGroups.get(i);
                    p.writeInt(findGroupID(groups, group));
                }
            }
        }

        UserEvent userEvents[] = new UserEvent[numUserEvents];
        idx = 0;
        // write out user event names
        p.writeInt(numUserEvents);
        for (Iterator it = dataSource.getUserEvents(); it.hasNext();) {
            UserEvent userEvent = (UserEvent) it.next();
            userEvents[idx++] = userEvent;
            p.writeUTF(userEvent.getName());
        }

        p.writeInt(dataSource.getTotalNumberOfThreads());

        for (Iterator it = dataSource.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();

                    //System.out.println("writing " + thread.getNodeID() + "," + thread.getContextID() + "," + thread.getThreadID());
                    p.writeInt(thread.getNodeID());
                    p.writeInt(thread.getContextID());
                    p.writeInt(thread.getThreadID());

                    // count function profiles
                    int count = 0;
                    for (int i = 0; i < numFunctions; i++) {
                        FunctionProfile fp = thread.getFunctionProfile(functions[i]);
                        if (fp != null) {
                            count++;
                        }
                    }
                    p.writeInt(count);

                    // write out function profiles
                    for (int i = 0; i < numFunctions; i++) {
                        FunctionProfile fp = thread.getFunctionProfile(functions[i]);

                        if (fp != null) {
                            p.writeInt(i); // which function
                            p.writeDouble(fp.getNumCalls());
                            p.writeDouble(fp.getNumSubr());

                            for (int j = 0; j < numMetrics; j++) {
                                p.writeDouble(fp.getExclusive(j));
                                p.writeDouble(fp.getInclusive(j));
                            }
                        }
                    }

                    // count user event profiles
                    count = 0;
                    for (int i = 0; i < numUserEvents; i++) {
                        UserEventProfile uep = thread.getUserEventProfile(userEvents[i]);
                        if (uep != null) {
                            count++;
                        }
                    }

                    p.writeInt(count); // number of user event profiles

                    // write out user event profiles
                    for (int i = 0; i < numUserEvents; i++) {
                        UserEventProfile uep = thread.getUserEventProfile(userEvents[i]);

                        if (uep != null) {
                            p.writeInt(i);
                            p.writeInt(uep.getUserEventNumberValue());
                            p.writeDouble(uep.getUserEventMinValue());
                            p.writeDouble(uep.getUserEventMaxValue());
                            p.writeDouble(uep.getUserEventMeanValue());
                            p.writeDouble(uep.getUserEventSumSquared());
                        }
                    }
                }
            }
        }

        p.close();
        gzip.close();
        ostream.close();

    }

    public static void exportTrial(ParaProfTrial ppTrial, Component owner) {

        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Export Trial");
        //Set the directory.
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        javax.swing.filechooser.FileFilter fileFilters[] = fileChooser.getChoosableFileFilters();

        fileChooser.setFileFilter(new ParaProfImageFormatFileFilter(ParaProfImageFormatFileFilter.PPK));

        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int resultValue = fileChooser.showSaveDialog(owner);
        if (resultValue != JFileChooser.APPROVE_OPTION) {
            return;
        }

        try {

            File file = fileChooser.getSelectedFile();
            String path = file.getCanonicalPath();

            String extension = ParaProfImageFormatFileFilter.getExtension(file);
            if (extension == null) {
                path = path + ".ppk";
                file = new File(path);
            }

            if (file.exists()) {
                int response = JOptionPane.showConfirmDialog(owner, file + " already exists\nOverwrite existing file?",
                        "Confirm Overwrite", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
                if (response == JOptionPane.CANCEL_OPTION)
                    return;
            }

            writePacked(ppTrial.getDataSource(), file);

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }

    }

    private static void writeMetric(String root, DataSource dataSource, int metricID, Function[] functions, String[] groupStrings, UserEvent[] userEvents) throws IOException {

        int numFunctions = dataSource.getNumFunctions();
        int numMetrics = dataSource.getNumberOfMetrics();
        int numUserEvents = dataSource.getNumUserEvents();
        int numGroups = dataSource.getNumGroups();

        for (Iterator it = dataSource.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();

                    File file = new File(root + "/profile." + thread.getNodeID() + "." + thread.getContextID() + "."
                            + thread.getThreadID());

                    FileOutputStream out = new FileOutputStream(file);
                    OutputStreamWriter outWriter = new OutputStreamWriter(out);
                    BufferedWriter bw = new BufferedWriter(outWriter);

                    // count function profiles
                    int count = 0;
                    for (int i = 0; i < numFunctions; i++) {
                        FunctionProfile fp = thread.getFunctionProfile(functions[i]);
                        if (fp != null) {
                            count++;
                        }
                    }

                    if (dataSource.getNumberOfMetrics() == 0 && dataSource.getMetricName(metricID) == "Time") {
                        bw.write(count + " templated_functions\n");
                    } else {
                        bw.write(count + " templated_functions_MULTI_" + dataSource.getMetricName(metricID) + "\n");
                    }
                    bw.write("# Name Calls Subrs Excl Incl ProfileCalls\n");

                    // write out function profiles
                    for (int i = 0; i < numFunctions; i++) {
                        FunctionProfile fp = thread.getFunctionProfile(functions[i]);

                        if (fp != null) {
                            bw.write('"' + functions[i].getName() + "\" ");
                            bw.write((int)fp.getNumCalls() + " ");
                            bw.write((int)fp.getNumSubr() + " ");
                            bw.write(fp.getExclusive(metricID) + " ");
                            bw.write(fp.getInclusive(metricID) + " ");
                            bw.write("0 " + "GROUP=\"" + groupStrings[i] + "\"\n");
                        }
                    }
                    
                    bw.write("0 aggregates\n");
                    
                    // count user event profiles
                    count = 0;
                    for (int i = 0; i < numUserEvents; i++) {
                        UserEventProfile uep = thread.getUserEventProfile(userEvents[i]);
                        if (uep != null) {
                            count++;
                        }
                    }

                    bw.write(count + " userevents\n");
                    bw.write("# eventname numevents max min mean sumsqr\n");
                    
                    // write out user event profiles
                    for (int i = 0; i < numUserEvents; i++) {
                        UserEventProfile uep = thread.getUserEventProfile(userEvents[i]);

                        if (uep != null) {
                            bw.write('"' + userEvents[i].getName() + "\" ");
                            bw.write(uep.getUserEventNumberValue() + " ");
                            bw.write(uep.getUserEventMaxValue() + " ");
                            bw.write(uep.getUserEventMinValue() + " ");
                            bw.write(uep.getUserEventMeanValue() + " ");
                            bw.write(uep.getUserEventSumSquared() + "\n");
                        }
                    }
                    bw.close();
                    outWriter.close();
                    out.close();
                }
            }
        }

    }

    public static void writeProfiles(DataSource dataSource, File directory) throws IOException {

        
        int numFunctions = dataSource.getNumFunctions();
        int numMetrics = dataSource.getNumberOfMetrics();
        int numUserEvents = dataSource.getNumUserEvents();
        int numGroups = dataSource.getNumGroups();

        int idx = 0;

        // write out group names
        Group groups[] = new Group[numGroups];
        for (Iterator it = dataSource.getGroups(); it.hasNext();) {
            Group group = (Group) it.next();
            String groupName = group.getName();
            groups[idx++] = group;
        }

        Function functions[] = new Function[numFunctions];
        String groupStrings[] = new String[numFunctions];
        idx = 0;
        // write out function names
        for (Iterator it = dataSource.getFunctions(); it.hasNext();) {
            Function function = (Function) it.next();
            functions[idx] = function;

            Vector thisGroups = function.getGroups();
            if (thisGroups == null) {
                groupStrings[idx] = "";
            } else {
                groupStrings[idx] = "";

                for (int i = 0; i < thisGroups.size(); i++) {
                    Group group = (Group) thisGroups.get(i);
                    groupStrings[idx] = groupStrings[idx] + " " + group.getName();
                }

                groupStrings[idx] = groupStrings[idx].trim();
            }
            idx++;
        }

        UserEvent userEvents[] = new UserEvent[numUserEvents];
        idx = 0;
        // collect user event names
        for (Iterator it = dataSource.getUserEvents(); it.hasNext();) {
            UserEvent userEvent = (UserEvent) it.next();
            userEvents[idx++] = userEvent;
        }
        
        if (numMetrics == 1) {
            writeMetric(".", dataSource, 0, functions, groupStrings, userEvents);
        } else {
            for (int i = 0; i < numMetrics; i++) {
                String name = "MULTI__" + dataSource.getMetricName(i);
                boolean success = (new File(name).mkdir());
                if (!success) {
                    System.err.println("Failed to create directory: " + name);
                } else {
                    writeMetric(name, dataSource, 0, functions, groupStrings, userEvents);
                }
            }
        }

    }

    public static String getFunctionName(Function function) {
        if (ParaProf.preferences.getReversedCallPaths()) {
            return function.getReversedName();
        }
        return function.getName();
    }

    public static void handleException(Exception e) {
        new ParaProfErrorDialog(e);
    }

}
