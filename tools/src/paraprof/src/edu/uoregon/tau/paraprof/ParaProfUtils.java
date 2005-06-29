package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.zip.GZIPOutputStream;

import javax.swing.*;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.paraprof.interfaces.ImageExport;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;
import edu.uoregon.tau.paraprof.treetable.TreeTableWindow;

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

                        if (panel instanceof ImageExport) {
                            ImageExport ppImageInterface = (ImageExport) panel;
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

    private static JMenuItem createMenuItem(String text, ActionListener actionListener, boolean enabled) {
        JMenuItem menuItem = new JMenuItem(text);
        menuItem.setEnabled(enabled);
        menuItem.addActionListener(actionListener);
        return menuItem;
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
                            JOptionPane.showMessageDialog(owner, "3D Visualization requires Java 1.4 or above\n"
                                    + "Please make sure Java 1.4 is in your path, then reconfigure TAU and re-run ParaProf");
                            return;
                        }

                        //Gears.main(null);
                        //(new Gears()).show();

                        try {
                            (new ThreeDeeWindow(ppTrial)).show();
                            //(new ThreeDeeWindow()).show();
                        } catch (UnsatisfiedLinkError e) {
                            JOptionPane.showMessageDialog(owner, "Unable to load jogl library.  Possible reasons:\n"
                                    + "libjogl.so is not in your LD_LIBRARY_PATH.\n"
                                    + "Jogl is not built for this platform.\nOpenGL is not installed\n\n"
                                    + "Jogl is available at jogl.dev.java.net");
                        } catch (UnsupportedClassVersionError e) {
                            JOptionPane.showMessageDialog(owner, "Unsupported class version.  Are you using Java 1.4 or above?");
                        }
                    } else if (arg.equals("Call Path Relations")) {
                        CallPathTextWindow tmpRef = new CallPathTextWindow(ppTrial, ppTrial.getDataSource().getMeanData(), 1);
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

        //menuItem = new JMenuItem("Call Path Relations");
        //menuItem.addActionListener(actionListener);
        //windowsMenu.add(menuItem);

        windowsMenu.add(new JSeparator());

        ActionListener fActionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                FunctionSelectorDialog fSelector = new FunctionSelectorDialog(owner, true,
                        ppTrial.getDataSource().getFunctions(), null, false);
                if (fSelector.choose()) {
                    Function selectedFunction = (Function) fSelector.getSelectedObject();

                    Object EventSrc = evt.getSource();

                    String arg = evt.getActionCommand();

                    if (arg.equals("Bar Chart")) {
                        FunctionDataWindow w = new FunctionDataWindow(ppTrial, selectedFunction);
                        w.show();
                    } else if (arg.equals("Histogram")) {
                        HistogramWindow w = new HistogramWindow(ppTrial, selectedFunction);
                        w.show();
                    }
                }
            }
        };

        final JMenu functionWindows = new JMenu("Function");
        functionWindows.add(createMenuItem("Bar Chart", fActionListener, true));
        functionWindows.add(createMenuItem("Histogram", fActionListener, true));
        windowsMenu.add(functionWindows);

        ActionListener tActionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                Object EventSrc = evt.getSource();
                String arg = evt.getActionCommand();

                List list = new ArrayList(ppTrial.getDataSource().getAllThreads());
                if (ppTrial.getDataSource().getAllThreads().size() > 1  && arg.equals("User Event Statistics") == false) {
                    list.add(0, ppTrial.getDataSource().getStdDevData());
                    list.add(1, ppTrial.getDataSource().getMeanData());
                }

                FunctionSelectorDialog fSelector = new FunctionSelectorDialog(owner, true, list.iterator(), null, false);
                fSelector.setTitle("Select a Thread");
                if (fSelector.choose()) {
                    edu.uoregon.tau.dms.dss.Thread selectedThread = (edu.uoregon.tau.dms.dss.Thread) fSelector.getSelectedObject();


                    if (arg.equals("Bar Chart")) {
                        ThreadDataWindow w = new ThreadDataWindow(ppTrial, selectedThread);
                        w.setVisible(true);
                    } else if (arg.equals("Statistics Text")) {
                        (new StatWindow(ppTrial,selectedThread, false)).setVisible(true);
                    } else if (arg.equals("Statistics Table")) {
                        (new TreeTableWindow(ppTrial,selectedThread)).setVisible(true);
                    } else if (arg.equals("Call Graph")) {
                        (new CallGraphWindow(ppTrial,selectedThread)).setVisible(true);
                    } else if (arg.equals("Call Path Relations")) {
                        (new CallPathTextWindow(ppTrial,selectedThread,0)).setVisible(true);
                    } else if (arg.equals("User Event Statistics")) {
                        (new StatWindow(ppTrial,selectedThread, true)).setVisible(true);
                    }
                }
            }
        };

        final JMenu threadWindows = new JMenu("Thread");
        threadWindows.add(createMenuItem("Bar Chart", tActionListener, true));
        threadWindows.add(createMenuItem("Statistics Text", tActionListener, true));
        threadWindows.add(createMenuItem("Statistics Table", tActionListener, true));
        threadWindows.add(createMenuItem("Call Graph", tActionListener, ppTrial.callPathDataPresent()));
        threadWindows.add(createMenuItem("Call Path Relations", tActionListener, ppTrial.callPathDataPresent()));
        threadWindows.add(createMenuItem("User Event Statistics", tActionListener, ppTrial.userEventsPresent()));

        windowsMenu.add(threadWindows);

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

    public static JPopupMenu createFunctionClickPopUp(final ParaProfTrial ppTrial, final Function function, final JComponent owner) {
        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    Object EventSrc = evt.getSource();

                    String arg = evt.getActionCommand();

                    if (arg.equals("Show Function Bar Chart")) {
                        FunctionDataWindow functionDataWindow = new FunctionDataWindow(ppTrial, function);
                        functionDataWindow.show();
                    } else if (arg.equals("Show Function Histogram")) {
                        HistogramWindow hw = new HistogramWindow(ppTrial, function);
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
        JMenuItem functionDetailsItem = new JMenuItem("Show Function Bar Chart");
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

    public static JMenuItem createStatisticsMenuItem(String text, final ParaProfTrial ppTrial,
            final edu.uoregon.tau.dms.dss.Thread thread, final boolean userEvent) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                StatWindow statWindow = new StatWindow(ppTrial, thread, userEvent);
                statWindow.show();
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createStatisticsTableMenuItem(String text, final ParaProfTrial ppTrial,
            final edu.uoregon.tau.dms.dss.Thread thread) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                TreeTableWindow ttWindow = new TreeTableWindow(ppTrial, thread);
                ttWindow.show();
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createCallGraphMenuItem(String text, final ParaProfTrial ppTrial,
            final edu.uoregon.tau.dms.dss.Thread thread) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                CallGraphWindow tmpRef = new CallGraphWindow(ppTrial, thread);
                tmpRef.show();
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createCallPathThreadRelationMenuItem(String text, final ParaProfTrial ppTrial,
            final edu.uoregon.tau.dms.dss.Thread thread) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                CallPathTextWindow callPathTextWindow = new CallPathTextWindow(ppTrial, thread, 0);
                callPathTextWindow.show();
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createThreadDataMenuItem(String text, final ParaProfTrial ppTrial,
            final edu.uoregon.tau.dms.dss.Thread thread) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                ThreadDataWindow w = new ThreadDataWindow(ppTrial, thread);
                w.show();
            }

        });
        return jMenuItem;
    }

    public static void handleThreadClick(final ParaProfTrial ppTrial, final edu.uoregon.tau.dms.dss.Thread thread, JPanel owner,
            MouseEvent evt) {
        if (thread.getNodeID() == -1) { // mean
            JPopupMenu meanThreadPopup = new JPopupMenu();
            meanThreadPopup.add(createThreadDataMenuItem("Show Mean Bar Chart", ppTrial, thread));
            meanThreadPopup.add(createStatisticsMenuItem("Show Mean Statistics Text Window", ppTrial, thread, false));
            meanThreadPopup.add(createStatisticsTableMenuItem("Show Mean Statistics Table", ppTrial, thread));
            meanThreadPopup.add(createCallGraphMenuItem("Show Mean Call Graph", ppTrial, thread));
            meanThreadPopup.add(createCallPathThreadRelationMenuItem("Show Mean Call Path Relations", ppTrial, thread));
            meanThreadPopup.show(owner, evt.getX(), evt.getY());
        } else if (thread.getNodeID() == -3) { // stddev
            JPopupMenu threadPopup = new JPopupMenu();
            threadPopup.add(createThreadDataMenuItem("Show Standard Deviation Bar Chart", ppTrial, thread));
            threadPopup.add(createStatisticsMenuItem("Show Standard Deviation Statistics Text Window", ppTrial, thread, false));
            threadPopup.add(createStatisticsTableMenuItem("Show Standard Deviation Statistics Table", ppTrial, thread));
            threadPopup.add(createCallGraphMenuItem("Show Standard Deviation Call Graph", ppTrial, thread));
            threadPopup.add(createCallPathThreadRelationMenuItem("Show Standard Deviation Call Path Thread Relations", ppTrial,
                    thread));
            threadPopup.show(owner, evt.getX(), evt.getY());
        } else {
            JPopupMenu threadPopup = new JPopupMenu();
            threadPopup.add(createThreadDataMenuItem("Show Thread Bar Chart", ppTrial, thread));
            threadPopup.add(createStatisticsMenuItem("Show Thread Statistics Text Window", ppTrial, thread, false));
            threadPopup.add(createStatisticsTableMenuItem("Show Thread Statistics Table", ppTrial, thread));
            threadPopup.add(createCallGraphMenuItem("Show Thread Call Graph", ppTrial, thread));
            threadPopup.add(createCallPathThreadRelationMenuItem("Show Thread Call Path Relations", ppTrial, thread));
            if (ppTrial.userEventsPresent()) {
                threadPopup.add(createStatisticsMenuItem("Show User Event Statistics Window", ppTrial, thread, true));
            }
            threadPopup.show(owner, evt.getX(), evt.getY());

        }
    }

    public static int[] computeClipping(Rectangle clipRect, Rectangle viewRect, boolean toScreen, boolean fullWindow, int size,
            int barSpacing, int yCoord) {

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

    public static JMenu createUnitsMenu(final UnitListener unitListener, int initialUnits, boolean doHours) {

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

            List thisGroups = function.getGroups();
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

        p.writeInt(dataSource.getAllThreads().size());

        for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
            edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it.next();

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

    private static void writeMetric(String root, DataSource dataSource, int metricID, Function[] functions,
            String[] groupStrings, UserEvent[] userEvents) throws IOException {

        int numFunctions = dataSource.getNumFunctions();
        int numMetrics = dataSource.getNumberOfMetrics();
        int numUserEvents = dataSource.getNumUserEvents();
        int numGroups = dataSource.getNumGroups();

        for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
            edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it.next();

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

            if (dataSource.getNumberOfMetrics() == 1 && dataSource.getMetricName(metricID).equals("Time")) {
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
                    bw.write((int) fp.getNumCalls() + " ");
                    bw.write((int) fp.getNumSubr() + " ");
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

            if (count > 0) {
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
            }
            bw.close();
            outWriter.close();
            out.close();
        }

    }

    public static String createSafeMetricName(String name) {
        String ret = name.replace('/', '\\');
        return ret;
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

            List thisGroups = function.getGroups();

            if (thisGroups == null) {
                groupStrings[idx] = "";
            } else {
                groupStrings[idx] = "";

                for (int i = 0; i < thisGroups.size(); i++) {
                    Group group = (Group) thisGroups.get(i);
                    groupStrings[idx] = groupStrings[idx] + " | " + group.getName();
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
                String name = "MULTI__" + createSafeMetricName(dataSource.getMetricName(i));
                boolean success = (new File(name).mkdir());
                if (!success) {
                    System.err.println("Failed to create directory: " + name);
                } else {
                    writeMetric(name, dataSource, i, functions, groupStrings, userEvents);
                }
            }
        }

    }
            
    public static boolean rightClick(MouseEvent evt) {
        if ((evt.getModifiers() & InputEvent.BUTTON3_MASK) != 0) {
            return true;
        }
        return false;
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
