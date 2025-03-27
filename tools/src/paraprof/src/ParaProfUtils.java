package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.InputEvent;
import java.awt.event.MouseEvent;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.awt.print.PrinterJob;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.net.URL;
import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Observer;

import javax.swing.ButtonGroup;
import javax.swing.ImageIcon;
import javax.swing.JColorChooser;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPopupMenu;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.JSeparator;
import javax.swing.JTextArea;
import javax.swing.JToolBar;
import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import edu.uoregon.tau.common.ExternalTool;
import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.common.VectorExport;
import edu.uoregon.tau.paraprof.barchart.BarChart;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.SortListener;
import edu.uoregon.tau.paraprof.interfaces.ToolBarListener;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;
import edu.uoregon.tau.paraprof.script.ParaProfFunctionScript;
import edu.uoregon.tau.paraprof.script.ParaProfScript;
import edu.uoregon.tau.paraprof.script.ParaProfTrialScript;
import edu.uoregon.tau.paraprof.sourceview.SourceViewer;
import edu.uoregon.tau.paraprof.treetable.ContextEventWindow;
import edu.uoregon.tau.paraprof.treetable.TreeTableWindow;
import edu.uoregon.tau.paraprof.util.MapViewer;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DataSourceExport;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.PhaseConvertedDataSource;
import edu.uoregon.tau.perfdmf.Snapshot;
import edu.uoregon.tau.perfdmf.SourceRegion;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UserEvent;
import edu.uoregon.tau.perfdmf.UserEventProfile;
import edu.uoregon.tau.perfdmf.UtilFncs;
import edu.uoregon.tau.vis.HeatMapWindow;

/**
 * Utility class for ParaProf
 * 
 * <P>
 * CVS $Id: ParaProfUtils.java,v 1.62 2010/05/01 00:26:33 amorris Exp $
 * </P>
 * 
 * @author Alan Morris
 * @version $Revision: 1.62 $
 */
public class ParaProfUtils {

    static boolean verbose;
    static boolean verboseSet;

    // Suppress default constructor for noninstantiability
    private ParaProfUtils() {
    // This constructor will never be invoked
    }

    public static FunctionBarChartWindow createFunctionBarChartWindow(ParaProfTrial ppTrial, Function function, Component parent) {
        return new FunctionBarChartWindow(ppTrial, function, parent);
    }

    public static FunctionBarChartWindow createFunctionBarChartWindow(ParaProfTrial ppTrial, Thread thread, Function phase,
            Component parent) {
        return new FunctionBarChartWindow(ppTrial, thread, phase, parent);
    }

    public static LedgerWindow createLedgerWindow(ParaProfTrial ppTrial, int windowType) {
        return new LedgerWindow(ppTrial, windowType, null);
    }

    public static LedgerWindow createLedgerWindow(ParaProfTrial ppTrial, int windowType, Component parent) {
        return new LedgerWindow(ppTrial, windowType, parent);
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

    public static void print(Printable printable) {
        PrinterJob job = PrinterJob.getPrinterJob();
        PageFormat defaultFormat = job.defaultPage();
        PageFormat selectedFormat = job.pageDialog(defaultFormat);

        if (defaultFormat != selectedFormat) { // only proceed if the user did not select cancel

            job.setPrintable(printable, selectedFormat);
            if (job.printDialog()) { // only proceed if the user did not select cancel
                try {
                    job.print();
                } catch (PrinterException e) {
                    ParaProfUtils.handleException(e);
                }
            }
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
                            ImageIcon icon = Utility.getImageIconResource("tau-medium.png");
                            JOptionPane.showMessageDialog(owner, ParaProf.getInfoString(), "About ParaProf",
                                    JOptionPane.INFORMATION_MESSAGE, icon);
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

                    String arg = evt.getActionCommand();

                    if (arg.equals("Print")) {
                        ParaProfUtils.print(printable);
                    } else if (arg.equals("Preferences...")) {
                        // this is just in case there is ever a ParaProfWindow that is not a JFrame (there shouldn't be)
                        ParaProf.preferencesWindow.showPreferencesWindow(window instanceof JFrame ? (JFrame) window : null);
                    } else if (arg.equals("Save as Bitmap Image")) {

                        if (panel instanceof ImageExport) {
                            ParaProfImageOutput.saveImage((ImageExport) panel);
                        } else if (panel instanceof ThreeDeeImageProvider) {
                            ParaProfImageOutput.save3dImage((ThreeDeeImageProvider) panel);
                        } else {
                            throw new ParaProfException("Don't know how to \"Save Image\" for " + panel.getClass());
                        }

                    } else if (arg.equals("Save as Vector Graphics")) {
                        if (panel instanceof HeatMapWindow) {
                            JOptionPane.showMessageDialog(window.getFrame(), "Can't save heat map as vector graphics");
                        } else if (panel instanceof ThreeDeeImageProvider) {
                            JOptionPane.showMessageDialog(window.getFrame(), "Can't save 3D visualization as vector graphics");
                        } else if (panel instanceof SourceViewer) {
                            JOptionPane.showMessageDialog(window.getFrame(), "Can't save Source Viewer as vector graphics");
                        } else if (panel instanceof ImageExport) {
                            VectorExport.promptForVectorExport((ImageExport) panel, "ParaProf");
                        } else {
                            throw new ParaProfException("Don't know how to \"Save as Vector Graphics\" for " + panel.getClass());
                        }
                    } else if (arg.equals("Close This Window")) {
                        window.closeThisWindow();
                    } else if (arg.equals("Exit ParaProf!")) {
                        ParaProf.exitParaProf(0);
                    } else if (arg.equals("Export Profile")) {
                        GlobalDataWindow gdw = (GlobalDataWindow) window;
                        ParaProfUtils.exportTrial(gdw.getPpTrial(), gdw);
                    } else if (arg.equals("Convert to Phase Profile")) {
                        GlobalDataWindow gdw = (GlobalDataWindow) window;
                        ParaProfUtils.phaseConvertTrial(gdw.getPpTrial(), gdw);
                    } else if (arg.equals("Create Selective Instrumentation File")) {
                        GlobalDataWindow gdw = (GlobalDataWindow) window;
                        SelectiveFileGenerator.showWindow(gdw.getPpTrial(), gdw);
                    }

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JMenu fileMenu = new JMenu("File");

        if (window instanceof GlobalDataWindow) {
            JMenuItem jMenuItem;
            jMenuItem = new JMenuItem("Export Profile");
            jMenuItem.addActionListener(actionListener);
            fileMenu.add(jMenuItem);
            jMenuItem = new JMenuItem("Convert to Phase Profile");
            jMenuItem.addActionListener(actionListener);
            fileMenu.add(jMenuItem);
            jMenuItem = new JMenuItem("Create Selective Instrumentation File");
            jMenuItem.addActionListener(actionListener);
            fileMenu.add(jMenuItem);
            jMenuItem = new JMenuItem("Add Mean to Comparison Window");
            jMenuItem.addActionListener(actionListener);
            fileMenu.add(jMenuItem);
            fileMenu.add(new JSeparator());
        }

        JMenu subMenu = new JMenu("Save ...");
        subMenu.getPopupMenu().setLightWeightPopupEnabled(false);

        JMenuItem menuItem = new JMenuItem("Save as Bitmap Image");
        menuItem.addActionListener(actionListener);
        subMenu.add(menuItem);
        menuItem = new JMenuItem("Save as Vector Graphics");
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

    private static JMenuItem createMenuItem(String text, ActionListener actionListener, boolean enabled) {
        JMenuItem menuItem = new JMenuItem(text);
        menuItem.setEnabled(enabled);
        menuItem.addActionListener(actionListener);
        return menuItem;
    }

    public static JMenu createScriptMenu(final ParaProfTrial ppTrial, final JFrame owner) {

        final JMenu menu = new JMenu("PyScript");

        menu.addMenuListener(new MenuListener() {

            public void menuCanceled(MenuEvent e) {
            // TODO Auto-generated method stub

            }

            public void menuDeselected(MenuEvent e) {
            // TODO Auto-generated method stub

            }

            public void menuSelected(MenuEvent e) {
                menu.removeAll();
                JMenuItem menuitem = new JMenuItem("Reload scripts");
                menuitem.addActionListener(new ActionListener() {

                    public void actionPerformed(ActionEvent e) {
                        ParaProf.loadScripts();
                    }
                });
                menu.add(menuitem);
                //menu.add
                for (int i = 0; i < ParaProf.scripts.size(); i++) {
                    final ParaProfScript pps = ParaProf.scripts.get(i);
                    if (pps instanceof ParaProfTrialScript) {
                        JMenuItem menuItem = new JMenuItem("[Script] " + pps.getName());
                        menuItem.addActionListener(new ActionListener() {
                            public void actionPerformed(ActionEvent e) {
                                try {
                                    ((ParaProfTrialScript) pps).run(ppTrial);
                                } catch (Exception ex) {
                                    new ParaProfErrorDialog("Exception while executing script:", ex);
                                }
                            }
                        });
                        menu.add(menuItem);
                    }
                }

            }
        });

        return menu;
    }

    public static void showCommMatrix(ParaProfTrial ppTrial, JFrame parentFrame) {
        JFrame window = CommunicationMatrixWindow.createCommunicationMatrixWindow(ppTrial, parentFrame);
        if (window != null) {
            window.setVisible(true);
        }
    }

    public static void show3dCommMatrix(ParaProfTrial ppTrial, JFrame parentFrame) {
    	
        JFrame window = null; 
        try {
        	window = ThreeDeeCommMatrixWindow.createCommunicationMatrixWindow(ppTrial, parentFrame);
        } 
        catch(NoClassDefFoundError e){
        	printLibjoglError(parentFrame,e);
        }
        
        catch (UnsatisfiedLinkError e) {
        	printLibjoglError(parentFrame,e);
        } catch (UnsupportedClassVersionError e) {
            JOptionPane.showMessageDialog(parentFrame,
                    "Unsupported class version.  Are you sure you're using Java 1.4 or above?");
        } catch (Exception gle) {
            new ParaProfErrorDialog("Unable to initialize OpenGL: ", gle);
        }
        
        
        if (window != null) {
            window.setVisible(true);
        }
    }
    
    public static void show3dVisualization(ParaProfTrial ppTrial,JFrame parentFrame){
        if (JVMDependent.version.equals("1.3")) {
            JOptionPane.showMessageDialog(parentFrame, "3D Visualization requires Java 1.4 or above\n"
                    + "Please make sure Java 1.4 is in your path, then reconfigure TAU and re-run ParaProf");
            return;
        }

        //Gears.main(null);
        //(new Gears()).show();

        JFrame window = null;
        
        try {
            window = ThreeDeeWindow.createThreeDeeWindow(ppTrial, parentFrame);//new ThreeDeeWindow(ppTrial, parentFrame);
        } 
        
        catch(NoClassDefFoundError e){
        	printLibjoglError(parentFrame,e);
        }
        
        catch (UnsatisfiedLinkError e) {
           printLibjoglError(parentFrame,e);
        } catch (UnsupportedClassVersionError e) {
            JOptionPane.showMessageDialog(parentFrame,
                    "Unsupported class version.  Are you sure you're using Java 1.4 or above?");
        } catch (Exception gle) {
            new ParaProfErrorDialog("Unable to initialize OpenGL: ", gle);
        }

        if (window != null) {
            window.setVisible(true);
        }
        
    }
    
    private static void printLibjoglError(JFrame parentFrame, Error e){
    	
    	String lowerOS = System.getProperty("os.name").toLowerCase();
    	String macline="";
    	if(lowerOS.indexOf("mac")>=0){
    		macline="You are on a 64 bit Mac and thus need to install 32 bit compatable Java 6, available here:\n http://tau.uoregon.edu/java.dmg\n\n";
    	}
    	
    	String text = "Unable to load jogl library.  Possible reasons:\n"
    			+ macline
                + "libjogl.so is not in your LD_LIBRARY_PATH.\n"
                + "Jogl is not built for this platform.\nOpenGL is not installed\n\n"
                + "Jogl is available at jogl.dev.java.net\n\n" + "Message : " + e.getMessage();
    	
    	JTextArea area = new JTextArea(text);
    	 JOptionPane.showMessageDialog(parentFrame, area);
    }

	 private static void refreshWindowsMenu(final JMenu windowsMenu, final ParaProfTrial pptrial) {
	        // Remove only previously added window items
	        for (int i = windowsMenu.getItemCount() - 1; i >= 0; i--) {
	            JMenuItem item = windowsMenu.getItem(i);
	            if (item != null && "DYNAMIC_WINDOW_ITEM".equals(item.getActionCommand())) {
	                windowsMenu.remove(i);
	            }
	        }

	     // Create a copy to avoid ConcurrentModificationException
	        List<Observer> observers = new ArrayList<Observer>(pptrial.getObservers());

	        for (int i = 0; i< observers.size(); i++) {
	        	final Object obs = observers.get(i);
	            if (obs instanceof JFrame) {
	                final JFrame window = (JFrame) obs;
	                
	               if(!window.isShowing()) {
	            	   pptrial.deleteObserver((Observer)obs);
	            	   continue;
	               }
	                
	                JMenuItem windowItem = new JMenuItem(window.getTitle());

	                windowItem.setActionCommand("DYNAMIC_WINDOW_ITEM"); // Mark it for future removal
	                windowItem.addActionListener(new ActionListener() {
	                	public void actionPerformed(ActionEvent e) {
	                    window.toFront();
	                    window.requestFocus();
	                	}});

	                windowsMenu.add(windowItem);
	            }
	        }
	    }
    
    public static JMenu createWindowsMenu(final ParaProfTrial ppTrial, final JFrame owner) {

        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {

                try {

                    String arg = evt.getActionCommand();

                    if (arg.equals("ParaProf Manager")) {
                        ParaProf.paraProfManagerWindow.setVisible(true);
                    } else if (arg.equals("Function Legend")) {
                        (new LedgerWindow(ppTrial, 0, owner)).setVisible(true);
                    } else if (arg.equals("Group Changer")) {
                        GroupChangerWindow gcw = GroupChangerWindow.createGroupChangerWindow(ppTrial, owner);
                        gcw.setVisible(true);
                    } else if (arg.equals("Group Legend")) {
                        (new LedgerWindow(ppTrial, 1, owner)).setVisible(true);
                    } else if (arg.equals("User Event Legend")) {
                        (new LedgerWindow(ppTrial, 2, owner)).setVisible(true);
                    } else if (arg.equals("Phase Legend")) {
                        (new LedgerWindow(ppTrial, 3, owner)).setVisible(true);
                    } else if (arg.equals("Snapshot Controller")) {
                        ppTrial.showSnapshotController();
                    } else if (arg.equals("Communication Matrix")) {
                        ParaProfUtils.showCommMatrix(ppTrial, owner);
                    } else if (arg.equals("3D Communication Matrix")) {
                        ParaProfUtils.show3dCommMatrix(ppTrial, owner);
                    } else if (arg.equals("3D Visualization")) {
                    	ParaProfUtils.show3dVisualization(ppTrial, owner);
                    } else if (arg.equals("Close All Sub-Windows")) {
                        ppTrial.updateRegisteredObjects("subWindowCloseEvent");
                    }

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        final JMenu windowsMenu = new JMenu("Windows");

        JMenuItem menuItem;

        menuItem = new JMenuItem("ParaProf Manager");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        windowsMenu.add(new JSeparator());

        menuItem = new JMenuItem("3D Visualization");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        // still in development
        menuItem = new JMenuItem("3D Communication Matrix");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Communication Matrix");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        if (ppTrial.getDataSource().getWellBehavedSnapshots()) {
            menuItem = new JMenuItem("Snapshot Controller");
            menuItem.addActionListener(actionListener);
            windowsMenu.add(menuItem);
        }

        //menuItem = new JMenuItem("Call Path Relations");
        //menuItem.addActionListener(actionListener);
        //windowsMenu.add(menuItem);

        windowsMenu.add(new JSeparator());

        ActionListener fActionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                FunctionSelectorDialog fSelector = new FunctionSelectorDialog(owner, true,
                        ppTrial.getDataSource().getFunctionIterator(), null, false, false);
                if (fSelector.choose()) {
                    Function selectedFunction = (Function) fSelector.getSelectedObject();

                    String arg = evt.getActionCommand();

                    if (arg.equals("Bar Chart")) {
                        FunctionBarChartWindow w = new FunctionBarChartWindow(ppTrial, selectedFunction, owner);
                        w.setVisible(true);
                    } else if (arg.equals("Histogram")) {
                        HistogramWindow w = new HistogramWindow(ppTrial, selectedFunction, owner);
                        w.setVisible(true);
                    }
                }
            }
        };

        final JMenu functionWindows = new JMenu("Function");
        functionWindows.getPopupMenu().setLightWeightPopupEnabled(false);

        functionWindows.add(createMenuItem("Bar Chart", fActionListener, true));
        functionWindows.add(createMenuItem("Histogram", fActionListener, true));
        windowsMenu.add(functionWindows);

        ActionListener tActionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                String arg = evt.getActionCommand();

                List<Thread> list = new ArrayList<Thread>(ppTrial.getDataSource().getAllThreads());
                if (ppTrial.getDataSource().getAllThreads().size() > 1 && arg.equals("User Event Statistics") == false) {
                    list.add(0, ppTrial.getDataSource().getStdDevData());
                    list.add(1, ppTrial.getDataSource().getMeanData());
                }

                FunctionSelectorDialog fSelector = new FunctionSelectorDialog(owner, true, list.iterator(), null, false, false);
                fSelector.setTitle("Select a Thread");
                if (fSelector.choose()) {
                    Thread selectedThread = (Thread) fSelector.getSelectedObject();

                    if (arg.equals("Bar Chart")) {
                        FunctionBarChartWindow w = new FunctionBarChartWindow(ppTrial, selectedThread, null, owner);
                        w.setVisible(true);
                    } else if (arg.equals("Statistics Text")) {
                        (new StatWindow(ppTrial, selectedThread, false, null, owner)).setVisible(true);
                    } else if (arg.equals("Statistics Table")) {
                        (new TreeTableWindow(ppTrial, selectedThread, owner)).setVisible(true);
                    } else if (arg.equals("Call Graph")) {
                        (new CallGraphWindow(ppTrial, selectedThread, owner)).setVisible(true);
                    } else if (arg.equals("Call Path Relations")) {
                        (new CallPathTextWindow(ppTrial, selectedThread, owner)).setVisible(true);
                    } else if (arg.equals("Context Event Window")) {
                        (new ContextEventWindow(ppTrial, selectedThread, owner)).setVisible(true);
                    } else if (arg.equals("User Event Bar Chart")) {
                        (new UserEventWindow(ppTrial, selectedThread, owner)).setVisible(true);
                    } else if (arg.equals("User Event Statistics")) {
                        (new StatWindow(ppTrial, selectedThread, true, null, owner)).setVisible(true);
                    }
                }
            }
        };

        final JMenu threadWindows = new JMenu("Thread");
        threadWindows.getPopupMenu().setLightWeightPopupEnabled(false);
        threadWindows.add(createMenuItem("Bar Chart", tActionListener, true));
        threadWindows.add(createMenuItem("Statistics Text", tActionListener, true));
        threadWindows.add(createMenuItem("Statistics Table", tActionListener, true));
        threadWindows.add(createMenuItem("Call Graph", tActionListener, ppTrial.callPathDataPresent()));
        threadWindows.add(createMenuItem("Call Path Relations", tActionListener, ppTrial.callPathDataPresent()));
        threadWindows.add(createMenuItem("Context Event Window", tActionListener, ppTrial.userEventsPresent()));
        threadWindows.add(createMenuItem("User Event Bar Chart", tActionListener, ppTrial.userEventsPresent()));
        threadWindows.add(createMenuItem("User Event Statistics", tActionListener, ppTrial.userEventsPresent()));

        windowsMenu.add(threadWindows);

	/*
        ActionListener mActionListener = new ActionListener() {
        	public void actionPerformed(ActionEvent evt) {
        		String arg = evt.getActionCommand();
        		if (arg.equals("Summary")) {
        			JFrame hw = 
        				MonSummaryWindow.createMonSummaryWindow(ppTrial, 
        						owner);
        			hw.setVisible(true);

        		} else if (arg.equals("Clusters")) {
        			System.out.println("Clusters");
        		}
        	}
        };
        final JMenu monWindows = new JMenu("Monitoring");
        monWindows.getPopupMenu().setLightWeightPopupEnabled(false);
        monWindows.add(createMenuItem("Summary", mActionListener, true));
        monWindows.add(createMenuItem("Clusters", mActionListener, true));

        ActionListener fmActionListener = new ActionListener() {
        	public void actionPerformed(ActionEvent evt) {	
        		FunctionSelectorDialog fSelector = new FunctionSelectorDialog(owner, true,
        				ppTrial.getDataSource().getFunctions(), null, false, false);
        		if (fSelector.choose()) {
        			//Function selectedFunction = (Function) fSelector.getSelectedObject();

        			String arg = evt.getActionCommand();
        			if (arg.equals("Base Statistics")) {
        				System.out.println("Monitoring: Base Statistics");
       				// JFrame hw = MonStatisticsWindow.createMonStatisticsWindow(ppTrial, selectedFunction, owner);
//        				hw.setVisible(true);
        			} else if (arg.equals("Histograms")) {
        				System.out.println("Monitoring: Histograms");
        			}
        		}
        	}
	    };

	    final JMenu monFunctionWindows = new JMenu("Function");
	    monFunctionWindows.getPopupMenu().setLightWeightPopupEnabled(false);
	    monFunctionWindows.add(createMenuItem("Base Statistics",
	    		fmActionListener, true));
	    monFunctionWindows.add(createMenuItem("Histograms",
	    		fmActionListener, true));
	    monWindows.add(monFunctionWindows);

	    windowsMenu.add(monWindows);
	*/
        windowsMenu.add(new JSeparator());

        menuItem = new JMenuItem("Function Legend");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);

        final JMenuItem groupLedger = new JMenuItem("Group Legend");
        groupLedger.addActionListener(actionListener);
        windowsMenu.add(groupLedger);

        final JMenuItem userEventLedger = new JMenuItem("User Event Legend");
        userEventLedger.addActionListener(actionListener);
        windowsMenu.add(userEventLedger);

        if (ppTrial.getDataSource().getPhasesPresent()) {
            final JMenuItem phaseLedger = new JMenuItem("Phase Legend");
            phaseLedger.addActionListener(actionListener);
            windowsMenu.add(phaseLedger);
        }

        final JMenuItem groupRemapper = new JMenuItem("Group Changer");
        groupRemapper.addActionListener(actionListener);
        windowsMenu.add(groupRemapper);

        windowsMenu.add(new JSeparator());

        menuItem = new JMenuItem("Close All Sub-Windows");
        menuItem.addActionListener(actionListener);
        windowsMenu.add(menuItem);
        windowsMenu.add(new JSeparator());
        MenuListener menuListener = new MenuListener() {
            public void menuSelected(MenuEvent evt) {
                try {
                    groupLedger.setEnabled(ppTrial.groupNamesPresent());
                    userEventLedger.setEnabled(ppTrial.userEventsPresent());
                    refreshWindowsMenu(windowsMenu,ppTrial);
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

            public void menuCanceled(MenuEvent e) {}

            public void menuDeselected(MenuEvent e) {}
        };

        windowsMenu.addMenuListener(menuListener);

        return windowsMenu;
    }

    public static void scaleForPrint(Graphics g, PageFormat pageFormat, int width, int height) {
        double pageWidth = pageFormat.getImageableWidth();
        double pageHeight = pageFormat.getImageableHeight();
        //int cols = (int) (width / pageWidth) + 1;
        //int rows = (int) (height / pageHeight) + 1;
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
    
    private static SourceRegion getBacktraceSourceLink(String btTuple){
    	SourceRegion sourceLink = new SourceRegion();
    	
    	int fb = btTuple.indexOf('[',1);
    	if(fb==-1)
    		return null;
    	if(fb==1)
    		fb = btTuple.indexOf('[',fb+1);
    	if(fb==-1)
    		return null;
    	
    	int endb=btTuple.indexOf(']',fb+1);
    	if(endb==-1)
    		return null;
    	
    	String src = btTuple.substring(fb+1, endb);
    	if(src==null)
    		return null;
    	
    	String[]chunks=src.split(":");
    	if(chunks.length!=2)
    		return null;
    	
    	if(chunks[0].equals("(unknown)"))
    		return null;
    	
    	sourceLink.setFilename(chunks[0]);
    	int line = Integer.parseInt(chunks[1]);
    	sourceLink.setStartLine(line);
    	sourceLink.setEndLine(line);
    	
    	return sourceLink;
    }
    
    public static JPopupMenu createMetadataClickPopUp(final String fName,
            final Component owner) {
    	
    	 final SourceRegion sr = getBacktraceSourceLink(fName);
    	
        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {

                    String arg = evt.getActionCommand();

                     if (arg.equals("Show Source Code")) {

                        if (ParaProf.controlMode) {
                            ExternalController.outputCommand("sourcecode " + sr);
                        } 
//                        else if (ParaProf.insideEclipse) {
//                            ParaProf.eclipseHandler.openSourceLocation(ppTrial, function);
//                        } 
                        else {
                            // use internal viewer
                            ParaProf.getDirectoryManager().showSourceCode(sr);
                        }
                    }

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JPopupMenu functionPopup = new JPopupMenu();

        
       
        
        if (sr != null) {
            JMenuItem functionDetailsItem = new JMenuItem("Show Source Code");
            functionDetailsItem.addActionListener(actionListener);
            functionPopup.add(functionDetailsItem);
        }
        else{
        	return null;
        }



        return functionPopup;

    }
    
    
    
    public static JMenuItem createContextSourcePopUp(final String fName) {
    	
    	 final SourceRegion sr = Function.getSourceLink(fName);
    	
        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {

                    String arg = evt.getActionCommand();

                     if (arg.equals("Show Source Code")) {

                        if (ParaProf.controlMode) {
                            ExternalController.outputCommand("sourcecode " + sr);
                        } 
//                        else if (ParaProf.insideEclipse) {
//                            ParaProf.eclipseHandler.openSourceLocation(ppTrial, function);
//                        } 
                        else {
                            // use internal viewer
                            ParaProf.getDirectoryManager().showSourceCode(sr);
                        }
                    }

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };


        
        JMenuItem contextSrcItem = null;
        
        if (sr != null&&sr.getFilename()!=null) {
            contextSrcItem = new JMenuItem("Show Source Code");
            contextSrcItem.addActionListener(actionListener);
        }
        



        return contextSrcItem;

    }
    
    

    public static JPopupMenu createFunctionClickPopUp(final ParaProfTrial ppTrial, final Function function, final Thread thread,
            final Component owner) {
        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {

                    String arg = evt.getActionCommand();

                    if (arg.equals("Show Function Bar Chart")) {
                        FunctionBarChartWindow functionDataWindow = new FunctionBarChartWindow(ppTrial, function, owner);
                        functionDataWindow.setVisible(true);
                    } else if (arg.equals("Show Function Data over Phases")) {
                        FunctionBarChartWindow functionBarChartWindow = FunctionBarChartWindow.CreateFunctionsOverPhaseDisplay(
                                ppTrial, function, thread, owner);
                        functionBarChartWindow.setVisible(true);
                    } else if (arg.equals("Show Function Histogram")) {
                        String dagstuhl_demo = (String) ppTrial.getDataSource().getMetaData().get(
                                "TAU Internal Profile Attribute");
                        if (dagstuhl_demo != null && dagstuhl_demo.equals("collate_dump_dagstuhl")) {
                            JFrame hw = PreCompHistogramWindow.createHistogramWindow(ppTrial, function, thread, owner);
                            hw.setVisible(true);
                        } else {
                            HistogramWindow hw = new HistogramWindow(ppTrial, function, owner);
                            hw.setVisible(true);
                        }
                    } else if (arg.equals("Assign Function Color")) {
                        ParaProf.colorMap.assignColor(owner, function);
                    } else if (arg.equals("Rename")) {

                        String newName = (String) JOptionPane.showInputDialog(owner, "Rename event:", "Rename event",
                                JOptionPane.PLAIN_MESSAGE, null, null, function.toString());

                        if (newName != null) {
                            ppTrial.getDataSource().renameFunction(function, newName);
                            ppTrial.updateRegisteredObjects("dataEvent");
                        }

                    } else if (arg.equals("Reset to Default Color")) {
                        ParaProf.colorMap.removeColor(function);
                        ParaProf.colorMap.reassignColors();
                    } else if (arg.equals("Open Profile for this Phase")) {
                        GlobalDataWindow fdw = new GlobalDataWindow(ppTrial, function.getActualPhase());
                        fdw.setVisible(true);
                        ParaProf.incrementNumWindows();
                    } else if (arg.equals("Show Source Code")) {

                        if (ParaProf.controlMode) {
                            ExternalController.outputCommand("sourcecode " + function.getSourceLink());
                        } else if (ParaProf.insideEclipse) {
                            ParaProf.eclipseHandler.openSourceLocation(ppTrial, function);
                        } else {
                            // use internal viewer
                            ParaProf.getDirectoryManager().showSourceCode(function.getSourceLink());
                        }
                    } else if (arg.equals("Launch External Tool for this Function & Metric")) {
                        String metricName = "TIME";
                        if (owner instanceof BarChart) {
                            BarChart tmp = (BarChart) owner;
                            metricName = tmp.getBarChartModel().getDataSorter().getSelectedMetric().getName();
                        }
                        List<ExternalTool> tools = ExternalTool.findMatchingTools(ppTrial.getTrial().getMetaData().get(
                                DataSource.FILE_TYPE_NAME), ppTrial.getTrial().getName());
                        ExternalTool.CommandParameters params = new ExternalTool.CommandParameters();
                        params.function = function.getName();
                        params.metric = metricName;
                        params.nodeID = thread.getNodeID();
                        params.threadID = thread.getThreadID();
                        MetaDataMap map = new MetaDataMap();
                        map.putAll(thread.getMetaData());
                        map.putAll(ppTrial.getDataSource().getMetaData());
                        params.metadata = map;
                        ExternalTool.launch(tools, params, owner);
                    } else if (arg.equals("Show In Statistics Table")){
                    	TreeTableWindow ttWindow = new TreeTableWindow(ppTrial, thread, owner);
                    	ttWindow.select(function);
                        ttWindow.setVisible(true);
                    }

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }

        };

        JPopupMenu functionPopup = new JPopupMenu();

        //if (ParaProf.insideEclipse && function.getSourceLink().getFilename() != null) {
        if (function.getSourceLink().getFilename() != null) {
            JMenuItem functionDetailsItem = new JMenuItem("Show Source Code");
            functionDetailsItem.addActionListener(actionListener);
            functionPopup.add(functionDetailsItem);
        }

        if (function.isPhase()) {
            JMenuItem functionDetailsItem = new JMenuItem("Open Profile for this Phase");
            functionDetailsItem.addActionListener(actionListener);
            functionPopup.add(functionDetailsItem);
        }

        JMenuItem functionStatisticsItem = new JMenuItem("Show In Statistics Table");
        functionStatisticsItem.addActionListener(actionListener);
        functionPopup.add(functionStatisticsItem);

        JMenuItem functionHistogramItem = new JMenuItem("Show Function Histogram");
        functionHistogramItem.addActionListener(actionListener);
        functionPopup.add(functionHistogramItem);

        JMenuItem functionDetailsItem = new JMenuItem("Show Function Bar Chart");
        functionDetailsItem.addActionListener(actionListener);
        functionPopup.add(functionDetailsItem);
        
        if (ppTrial.getDataSource().getPhasesPresent()) {
            JMenuItem jMenuItem = new JMenuItem("Show Function Data over Phases");
            jMenuItem.addActionListener(actionListener);
            functionPopup.add(jMenuItem);
        }

        JMenuItem jMenuItem = new JMenuItem("Assign Function Color");
        jMenuItem.addActionListener(actionListener);
        functionPopup.add(jMenuItem);

        jMenuItem = new JMenuItem("Reset to Default Color");
        jMenuItem.addActionListener(actionListener);
        functionPopup.add(jMenuItem);

        if (!function.isGroupMember("TAU_CALLPATH")) {
            jMenuItem = new JMenuItem("Rename");
            jMenuItem.addActionListener(actionListener);
            functionPopup.add(jMenuItem);
        }

        if ((thread.getNodeID() >= 0)
                && (ExternalTool.matchingToolExists(ppTrial.getTrial().getMetaData().get(DataSource.FILE_TYPE_NAME),
                        (String) ppTrial.getTrial().getName()))) {
            JMenuItem toolMenuItem = new JMenuItem("Launch External Tool for this Function & Metric");
            toolMenuItem.addActionListener(actionListener);
            functionPopup.add(toolMenuItem);
        }

        // count function scripts
        int functionScripts = 0;
        for (int i = 0; i < ParaProf.scripts.size(); i++) {
            ParaProfScript pps = ParaProf.scripts.get(i);
            if (pps instanceof ParaProfFunctionScript) {
                functionScripts++;
            }
        }
        if (functionScripts > 1) {
            functionPopup.add(new JSeparator());
        }
        for (int i = 0; i < ParaProf.scripts.size(); i++) {
            final ParaProfScript pps = ParaProf.scripts.get(i);
            if (pps instanceof ParaProfFunctionScript) {
                JMenuItem menuItem = new JMenuItem("[Script] " + pps.getName());
                menuItem.addActionListener(new ActionListener() {
                    public void actionPerformed(ActionEvent e) {
                        try {
                            ((ParaProfFunctionScript) pps).runFunction(ppTrial, function);
                        } catch (Exception ex) {
                            new ParaProfErrorDialog("Exception while executing script: ", ex);
                        }

                    }
                });
                functionPopup.add(menuItem);
            }
        }

        return functionPopup;

    }

    public static JMenuItem createStatisticsMenuItem(String text, final ParaProfTrial ppTrial, final Function phase,
            final Thread thread, final boolean userEvent, final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                StatWindow statWindow = new StatWindow(ppTrial, thread, userEvent, phase, owner);
                statWindow.setVisible(true);
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createStatisticsTableMenuItem(String text, final ParaProfTrial ppTrial, final Function phase,
            final Thread thread, final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                TreeTableWindow ttWindow = new TreeTableWindow(ppTrial, thread, owner);
                ttWindow.setVisible(true);
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createCallGraphMenuItem(String text, final ParaProfTrial ppTrial, final Thread thread,
            final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                CallGraphWindow tmpRef = new CallGraphWindow(ppTrial, thread, owner);
                tmpRef.setVisible(true);
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createCallPathThreadRelationMenuItem(String text, final ParaProfTrial ppTrial, final Thread thread,
            final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                CallPathTextWindow callPathTextWindow = new CallPathTextWindow(ppTrial, thread, owner);
                callPathTextWindow.setVisible(true);
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createThreadDataMenuItem(String text, final ParaProfTrial ppTrial, final Function phase,
            final Thread thread, final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                FunctionBarChartWindow w = new FunctionBarChartWindow(ppTrial, thread, phase, owner);
                w.setVisible(true);
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createComparisonMenuItem(String text, final ParaProfTrial ppTrial, final Thread thread,
            final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                if (ParaProf.theComparisonWindow == null) {
                    ParaProf.theComparisonWindow = FunctionBarChartWindow.CreateComparisonWindow(ppTrial, thread, owner);
                } else {
                    ParaProf.theComparisonWindow.addThread(ppTrial, thread);
                }
                ParaProf.theComparisonWindow.setVisible(true);
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createSnapShotMenuItem(String text, final ParaProfTrial ppTrial, final Thread thread,
            final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //SnapshotThreadWindow w = new SnapshotThreadWindow(ppTrial, thread, owner);
                Frame w = new SnapshotBreakdownWindow(ppTrial, thread, owner);
                w.setVisible(true);
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createThreadMetaDataMenuItem(String text, final ParaProfTrial ppTrial, final Thread thread,
            final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                MetaDataMap map = new MetaDataMap();
                map.putAll(thread.getMetaData());
                map.putAll(ppTrial.getDataSource().getMetaData());
                Frame w = new MapViewer("Metadata for " + thread, map, ParaProf.preferences.getFont());
                w.setVisible(true);
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createUserEventBarChartMenuItem(String text, final ParaProfTrial ppTrial, final Thread thread,
            final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                UserEventWindow w = new UserEventWindow(ppTrial, thread, owner);
                w.setVisible(true);
            }

        });
        return jMenuItem;
    }

    public static JMenuItem createContextEventMenuItem(String text, final ParaProfTrial ppTrial, final Thread thread,
            final Component owner) {
        JMenuItem jMenuItem = new JMenuItem(text);
        jMenuItem.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {

            	Runnable bt = new Runnable(){

					public void run() {
						try{
						owner.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
						ContextEventWindow w = new ContextEventWindow(ppTrial, thread, owner);
		                w.setVisible(true);
						}finally{
							owner.setCursor(Cursor.getDefaultCursor());
		            	}
		                
						
					}
            		
            	};
            	new java.lang.Thread(bt).start();

            }
        });
        return jMenuItem;
    }

    public static void handleUserEventClick(final ParaProfTrial ppTrial, final UserEvent userEvent,  final JComponent owner,
            MouseEvent evt) {
    	handleUserEventClick(ppTrial, userEvent, null, owner, evt);
    }
    
    public static void handleUserEventClick(final ParaProfTrial ppTrial, final UserEvent userEvent, final Thread thread, final JComponent owner,
            MouseEvent evt) {
    	
        ActionListener act = new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                Object EventSrc = e.getSource();
                if (EventSrc instanceof JMenuItem) {
                    String arg = e.getActionCommand();
                    if (arg.equals("Show User Event Bar Chart")) {
                        UserEventWindow tmpRef = new UserEventWindow(ppTrial, userEvent, owner);
                        tmpRef.setVisible(true);
                    } else if (arg.equals("Change User Event Color")) {

                        Color tmpCol = userEvent.getColor();
                        tmpCol = JColorChooser.showDialog(owner, "Please select a new color", tmpCol);
                        if (tmpCol != null) {
                            userEvent.setSpecificColor(tmpCol);
                            userEvent.setColorFlag(true);
                            ppTrial.updateRegisteredObjects("colorEvent");
                        }
                    } else if (arg.equals("Reset to Generic Color")) {

                        userEvent.setColorFlag(false);
                        ppTrial.updateRegisteredObjects("colorEvent");
					} else if (arg.equals("Show/Hide Total")) {
						userEvent.setShowTotal(!userEvent.isShowTotal());
						ppTrial.updateRegisteredObjects("colorEvent");
                    }
                    else if (arg.equals("Show In Context Event Window")&&thread!=null) {

                    	//if(owner instanceof UserEventWindow){
                    		//UserEventWindow uew = (UserEventWindow)owner;
                    	
                    		ContextEventWindow cew = new ContextEventWindow( ppTrial,  thread,  owner);
                    		UserEventProfile uep = thread.getUserEventProfile(userEvent);
                    		cew.selectEvent(uep);
                    		cew.setVisible(true);
                    		//cew.selectEvent();
                    	//}
                    }
                }

            }
        };

        JPopupMenu popup = new JPopupMenu();
        JMenuItem menuItem;
        
        menuItem = new JMenuItem("Show In Context Event Window");
        menuItem.addActionListener(act);
        popup.add(menuItem);
        
        menuItem = new JMenuItem("Show User Event Bar Chart");
        menuItem.addActionListener(act);
        popup.add(menuItem);

        menuItem = new JMenuItem("Change User Event Color");
        menuItem.addActionListener(act);
        popup.add(menuItem);

        menuItem = new JMenuItem("Reset to Generic Color");
        menuItem.addActionListener(act);
        popup.add(menuItem);
        
		menuItem = new JMenuItem("Show/Hide Total");
		menuItem.addActionListener(act);
		popup.add(menuItem);

        menuItem = createContextSourcePopUp(userEvent.getName());
        if(menuItem!=null){
        	popup.add(menuItem);
        }

        popup.show(owner, evt.getX(), evt.getY());

    }

    public static void handleThreadClick(final ParaProfTrial ppTrial, final Function phase, final Thread thread,
            JComponent owner, MouseEvent evt) {

        String ident;

        if (thread.getNodeID() == -1 || thread.getNodeID() == -6) {
            ident = "Mean";
        } else if (thread.getNodeID() == -2) {
            ident = "Total";
        } else if (thread.getNodeID() == -3 || thread.getNodeID() == -7) {
            ident = "Standard Deviation";
        } else {
            ident = "Thread";
        }

        JPopupMenu threadPopup = new JPopupMenu();
        threadPopup.add(createThreadDataMenuItem("Show " + ident + " Bar Chart", ppTrial, phase, thread, owner));
        threadPopup.add(createStatisticsMenuItem("Show " + ident + " Statistics Text Window", ppTrial, phase, thread, false,
                owner));
        threadPopup.add(createStatisticsTableMenuItem("Show " + ident + " Statistics Table", ppTrial, phase, thread, owner));
        threadPopup.add(createCallGraphMenuItem("Show " + ident + " Call Graph", ppTrial, thread, owner));
        threadPopup.add(createCallPathThreadRelationMenuItem("Show " + ident + " Call Path Relations", ppTrial, thread, owner));
        if (ppTrial.userEventsPresent()) {
            threadPopup.add(createUserEventBarChartMenuItem("Show User Event Bar Chart", ppTrial, thread, owner));
            threadPopup.add(createStatisticsMenuItem("Show User Event Statistics Window", ppTrial, null, thread, true, owner));
            threadPopup.add(createContextEventMenuItem("Show Context Event Window", ppTrial, thread, owner));
        }

        if (thread.getNumSnapshots() > 1) {
            threadPopup.add(createSnapShotMenuItem("Show Snapshots for " + ident, ppTrial, thread, owner));
        }

        if (thread.getMetaData() != null) {
            threadPopup.add(createThreadMetaDataMenuItem("Show Metadata for " + ident, ppTrial, thread, owner));
        }

        threadPopup.add(createComparisonMenuItem("Add " + ident + " to Comparison Window", ppTrial, thread, owner));

        threadPopup.show(owner, evt.getX(), evt.getY());
    }

    public static void handleSnapshotClick(final ParaProfTrial ppTrial, final Thread thread, final Snapshot snapshot,
            JComponent owner, MouseEvent evt) {
        JPopupMenu popup = new JPopupMenu();

        JMenuItem menuItem = new JMenuItem("Hide this snapshot (" + snapshot + ")");
        popup.add(menuItem);

        menuItem = new JMenuItem("Show all snapshots");
        popup.add(menuItem);

        popup.show(owner, evt.getX(), evt.getY());

    }

    public static int[] computeClipping(Rectangle clipRect, Rectangle viewRect, boolean toScreen, boolean fullWindow, int size,
            int rowHeight, int yCoord) {

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
            startElement = ((yBeg - yCoord) / rowHeight) - 1;
            endElement = ((yEnd - yCoord) / rowHeight) + 1;

            if (startElement < 0)
                startElement = 0;

            if (endElement < 0)
                endElement = 0;

            if (startElement > (size - 1))
                startElement = (size - 1);

            if (endElement > (size - 1))
                endElement = (size - 1);

            if (toScreen) {
                yCoord = yCoord + (startElement * rowHeight);
            }
        } else {
            startElement = 0;
            endElement = (size - 1);
        }

        if (startElement < 0) {
            startElement = 0;
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

        JMenu unitsSubMenu = new JMenu("Select Units...");
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

    public static void phaseConvertTrial(ParaProfTrial srcPpTrial, JFrame owner) {

        if (srcPpTrial.getDataSource().getCallPathDataPresent() == false) {
            JOptionPane.showMessageDialog(owner, "Can't phase convert non-callpath profiles");
            return;
        }

        FunctionSelectorDialog fSelector = new FunctionSelectorDialog(owner, true, srcPpTrial.getDataSource().getFunctionIterator(),
                null, false, true);
        fSelector.setTitle("Choose Phases");

        if (fSelector.choose()) {
            List<Object> phases = fSelector.getSelectedObjects();

            List<String> phaseStrings = new ArrayList<String>();
            for (Iterator<Object> it = phases.iterator(); it.hasNext();) {
                Function f = (Function) it.next();
                phaseStrings.add(f.getName());
            }

            DataSource phaseDataSource = new PhaseConvertedDataSource(srcPpTrial.getDataSource(), phaseStrings);

            ParaProfApplication application = ParaProf.applicationManager.addApplication();
            application.setName("New Application");

            ParaProfExperiment experiment = application.addExperiment();
            experiment.setName("New Experiment");

            ParaProf.paraProfManagerWindow.expandApplicationType(0, application.getID(), application);
            ParaProf.paraProfManagerWindow.expandApplication(0, application, experiment);

            final ParaProfTrial ppTrial = new ParaProfTrial();
            ppTrial.getTrial().setDataSource(phaseDataSource);

            ppTrial.setExperiment(experiment);
            ppTrial.setApplicationID(experiment.getApplicationID());
            ppTrial.setExperimentID(experiment.getID());

            ppTrial.getTrial().setName("Phase Converted from " + srcPpTrial.getName());

            experiment.addTrial(ppTrial);
            ppTrial.finishLoad();
            ParaProf.paraProfManagerWindow.populateTrialMetrics(ppTrial);
            EventQueue.invokeLater(new Runnable() {
                public void run() {
                    ppTrial.showMainWindow();
                }
            });
        }
    }

    public static void exportTrial(ParaProfTrial ppTrial, Component owner) {

        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Export Trial");
        //Set the directory.
        //fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        javax.swing.filechooser.FileFilter fileFilters[] = fileChooser.getChoosableFileFilters();

        for (int i = 0; i < fileFilters.length; i++) {
            fileChooser.removeChoosableFileFilter(fileFilters[i]);
        }
        fileChooser.addChoosableFileFilter(new ParaProfFileFilter(ParaProfFileFilter.TXT));
        fileChooser.addChoosableFileFilter(new ParaProfFileFilter(ParaProfFileFilter.PPK));
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));

        int resultValue = fileChooser.showSaveDialog(owner);
        if (resultValue != JFileChooser.APPROVE_OPTION) {
            return;
        }

        try {

            File file = fileChooser.getSelectedFile();
            String path = file.getCanonicalPath();

            String extension = ParaProfFileFilter.getExtension(file);
            if (extension == null) {
                javax.swing.filechooser.FileFilter fileFilter = fileChooser.getFileFilter();
                if (fileFilter instanceof ParaProfFileFilter) {
                    ParaProfFileFilter paraProfImageFormatFileFilter = (ParaProfFileFilter) fileFilter;
                    path = path + "." + paraProfImageFormatFileFilter.getExtension();
                }
                file = new File(path);
            }

            if (file.exists()) {
                int response = JOptionPane.showConfirmDialog(owner, file + " already exists\nOverwrite existing file?",
                        "Confirm Overwrite", JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
                if (response == JOptionPane.CANCEL_OPTION)
                    return;
            }

            extension = ParaProfFileFilter.getExtension(file).toLowerCase();

            if (extension.compareTo("txt") == 0) {
                DataSourceExport.writeDelimited(ppTrial.getDataSource(), file);
            } else {
                DataSourceExport.writePacked(ppTrial.getDataSource(), file);
            }

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }

    }

    public static boolean rightClick(MouseEvent evt) {
        if ((evt.getModifiers() & InputEvent.BUTTON3_MASK) != 0) {
            return true;
        }
        return false;
    }

    // remove the source code location, if preferences are set for it
    private static String removeSource(String str) {
        if (!ParaProf.preferences.getShowSourceLocation()) {
            if (str.startsWith("Loop:")) {
                return str;
            }
            while (str.indexOf("[{") != -1) {
                int a = str.indexOf("[{");
                int b = str.indexOf("}]");
                str = str.substring(0, a) + str.substring(b + 2);
            }
        }
        return str;
    }

    // remove the source code location, if preferences are set for it
    public static String removeSourceLocation(String str) {
        if (str.startsWith("Loop:")) {
            return str;
        }
        while (str.indexOf("[{") != -1) {
            int a = str.indexOf("[{");
            int b = str.indexOf("}]");
            str = str.substring(0, a) + str.substring(b + 2);
        }
        return str;
    }
    
    //remove the 'throttled' tag from the function name
    public static String removeThrottledTag(String str){
    	int dex=str.indexOf("  [THROTTLED]");
    	if(dex>=0){
    		str = str.substring(0,dex);
    	}
    	else if((dex=str.indexOf(" [THROTTLED]"))>=0){
    		str = str.substring(0,dex);
    		}
    	return str;
    }

    // handles reversed callpaths
    public static String getDisplayName(Function function) {
        if (ParaProf.preferences.getReversedCallPaths()) {
            return removeSource(function.getReversedName());
        } else {
            return removeSource(function.getName());
        }
    }

	// handles reversed callpaths
	public static String getUserEventDisplayName(UserEvent event) {
		// if (ParaProf.preferences.getReversedCallPaths()) {
		// return removeSource(event.getReversedName());
		// } else {
		// return removeSource(event.getName());
		// }

		/*
		 * TODO: Support reversed callpaths for context events?
		 */
		return removeSource(event.getName());
	}

    public static String getLeafDisplayName(Function function) {
        String name = function.getName();
        int loc = name.lastIndexOf("=>");
        if (loc != -1) {
            name = name.substring(loc + 2).trim();
        }
        return removeSource(name);
    }

    public static String getReversedLeafDisplayName(Function function) {
        String name = function.getReversedName();
        int loc = name.lastIndexOf("<=");
        if (loc != -1) {
            name = name.substring(loc + 2).trim();
        }
        return removeSource(name);
    }

    private static final String TAU_APPLICATION_NAME="TAU_APPLICATION_NAME";
    public static String getThreadLabel(Thread thread) {

    	MetaDataMap mdm = thread.getMetaData();
    	String tan="";
    	if(ParaProf.preferences.getAppNameLabels()==1&&mdm.containsKey(TAU_APPLICATION_NAME)){
    		tan = mdm.get(TAU_APPLICATION_NAME)+": ";
    	}
    	
        if (thread.getNodeID() == -1 || thread.getNodeID() == -6) {
            return "Mean";
        } else if (thread.getNodeID() == -2) {
            return "Total";
        } else if (thread.getNodeID() == -3 || thread.getNodeID() == -7) {
            return "Std. Dev.";
        }
        else if (thread.getNodeID() == -4) {
            return "Min";
        }
        else if (thread.getNodeID() == -5) {
            return "Max";
        }else {
            if (ParaProf.preferences.getAutoLabels()) {
                DataSource dataSource = thread.getDataSource();
                if (dataSource.getHasContexts() == false && dataSource.getHasThreads() == false) {
                    return tan+"node " + thread.getNodeID();
                }
                if (dataSource.getHasContexts() == false) {
                    return tan+"node " + thread.getNodeID() + ", thread " + thread.getThreadID();
                }
            }
            return tan+"n,c,t " + thread.getNodeID() + "," + thread.getContextID() + "," + thread.getThreadID();
        }

    }

    public static void logException(Exception e) {
        try {
            String file = ParaProf.paraProfHomeDirectory + "/ParaProf.errors";
            FileOutputStream out = new FileOutputStream(file, true);
            PrintStream p = new PrintStream(out);
            p.println("ParaProf Build (" + ParaProf.getVersionString() + ") encountered the following error on ("
                    + new java.util.Date() + ") : ");
            e.printStackTrace(p);
            p.println("");
            p.close();
            out.close();
        } catch (Exception ex) {
            // oh well...
        }
    }

    public static void handleException(Exception e) {
        logException(e);
        new ParaProfErrorDialog(null, e);
    }

    public static Dimension checkSize(Dimension d) {
        if (!ParaProf.demoMode) {
            return d;
        }

        int width = d.width;
        int height = d.height;

        width = Math.min(width, 640);
        height = Math.min(height, 480);
        return new Dimension(width, height);
    }

    public static NumberFormat createNumberFormatter(final int units, final boolean timeDenominator) {
        return new NumberFormat() {

            /**
			 * 
			 */
			private static final long serialVersionUID = 3533959839796773891L;

			public Number parse(String source, ParsePosition parsePosition) {
                // TODO Auto-generated method stub
                return null;
            }

            public StringBuffer format(double number, StringBuffer toAppendTo, FieldPosition pos) {
                return toAppendTo.append(UtilFncs.getOutputString(units, number, 5, timeDenominator));
            }

            public StringBuffer format(long number, StringBuffer toAppendTo, FieldPosition pos) {
                return toAppendTo.append(UtilFncs.getOutputString(units, number, 5, timeDenominator));
            }
        };
    }

    public static void setFrameIcon(Frame frame) {
        URL url = Utility.getResource("tau16x16.gif");
        if (url != null) {
            frame.setIconImage(Toolkit.getDefaultToolkit().getImage(url));
        }
    }

    public static void createMetricToolbarItems(JToolBar bar, ParaProfTrial ppTrial, final DataSorter dataSorter,
            final ToolBarListener listener) {

        final JComboBox metricBox = new JComboBox(ppTrial.getMetricArray());
        final JComboBox valueBox = new JComboBox(ValueType.VALUES);
        metricBox.setMaximumSize(metricBox.getPreferredSize());
        valueBox.setMaximumSize(valueBox.getPreferredSize());

        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                dataSorter.setSelectedMetric((Metric) metricBox.getSelectedItem());
                dataSorter.setValueType((ValueType) valueBox.getSelectedItem());
                listener.toolBarUsed();
            }

        };

        metricBox.addActionListener(actionListener);
        valueBox.addActionListener(actionListener);

        bar.add(metricBox);
        bar.add(valueBox);
    }

    private static Component createMetricMenu(ParaProfTrial ppTrial, final ValueType valueType, boolean enabled,
            ButtonGroup group, final boolean sort, final DataSorter dataSorter, final SortListener sortListener, final ActionListener al) {
        JRadioButtonMenuItem button = null;

        if (ppTrial.getNumberOfMetrics() == 1) {
            button = new JRadioButtonMenuItem(valueType.toString(), enabled);

            button.addActionListener(new ActionListener() {
                public void actionPerformed(ActionEvent evt) {
                    if (sort) {
                        dataSorter.setSortByVisible(false);
                        dataSorter.setSortType(SortType.VALUE);
                        dataSorter.setSortValueType(valueType);
                    } else {
                        dataSorter.setValueType(valueType);
                    }
                    if(sortListener!=null)
                    	sortListener.resort();
                    	
                    else{
                    	al.actionPerformed(null);
                    }
                }
            });
            group.add(button);
            return button;
        } else {
            JMenu subSubMenu = new JMenu(valueType.toString() + "...");
            subSubMenu.getPopupMenu().setLightWeightPopupEnabled(false);
            for (Iterator<Metric> it = ppTrial.getMetrics().iterator(); it.hasNext();) {
                Metric metric =  it.next();
                if (metric == null) {
                    continue;
                }

                int id = metric.getID();

                if (id == dataSorter.getSelectedMetric().getID() && enabled) {
                    button = new JRadioButtonMenuItem(metric.getName(), true);
                } else {
                    button = new JRadioButtonMenuItem(metric.getName());
                }

                final Metric useMetric = metric;

                button.addActionListener(new ActionListener() {

                    public void actionPerformed(ActionEvent evt) {
                        if (sort) {
                            dataSorter.setSortByVisible(false);
                            dataSorter.setSortType(SortType.VALUE);
                            dataSorter.setSortMetric(useMetric);
                            dataSorter.setSortValueType(valueType);
                        } else {
                            dataSorter.setSelectedMetric(useMetric);
                            dataSorter.setValueType(valueType);
                        }
                        if(sortListener!=null)
                        	sortListener.resort();
                        	
                        else{
                        	al.actionPerformed(null);
                        }
                        //sortListener.resort();
                    }
                });
                group.add(button);
                subSubMenu.add(button);
            }
            return subSubMenu;
        }
    }

    static class SortAction implements ActionListener{

    	DataSorter dataSorter;
    	SortListener sortListener;
    	
    	SortAction(final DataSorter ds, final SortListener sortListener){
    		dataSorter=ds;
    		this.sortListener=sortListener;
    	}
    	
		public void actionPerformed(ActionEvent e) {
			String arg = e.getActionCommand();
			if(arg.equals("Same as Visible Metric")){
				dataSorter.setSortType(SortType.VALUE);
                dataSorter.setSortByVisible(true);
			}else if(arg.equals("Sort By N,C,T")){
                dataSorter.setSortType(SortType.NCT);
			}else if(arg.equals("Name")){
				dataSorter.setSortType(SortType.NAME);
			}else if(arg.equals("Number of Calls")){
                dataSorter.setSortByVisible(false);
                dataSorter.setSortType(SortType.VALUE);
                dataSorter.setSortValueType(ValueType.NUMCALLS);
			}else if(arg.equals("Number of Child Calls")){
                dataSorter.setSortByVisible(false);
                dataSorter.setSortType(SortType.VALUE);
                dataSorter.setSortValueType(ValueType.NUMSUBR);
			}
            sortListener.resort();
		}
    	
    }
    
    static class MetricAction implements ActionListener{

    	DataSorter dataSorter;
    	SortListener sortListener;
    	
    	MetricAction(final DataSorter ds, final SortListener sortListener){
    		dataSorter=ds;
    		this.sortListener=sortListener;
    	}
    	
		public void actionPerformed(ActionEvent e) {
			String arg = e.getActionCommand();
			 if(arg.equals("Number of Calls")){
	                dataSorter.setValueType(ValueType.NUMCALLS);
				}else if(arg.equals("Number of Child Calls")){
	                dataSorter.setValueType(ValueType.NUMSUBR);
				}
			 sortListener.resort();
		}
    	
    }
    
    class HistogramAction implements ActionListener{

    	
		public void actionPerformed(ActionEvent e) {
			// TODO Auto-generated method stub
			
		}
    	
    }
    
    public static JMenu createMetricSelectionMenu(final ParaProfTrial ppTrial, String name, final boolean sort, boolean nct,
            final DataSorter dataSorter, final SortListener sortListener, boolean visibleMetric) {
    	
        ActionListener al=null;
        if(sort){
        	al = new SortAction(dataSorter,sortListener);
        }else{
        	al = new MetricAction(dataSorter,sortListener);
        }
        
        return createMetricSelectionMenu(ppTrial,name,sort,nct,dataSorter,sortListener,al,visibleMetric);
    }
        
        public static JMenu createMetricSelectionMenu(final ParaProfTrial ppTrial, String name, final boolean sort, boolean nct,
                final DataSorter dataSorter, final SortListener sortListener, final ActionListener al, boolean visibleMetric) {
    	
        JRadioButtonMenuItem button;
        JMenu subMenu = new JMenu(name);
        ButtonGroup group = new ButtonGroup();


        if (sort) {

            if (visibleMetric) {
                button = new JRadioButtonMenuItem("Same as Visible Metric", dataSorter.getSortByVisible());
                button.addActionListener(al);
                group.add(button);
                subMenu.add(button);
            }

            if (nct) {
                button = new JRadioButtonMenuItem("Sort By N,C,T", dataSorter.getSortType() == SortType.NCT);
                button.addActionListener(al);
                group.add(button);
                subMenu.add(button);
            } else {
                button = new JRadioButtonMenuItem("Name", dataSorter.getSortType() == SortType.NAME);
                button.addActionListener(al);
                group.add(button);
                subMenu.add(button);
            }
        }

        subMenu.add(createMetricMenu(ppTrial, ValueType.EXCLUSIVE, dataSorter.getValueType() == ValueType.EXCLUSIVE
                || dataSorter.getValueType() == ValueType.EXCLUSIVE_PERCENT, group, sort, dataSorter, sortListener,al));
        subMenu.add(createMetricMenu(ppTrial, ValueType.INCLUSIVE, dataSorter.getValueType() == ValueType.INCLUSIVE
                || dataSorter.getValueType() == ValueType.INCLUSIVE_PERCENT, group, sort, dataSorter, sortListener,al));
        subMenu.add(createMetricMenu(ppTrial, ValueType.EXCLUSIVE_PER_CALL,
                dataSorter.getValueType() == ValueType.EXCLUSIVE_PER_CALL, group, sort, dataSorter, sortListener,al));
        subMenu.add(createMetricMenu(ppTrial, ValueType.INCLUSIVE_PER_CALL,
                dataSorter.getValueType() == ValueType.INCLUSIVE_PER_CALL, group, sort, dataSorter, sortListener,al));

        button = new JRadioButtonMenuItem("Number of Calls", dataSorter.getValueType() == ValueType.NUMCALLS);
        button.addActionListener(al);
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Number of Child Calls", dataSorter.getValueType() == ValueType.NUMSUBR);
        button.addActionListener(al);
        group.add(button);
        subMenu.add(button);
        //if(ppTrial.metricButton!=null)
        	//group.setSelected(ppTrial.metricButton.getModel(), true);
        return subMenu;
    }

}
