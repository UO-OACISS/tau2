package edu.uoregon.tau.paraprof.treetable;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.util.Iterator;
import java.util.List;
import java.util.Observable;
import java.util.Observer;

import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTree;
import javax.swing.table.TableColumn;
import javax.swing.tree.DefaultTreeCellRenderer;
import javax.swing.tree.TreeModel;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.common.treetable.AbstractTreeTableModel;
import edu.uoregon.tau.common.treetable.JTreeTable;
import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.WindowPlacer;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UserEventProfile;

public class ContextEventWindow extends JFrame implements Observer,
		ParaProfWindow, Printable, ImageExport, ActionListener {

    /**
	 * 
	 */
	private static final long serialVersionUID = 1234017079628001497L;
	private ParaProfTrial ppTrial;
    private Thread thread;

    private ContextEventModel model;
    private JTreeTable treeTable;

    private JScrollPane scrollPane;

    public ContextEventWindow(ParaProfTrial ppTrial, Thread thread) {
        this(ppTrial, thread, null);
    }

    public ContextEventWindow(ParaProfTrial ppTrial, Thread thread, Component invoker) {
        this.ppTrial = ppTrial;
        this.thread = thread;
        ppTrial.addObserver(this);

        //        if (!ppTrial.getDataSource().getReverseDataAvailable()) {
        //            reverseTreeMenuItem.setEnabled(false);
        //        }

        setSize(ParaProfUtils.checkSize(new Dimension(1000, 600)));

        setLocation(WindowPlacer.getNewLocation(this, invoker));

        if (thread.getNodeID() == -1 || thread.getNodeID() == -6) {
            this.setTitle("TAU: ParaProf: Mean Context Events - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else if (thread.getNodeID() == -2) {
            this.setTitle("TAU: ParaProf: Total Context Events - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else if (thread.getNodeID() == -3 || thread.getNodeID() == -7) {
            this.setTitle("TAU: ParaProf: Std. Dev. Context Events - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else {
            this.setTitle("TAU: ParaProf: Context Events for: " + ParaProfUtils.getThreadLabel(thread)+ " - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));//+ "n,c,t, " +thread.getNodeID() + ","+ thread.getContextID() + "," + thread.getThreadID() +
        }
        ParaProfUtils.setFrameIcon(this);

        if (ParaProf.getHelpWindow().isVisible()) {
            this.help(false);
        }

        setupMenus();
        setupData();

        ParaProf.incrementNumWindows();

    }

	private static String showTot = "Show Total Column";
	JCheckBoxMenuItem showTotalMenuItem;
    private void setupMenus() {

        JMenuBar mainMenu = new JMenuBar();

        mainMenu.add(ParaProfUtils.createFileMenu((ParaProfWindow) this, this, this));

		JMenu optionsMenu = new JMenu("Options");
		showTotalMenuItem = new JCheckBoxMenuItem(showTot);
		showTotalMenuItem.setSelected(true);
		showTotalMenuItem.addActionListener(this);
		optionsMenu.add(showTotalMenuItem);
		mainMenu.add(optionsMenu);

        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
    }

    private void setupData() {
        model = new ContextEventModel(this, ppTrial, thread, false);
		model.showTotal(showTotalMenuItem.getState());
        createTreeTable(model);
        addComponents();
    }

	String hTitle = "Hide Totals";
	String sTitle = "Show Totals";
	class MidNodeActionListener implements ActionListener {
		ContextEventTreeNode node;

		MidNodeActionListener(ContextEventTreeNode node) {
			this.node = node;
		}

		public void actionPerformed(ActionEvent e) {

			String command = e.getActionCommand();
			boolean show = true;
			if (command.equals(hTitle)) {
				show = false;
			}
			showHideTotals(show, node);
			ppTrial.updateRegisteredObjects("colorEvent");

		}

		private void showHideTotals(boolean show, ContextEventTreeNode node) {
			List<ContextEventTreeNode> children = node.getChildren();
			Iterator<ContextEventTreeNode> cit = children.iterator();
			while (cit.hasNext()) {
				ContextEventTreeNode childNode = cit.next();
				UserEventProfile uep = childNode.getUserEventProfile();
				if (uep != null) {
					uep.getUserEvent().setShowTotal(show);
				} else {
					showHideTotals(show, childNode);
				}
				// System.out.println(childNode);
			}
		}

	}

    private void createTreeTable(AbstractTreeTableModel model) {
        treeTable = new JTreeTable(model, true, true);
        //final JComponent localThis = this;
       final JTree tree = 
        	treeTable.getTree();
       FontMetrics fm = tree.getFontMetrics(ParaProf.preferences.getFont());
		tree.setRowHeight(fm.getHeight());

        // Add a mouse listener for this tree.
        MouseListener ml = new MouseAdapter() {
            public void mousePressed(MouseEvent evt) {
                try {
                    //int selRow = tree.getRowForLocation(evt.getX(), evt.getY());
                    TreePath path = tree.getPathForLocation(evt.getX(), evt.getY());
                    if (path != null) {
                    	//Object o = path.getLastPathComponent();
                        ContextEventTreeNode node =  (ContextEventTreeNode) path.getLastPathComponent();
                        if (ParaProfUtils.rightClick(evt)) {
                            //JPopupMenu popup;
                            if (node.getUserEventProfile() != null) {
                                //popup = 
                                		ParaProfUtils.handleUserEventClick(ppTrial, node.getUserEventProfile().getUserEvent(), thread, treeTable, evt);
                                		//.createFunctionClickPopUp(node.getModel().getPPTrial(),node.getFunctionProfile().getFunction(), getThread(), treeTable);
                               // popup.show(treeTable, evt.getX(), evt.getY());
                            } else {
								// ParaProfUtils.handleUserEventClick(ppTrial,
								// null, thread,
								// treeTable, evt);

								JPopupMenu popup = new JPopupMenu();

								ActionListener al = new MidNodeActionListener(
										node);
								JMenuItem menuItem = new JMenuItem(sTitle);

								menuItem.addActionListener(al);
								popup.add(menuItem);

								menuItem = new JMenuItem(hTitle);

								menuItem.addActionListener(al);
								popup.add(menuItem);

								popup.show(treeTable, evt.getX(), evt.getY());

                            }
                        }
                    }
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            	
            	
            	
            	
            	
            //                try {
            //                    int selRow = tree.getRowForLocation(evt.getX(), evt.getY());
            //                    TreePath path = tree.getPathForLocation(evt.getX(), evt.getY());
            //                    if (path != null) {
            //                        TreeTableNode node = (TreeTableNode) path.getLastPathComponent();
            //                        if (ParaProfUtils.rightClick(evt)) {
            //                            JPopupMenu popup;
            //                            if (node.getFunctionProfile() != null) {
            //                                popup = ParaProfUtils.createFunctionClickPopUp(node.getModel().getPPTrial(),
            //                                node.getFunctionProfile().getFunction(), getThread(), treeTable);
            //                                popup.show(treeTable, evt.getX(), evt.getY());
            //                            } else {
            //                                //popup = new JPopupMenu();
            //                            }
            //                        }
            //                    }
            //                } catch (Exception e) {
            //                    ParaProfUtils.handleException(e);
            //                }
            }
        };

        treeTable.addMouseListener(ml);

        DefaultTreeCellRenderer renderer = new DefaultTreeCellRenderer() {

            /**
			 * 
			 */
			private static final long serialVersionUID = 2780103869814842355L;

			public Component getTreeCellRendererComponent(JTree tree, Object value, boolean selected, boolean expanded,
                    boolean leaf, int row, boolean hasFocus) {
                super.getTreeCellRendererComponent(tree, value, selected, expanded, leaf, row, hasFocus);

                this.setIcon(null);

                // shade every other row
                setBackgroundNonSelectionColor(null);
                if (row % 2 == 0) {
                    setBackgroundNonSelectionColor(new Color(235, 235, 235));
                } else {
                    this.setBackground(tree.getBackground());
                }

                return this;
            }
        };
        treeTable.getTree().setCellRenderer(renderer);

        Font font = ParaProf.preferencesWindow.getFont();
        treeTable.setFont(font);
        renderer.setFont(font);

        FontMetrics fontMetrics = getFontMetrics(font);
        treeTable.setRowHeight(fontMetrics.getMaxAscent() + fontMetrics.getMaxDescent() + 3);

        treeTable.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
        treeTable.setAutoResizeMode(JTable.AUTO_RESIZE_SUBSEQUENT_COLUMNS);

        TableColumn col = treeTable.getColumnModel().getColumn(0);

        int nameWidth = 450;

        col.setPreferredWidth(nameWidth);

        //        int numCols = treeTable.getColumnCount();
        //        col = treeTable.getColumnModel().getColumn(numCols - 2);
        //        col.setPreferredWidth(10);
        //        col = treeTable.getColumnModel().getColumn(numCols - 1);
        //        col.setPreferredWidth(10);
    }

    private void addComponents() {
        GridBagConstraints gbc = new GridBagConstraints();
        getContentPane().removeAll();
        getContentPane().setLayout(new GridBagLayout());
        scrollPane = new JScrollPane(treeTable);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.SOUTH;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        Utility.addCompItem(this, scrollPane, gbc, 0, 0, GridBagConstraints.REMAINDER, GridBagConstraints.REMAINDER);

        validate();
    }

    public int print(Graphics graphics, PageFormat pageFormat, int pageIndex) throws PrinterException {
        try {
            if (pageIndex >= 1) {
                return NO_SUCH_PAGE;
            }

            ParaProfUtils.scaleForPrint(graphics, pageFormat, (int) treeTable.getSize().getWidth(),
                    (int) treeTable.getSize().getHeight());
            export((Graphics2D) graphics, false, true, false);

            return Printable.PAGE_EXISTS;
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
            return NO_SUCH_PAGE;
        }
    }

    public void setUnits(int units) {
    // TODO Auto-generated method stub

    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        if (fullWindow) {
            // first draw the column headers
            scrollPane.getColumnHeader().paintAll(g2D);
            // translate past the column headers
            g2D.translate(0, scrollPane.getColumnHeader().getHeight());
            // draw the entire treetable
            treeTable.paintAll(g2D);
        } else {
            scrollPane.paintAll(g2D);
        }

    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        if (fullScreen) {
            Dimension d = treeTable.getSize();
            // we want to draw the column headers as well
            d.setSize(d.getWidth(), d.getHeight() + scrollPane.getColumnHeader().getHeight());
            return d;
        } else {
            return scrollPane.getSize();
        }
    }

    public boolean getTreeMode() {
        return true;
    }

    public void update(Observable o, Object arg) {
    	
    	if(thread.getNodeID()>=0){
    		this.setTitle("TAU: ParaProf: Context Events for: " + ParaProfUtils.getThreadLabel(thread)+ " - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
    	}
		setupData();
        treeTable.repaint();

    }

    public void closeThisWindow() {
        setVisible(false);
        ppTrial.deleteObserver(this);
        ParaProf.decrementNumWindows();
        dispose();
    }

    public void help(boolean display) {
        ParaProf.getHelpWindow().clearText();
        if (display) {
            ParaProf.getHelpWindow().setVisible(true);
        }
        ParaProf.getHelpWindow().writeText("This is the Context Event Window.\n");
        ParaProf.getHelpWindow().writeText("This window shows context events in a tree-table.)\n");
    }
    public JFrame getFrame() {
        return this;
    }
    
    public void selectEvent(UserEventProfile userEventProfile){
    	JTree t= treeTable.getTree();
    	TreeModel tm = t.getModel();
    	//If this isn't a context event it will return the full string
    	ContextEventTreeNode treeRoot=(ContextEventTreeNode)tm.getRoot();
    	String context = userEventProfile.getName().trim().replaceAll(" +", " ");
    	context=context.substring(context.indexOf(':')+1).trim();

    	int depth = 0;
    	//If it's a context event search the tree
    	if(userEventProfile.getUserEvent().isContextEvent()){
    		depth=treeSearch(treeRoot,userEventProfile,t,0,context);
    	}
    	//Otherwise search the flat event list (different APIs are necessary to select from each element of the UI)
    	else{
    		depth=tableSearch(model,userEventProfile);
    		treeTable.getSelectionModel().setSelectionInterval(depth, depth);
    	}
    	
    	int oneRow = treeTable.getRowHeight()+treeTable.getRowMargin();
    	treeTable.scrollRectToVisible(new Rectangle(0,oneRow*depth,5,5));
    	treeTable.validate();
    }
    
    /**
     * Returns the row in which the entry associated with the provided UserEventProfile is found, or 0 if not found.
     * @param model
     * @param uep
     * @return
     */
    private static int tableSearch(ContextEventModel model,UserEventProfile uep){
    	ContextEventTreeNode r = (ContextEventTreeNode)model.getRoot();
    	int count = model.getChildCount(model.getRoot());
    	for(int i=0;i<count;i++){
    		ContextEventTreeNode node = (ContextEventTreeNode)model.getChild(r, i);
    		if(node!=null){
    			UserEventProfile check = (UserEventProfile)node.getUserObject();
    			if(check!=null){
    				if(check.equals(uep)){
    					return i;
    				}
    			}
    		}
    	}
    	return 0;
    }
    
    /**
     * Opens the tree down at each successive level that matches the context event name string. Returns the row occupied by the selected UserEventProfile in the opened tree.
     * @param root
     * @param target
     * @param tree
     * @param depth
     * @param context
     * @return
     */
    private static int treeSearch(ContextEventTreeNode root,final UserEventProfile target, final JTree tree, int depth, final String context){

    	TreeModel tm = tree.getModel();
    	
    	UserEventProfile test = (UserEventProfile) root.getUserObject();
    	if(test!=null&&target.equals(test)){
    		if(depth==0){
    			return 0;
    		}
    		return depth-1;
    	}
    	int children = tm.getChildCount(root);
    	if(children==0){
    		return 0;
    	}
    	for(int i=0;i<children;i++)
    	{
    		
    		ContextEventTreeNode currentChild = (ContextEventTreeNode) tm.getChild(root, i);
    		if(currentChild==null)
    		{
    			continue;
    		}
    		
    		
    		if(context.startsWith(currentChild.getName().trim().replaceAll(" +", " "))){

    			tree.expandRow(i+depth);
    			depth++;
    			return treeSearch(currentChild,target,tree,i+depth,context);
    		}

    		
    		UserEventProfile testChild=(UserEventProfile) currentChild.getUserObject();
    		if(testChild!=null){
    			System.out.println(testChild.getName());
    		}
    		
    		if(testChild!=null&&testChild.equals(target)){
    			if(target.getUserEvent().isContextEvent())
    			{
    				tree.setSelectionRow(depth+i);
    			}

    			return depth+i;
    		}

    		
    	}
    	return 0;
    }

	public void actionPerformed(ActionEvent e) {
		if(e.getActionCommand().equals(showTot)){
			model.showTotal(showTotalMenuItem.getState());
			setupData();
		}
		
	}
}
