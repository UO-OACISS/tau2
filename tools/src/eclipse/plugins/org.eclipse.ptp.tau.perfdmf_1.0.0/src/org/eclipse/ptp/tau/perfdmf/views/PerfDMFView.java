package org.eclipse.ptp.tau.perfdmf.views;

import java.io.File;
import java.lang.reflect.Field;
import java.util.*;

import org.eclipse.core.resources.*;
import org.eclipse.core.runtime.IAdaptable;
import org.eclipse.jface.action.*;
import org.eclipse.jface.dialogs.MessageDialog;
import org.eclipse.jface.text.source.ISourceViewer;
import org.eclipse.jface.viewers.*;
import org.eclipse.ptp.tau.perfdmf.PerfDMFUIPlugin;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Image;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Menu;
import org.eclipse.ui.*;
import org.eclipse.ui.editors.text.TextEditor;
import org.eclipse.ui.part.DrillDownAdapter;
import org.eclipse.ui.part.FileEditorInput;
import org.eclipse.ui.part.ViewPart;
import org.eclipse.ui.plugin.AbstractUIPlugin;
import org.eclipse.ui.texteditor.AbstractTextEditor;

import edu.uoregon.tau.paraprof.GlobalDataWindow;
import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.paraprof.interfaces.EclipseHandler;
import edu.uoregon.tau.perfdmf.*;

/**
 * This sample class demonstrates how to plug-in a new
 * workbench view. The view shows data obtained from the
 * model. The sample creates a dummy model on the fly,
 * but a real implementation would connect to the model
 * available either in this or another plug-in (e.g. the workspace).
 * The view is connected to the model using a content provider.
 * <p>
 * The view uses a label provider to define how model
 * objects should be presented in the view. Each
 * view can present the same model objects using
 * different labels and icons, if needed. Alternatively,
 * a single label provider can be shared between views
 * in order to ensure that objects of the same type are
 * presented in the same way everywhere.
 * <p>
 */

public class PerfDMFView extends ViewPart {
    private TreeViewer viewer;
    private DrillDownAdapter drillDownAdapter;
    private Action action1;
    private Action refreshAction;
    private Action doubleClickAction;

    private Action paraprofAction;
    private Action launchparaprofAction;

    static {
        ParaProf.insideEclipse = true;
    }

    class TreeNode implements IAdaptable {
        private String name;
        private Object userObject;
        private TreeNode parent;

        private ArrayList children = new ArrayList();

        public TreeNode(String name, Object userObject) {
            this.name = name;
            this.userObject = userObject;
        }

        public Object getAdapter(Class adapter) {
            return null;
        }

        public String getName() {
            return name;
        }

        public void setParent(TreeNode parent) {
            this.parent = parent;
        }

        public TreeNode getParent() {
            return parent;
        }

        public String toString() {
            return getName();
        }

        public Object getUserObject() {
            return userObject;
        }

        public void setUserObject(Object userObject) {
            this.userObject = userObject;
        }

        public void addChild(TreeNode child) {
            children.add(child);
            child.setParent(this);
        }

        public void removeChild(TreeNode child) {
            children.remove(child);
            child.setParent(null);
        }

        public TreeNode[] getChildren() {
            return (TreeNode[]) children.toArray(new TreeNode[children.size()]);
        }

        public boolean hasChildren() {
            return children.size() > 0;
        }

    }

    /*
     * The content provider class is responsible for providing objects to the
     * view. It can wrap existing objects in adapters or simply return objects
     * as-is. These objects may be sensitive to the current input of the view,
     * or ignore it and always show the same content (like Task List, for
     * example).
     */

    IFile getFile(String filename, IResource[] resources) {
        try {
            for (int j = 0; j < resources.length; j++) {
                System.out.println("  considering resource '" + resources[j] + "'");
                if (resources[j] instanceof IFile) {
                    IFile f = (IFile) resources[j];
                    System.out.println("filename = " + f.getName());
                    if (f.getName().equals(filename)) {
                        return f;
                    }
                } else if (resources[j] instanceof IFolder) {
                    System.out.println("recurse on Folder");
                    IFile f = getFile(filename, ((IFolder) resources[j]).members());
                    if (f != null) {
                        return f;
                    }
                } else if (resources[j] instanceof IProject) {
                    System.out.println("recurse on Project");
                    IFile f = getFile(filename, ((IProject) resources[j]).members());
                    if (f != null) {
                        return f;
                    }
                }
            }
        } catch (Throwable t) {
            t.printStackTrace();
        }
        return null;
    }

    public static IWorkbenchPage getActivePage() {
        IWorkbenchWindow window = PlatformUI.getWorkbench().getActiveWorkbenchWindow();
        if (window != null) {
            return window.getActivePage();
        }
        return null;
    }

    private void openSource(String projectName, final SourceRegion sourceLink) {

        try {
            IWorkspace workspace = ResourcesPlugin.getWorkspace();
            IProject[] projects = workspace.getRoot().getProjects();
            IWorkspaceRoot root = workspace.getRoot();

            IFile file = getFile(sourceLink.getFilename(), root.members());

            if (file == null) {
                return;
            }
            IEditorInput iEditorInput = new FileEditorInput(file);
            
            IWorkbenchPage p = getActivePage();

            IEditorPart part = null;
            if (p != null) {
                part = p.openEditor(iEditorInput, "org.eclipse.ui.DefaultTextEditor", true);
            }
           
            
            //IEditorPart part = EditorUtility.openInEditor(file);

            TextEditor textEditor = (TextEditor) part;

            final int start = textEditor.getDocumentProvider().getDocument(textEditor.getEditorInput()).getLineOffset(
                    sourceLink.getStartLine() - 1);
            final int end = textEditor.getDocumentProvider().getDocument(textEditor.getEditorInput()).getLineOffset(
                    sourceLink.getEndLine());

            textEditor.setHighlightRange(start, end - start, true);

            AbstractTextEditor abstractTextEditor = textEditor;

            ISourceViewer viewer = null;

            final Field fields[] = AbstractTextEditor.class.getDeclaredFields();
            for (int i = 0; i < fields.length; ++i) {
                if ("fSourceViewer".equals(fields[i].getName())) {
                    Field f = fields[i];
                    f.setAccessible(true);
                    viewer = (ISourceViewer) f.get(abstractTextEditor);
                    break;
                }
            }

            if (viewer != null) {
                viewer.revealRange(start, end - start);
                viewer.setSelectedRange(start, end - start);
            }

        } catch (Throwable t) {
            t.printStackTrace();
        }
    }

    class ViewContentProvider implements IStructuredContentProvider, ITreeContentProvider {
        private TreeNode invisibleRoot;

        public void refresh(Viewer v) {
            invisibleRoot = null;
            v.refresh();
        }

        public Object getRoot() {
            return invisibleRoot;
        }

        public void inputChanged(Viewer v, Object oldInput, Object newInput) {
        }

        public void dispose() {
        }

        public Object[] getElements(Object parent) {
            if (parent.equals(getViewSite())) {
                if (invisibleRoot == null)
                    initialize();
                return getChildren(invisibleRoot);
            }
            return getChildren(parent);
        }

        public Object getParent(Object child) {
            return ((TreeNode) child).getParent();
        }

        public Object[] getChildren(Object parent) {
            return ((TreeNode) parent).getChildren();
        }

        public boolean hasChildren(Object parent) {
            return ((TreeNode) parent).hasChildren();
        }

        /*
         * We will set up a dummy model to initialize tree heararchy.
         * In a real code, you will connect to a real model and
         * expose its hierarchy.
         */
        private void initialize() {

            try {

                ParaProf.eclipseHandler = new EclipseHandler() {

                    public boolean openSourceLocation(ParaProfTrial ppTrial, final Function function) {
                        System.out.println("Opening Source Code for " + function);
                        //                        openSource(null,function.getSourceLink());

                        Display.getDefault().asyncExec(new Runnable() {

                            public void run() {
                                openSource(null, function.getSourceLink());
                            }

                        });
                        return true;
                    }

                };

                invisibleRoot = new TreeNode("", null);
                String perfdmf = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg";

                DatabaseAPI dbApi = new DatabaseAPI();

                dbApi.initialize(perfdmf, false);

                for (Iterator it = dbApi.getApplicationList().iterator(); it.hasNext();) {
                    Application app = (Application) it.next();
                    dbApi.setApplication(app);
                    System.out.println("> " + app.getName());

                    TreeNode root = new TreeNode(app.getName(), app);
                    for (Iterator it2 = dbApi.getExperimentList().iterator(); it2.hasNext();) {
                        Experiment exp = (Experiment) it2.next();
                        dbApi.setExperiment(exp);
                        System.out.println("-> " + exp.getName());

                        TreeNode tp = new TreeNode(exp.getName(), exp);

                        for (Iterator it3 = dbApi.getTrialList().iterator(); it3.hasNext();) {
                            Trial trial = (Trial) it3.next();
                            System.out.println("--> " + trial.getName());
                            TreeNode to = new TreeNode(trial.getName(), trial);
                            tp.addChild(to);
                        }
                        root.addChild(tp);
                    }
                    invisibleRoot.addChild(root);

                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    class ViewLabelProvider extends LabelProvider {

        public String getText(Object obj) {
            return obj.toString();
        }

        public Image getImage(Object obj) {
            String imageKey = ISharedImages.IMG_OBJ_FOLDER;

            if (((TreeNode) obj).getUserObject() instanceof Trial) {
                imageKey = ISharedImages.IMG_OBJ_ELEMENT;
            }
            return PlatformUI.getWorkbench().getSharedImages().getImage(imageKey);
        }
    }

    class NameSorter extends ViewerSorter {
    }

    /**
     * The constructor.
     */
    public PerfDMFView() {
        PerfDMFUIPlugin.registerPerfDMFView(this);
    }

    /**
     * This is a callback that will allow us
     * to create the viewer and initialize it.
     */
    public void createPartControl(Composite parent) {
        viewer = new TreeViewer(parent, SWT.MULTI | SWT.H_SCROLL | SWT.V_SCROLL);
        drillDownAdapter = new DrillDownAdapter(viewer);
        viewer.setContentProvider(new ViewContentProvider());
        viewer.setLabelProvider(new ViewLabelProvider());
        viewer.setSorter(new NameSorter());
        viewer.setInput(getViewSite());
        makeActions();
        hookContextMenu();
        hookDoubleClickAction();
        contributeToActionBars();
    }

    private void hookContextMenu() {
        MenuManager menuMgr = new MenuManager("#PopupMenu");
        menuMgr.setRemoveAllWhenShown(true);
        menuMgr.addMenuListener(new IMenuListener() {
            public void menuAboutToShow(IMenuManager manager) {
                PerfDMFView.this.fillContextMenu(manager);
            }
        });
        Menu menu = menuMgr.createContextMenu(viewer.getControl());
        viewer.getControl().setMenu(menu);
        getSite().registerContextMenu(menuMgr, viewer);
    }

    private void contributeToActionBars() {
        IActionBars bars = getViewSite().getActionBars();
        fillLocalPullDown(bars.getMenuManager());
        fillLocalToolBar(bars.getToolBarManager());
    }

    private void fillLocalPullDown(IMenuManager manager) {
        manager.add(action1);
        manager.add(new Separator());
        manager.add(refreshAction);
    }

    private void fillContextMenu(IMenuManager manager) {
        manager.add(paraprofAction);
        manager.add(refreshAction);
        manager.add(action1);
        manager.add(new Separator());
        drillDownAdapter.addNavigationActions(manager);
        // Other plug-ins can contribute there actions here
        manager.add(new Separator(IWorkbenchActionConstants.MB_ADDITIONS));
    }

    private void fillLocalToolBar(IToolBarManager manager) {
        manager.add(launchparaprofAction);
        manager.add(new Separator());
        drillDownAdapter.addNavigationActions(manager);
    }

    private void openInParaProf(Trial trial) {
        try {
            //JFrame frame = new JFrame("HelloWorldSwing");
            //Display the window.
            //frame.pack();
            //frame.setVisible(true);

            String perfdmf = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg";
            DatabaseAPI dbApi = new DatabaseAPI();
            dbApi.initialize(perfdmf, false);

            dbApi.setTrial(trial.getID());
            DBDataSource dbDataSource = new DBDataSource(dbApi);
            dbDataSource.load();
            trial.setDataSource(dbDataSource);

            ParaProf.initialize();

            ParaProfTrial ppTrial = new ParaProfTrial(trial);

            ppTrial.getTrial().setDataSource(dbDataSource);
            ppTrial.finishLoad();

            GlobalDataWindow gdw = new GlobalDataWindow(ppTrial, null);
            gdw.setVisible(true);

        } catch (Throwable e) {
            e.printStackTrace();
        }

    }

    private void makeActions() {
        action1 = new Action() {
            public void run() {
                try {
                    PlatformUI.getWorkbench().getActiveWorkbenchWindow().getActivePage().showView("sampleview.views.SampleView");
                } catch (Throwable t) {
                    t.printStackTrace();
                }
                //addProfile("This Project", "/tmp/profiles");

            }
        };
        action1.setText("Do something cool!");
        action1.setToolTipText("Action 1 tooltip");
        action1.setImageDescriptor(AbstractUIPlugin.imageDescriptorFromPlugin("PDMA", "icons/refresh.gif"));

        refreshAction = new Action() {
            public void run() {
                ((ViewContentProvider) viewer.getContentProvider()).refresh(viewer);
            }
        };
        refreshAction.setText("Refresh");
        refreshAction.setToolTipText("Refresh Data");
        refreshAction.setImageDescriptor(AbstractUIPlugin.imageDescriptorFromPlugin("PDMA", "icons/refresh.gif"));

        doubleClickAction = new Action() {
            public void run() {
                ISelection selection = viewer.getSelection();
                Object obj = ((IStructuredSelection) selection).getFirstElement();

                TreeNode to = (TreeNode) obj;

                if (to.getUserObject() != null) {
                    openInParaProf((Trial) to.getUserObject());
                }
                //showMessage("Double-click detected on " + obj.toString());
            }
        };

        launchparaprofAction = new Action() {
            public void run() {
                ParaProf.initialize();
                ParaProf.paraProfManagerWindow.setVisible(true);
            }
        };
        launchparaprofAction.setText("Launch ParaProf");
        launchparaprofAction.setToolTipText("Launch ParaProf");
        launchparaprofAction.setImageDescriptor(AbstractUIPlugin.imageDescriptorFromPlugin("PDMA", "icons/pp.gif"));

        paraprofAction = new Action() {
            public void run() {
                ISelection selection = viewer.getSelection();
                Object obj = ((IStructuredSelection) selection).getFirstElement();
                TreeNode node = (TreeNode) obj;
                Trial trial = (Trial) node.getUserObject();
                openInParaProf(trial);
            }
        };
        paraprofAction.setText("Open in ParaProf");
        paraprofAction.setToolTipText("Open in ParaProf");
        paraprofAction.setImageDescriptor(AbstractUIPlugin.imageDescriptorFromPlugin("PDMA", "icons/pp.gif"));

    }

    private void hookDoubleClickAction() {
        viewer.addDoubleClickListener(new IDoubleClickListener() {
            public void doubleClick(DoubleClickEvent event) {
                doubleClickAction.run();
            }
        });
    }

    private void showMessage(String message) {
        MessageDialog.openInformation(viewer.getControl().getShell(), "Performance Data View", message);
    }

    /**
     * Passing the focus request to the viewer's control.
     */
    public void setFocus() {
        viewer.getControl().setFocus();
    }

    public boolean addProfile(String project, String directory) {
        try {
            File[] dirs = new File[1];
            dirs[0] = new File(directory);

            DataSource dataSource = UtilFncs.initializeDataSource(dirs, 0, false);
            dataSource.load();

            // initialize database
            DatabaseAPI dbApi = new DatabaseAPI();
            String perfdmf = System.getProperty("user.home") + "/.ParaProf/perfdmf.cfg";
            dbApi.initialize(perfdmf, false);

            // create the trial
            Trial trial = new Trial();
            trial.setDataSource(dataSource);

            Calendar cal = Calendar.getInstance(TimeZone.getDefault());
            String DATE_FORMAT = "yyyy-MM-dd HH:mm:ss";
            java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat(DATE_FORMAT);
            sdf.setTimeZone(TimeZone.getDefault());

            trial.setName("The New Trial: " + sdf.format(cal.getTime()));
            Experiment exp = dbApi.getExperiment(project, "Experiment", true);
            trial.setExperimentID(exp.getID());

            // upload the trial
            dbApi.uploadTrial(trial);
            dbApi.terminate();

            ViewContentProvider vcp = (ViewContentProvider) viewer.getContentProvider();

            // reloads the tree
            vcp.refresh(viewer);

            Object[] objs;
            objs = vcp.getChildren(vcp.getRoot());

            for (int i = 0; i < objs.length; i++) {
                TreeNode node = (TreeNode) objs[i];
                if (((Application) node.getUserObject()).getID() == exp.getApplicationID()) {
                    viewer.setExpandedState(node, true);
                    Object[] expObjs = node.getChildren();
                    for (int j = 0; j < expObjs.length; j++) {
                        TreeNode expNode = (TreeNode) expObjs[j];
                        if (((Experiment) expNode.getUserObject()).getID() == exp.getID()) {
                            viewer.setExpandedState(expNode, true);

                            Object[] trialObjs = expNode.getChildren();
                            for (int k = 0; k < trialObjs.length; k++) {
                                TreeNode trialNode = (TreeNode) trialObjs[k];
                                if (((Trial) trialNode.getUserObject()).getID() == trial.getID()) {
                                    StructuredSelection selection = new StructuredSelection(trialNode);
                                    viewer.setSelection(selection);
                                }
                            }

                        }
                    }
                }
            }

        } catch (Throwable t) {
            if (t instanceof DatabaseException) {
                ((DatabaseException) t).getException().printStackTrace();
            }
            t.printStackTrace();
        }

        return true;
    }
}