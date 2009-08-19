/*
 * LoadTrialWindow.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.perfexplorer.client;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.net.URL;

import javax.swing.*;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;

public class LoadTrialProgressWindow extends JFrame implements ActionListener {

    private PerfExplorerClient mainWindow = null;
    private JProgressBar progressBar = null;
    private DataSource dataSource = null;
    private Trial ppTrial = null;
    private boolean aborted = false;
    private JLabel label;
    private boolean dbUpload = false;
    private javax.swing.Timer jTimer;

    public LoadTrialProgressWindow(PerfExplorerClient mainWindow, final DataSource dataSource,
            final Trial ppTrial, boolean justDB) {

        this.dataSource = dataSource;
        this.ppTrial = ppTrial;
        this.mainWindow = mainWindow;

        //Window Stuff.
        int windowWidth = 300;
        int windowHeight = 120;

        //Grab mainWindow position and size.
        Point parentPosition = mainWindow.getLocationOnScreen();
        Dimension parentSize = mainWindow.getSize();
        int parentWidth = parentSize.width;
        int parentHeight = parentSize.height;

        //Set the window to come up in the center of the screen.
        int xPosition = (parentWidth - windowWidth) / 2;
        int yPosition = (parentHeight - windowHeight) / 2;

        xPosition = (int) parentPosition.getX() + xPosition;
        yPosition = (int) parentPosition.getY() + yPosition;

        this.setLocation(xPosition, yPosition);
        setSize(new java.awt.Dimension(windowWidth, windowHeight));
        setTitle("TAU: ParaProf: Loading...");
    	URL url = Utility.getResource("tau32x32.gif");
    	if (url != null)
    		setIconImage(Toolkit.getDefaultToolkit().getImage(url));

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        Container contentPane = getContentPane();
        GridBagLayout gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(2, 2, 2, 2);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;
        label = new JLabel("Loading Profile Data...");
        addCompItem(label, gbc, 0, 0, 1, 1);

        progressBar = new JProgressBar(0, 100);

        progressBar.setPreferredSize(new Dimension(200, 14));
        gbc.fill = GridBagConstraints.HORIZONTAL;

        addCompItem(progressBar, gbc, 0, 1, 1, 1);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;

        JButton jButton = new JButton("Cancel");
        jButton.addActionListener(this);
        addCompItem(jButton, gbc, 0, 3, 1, 1);

        jTimer = new javax.swing.Timer(200, this);
        jTimer.start();

        execute(justDB);
    }

    private synchronized void setDBUpload(boolean dbUpload) {
        this.dbUpload = dbUpload;
    }

    private synchronized boolean getDBUpload() {
        return dbUpload;
    }

    private void upload() throws DatabaseException {
        label.setText("Uploading Trial");
        progressBar.setValue(0);
        setDBUpload(true);

        DatabaseAPI dbAPI = PerfExplorerServer.getServer().getSession();
        if (dbAPI != null) {
            // this call will block until the entire thing is uploaded (could be a while)
            ppTrial.setID(dbAPI.uploadTrial(ppTrial));
            // get the root of the tree
            PerfExplorerJTree.getTree().expandLast(true);
        }
    }

    public synchronized void waitForLoad() {
        try {
            this.wait();
        } catch (InterruptedException ie) {
            // ?
        }
    }

    public synchronized void finishLoad() {
        this.notifyAll();
    }

    private synchronized void execute(final boolean justDB) {

        java.lang.Thread thread = new java.lang.Thread(new Runnable() {

            public void run() {
                try {

                    if (justDB) {
                        upload();
                        jTimer.stop();
                        finishLoad();
                        LoadTrialProgressWindow.this.dispose();
                    } else {

                        dataSource.load();
                        if (!aborted) {
//                            progressBar.setValue(100);
                            //ppTrial.update(dataSource);

                                upload();
                        }
                        ppTrial = null;
                        jTimer.stop();
                        LoadTrialProgressWindow.this.dispose();
                    }
                } catch (final Exception e) {
                    LoadTrialProgressWindow.this.dispose();
                    System.err.print(e.getMessage());
                    e.printStackTrace(System.err);
                }
            }

        });
        thread.start();

    }

    public void updateProgress() {
        if (getDBUpload()) {
            // we are on the db upload phase
            DatabaseAPI dbAPI = PerfExplorerServer.getServer().getSession();
            if (dbAPI != null) { // it may not be set yet
                int progress = dbAPI.getProgress();
                progressBar.setValue(progress);
            }

        } else {
            int progress = (int) dataSource.getProgress();
            if (progress > 100)
                progress = 100;
            progressBar.setValue(progress);
        }
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();
            if (EventSrc instanceof javax.swing.Timer) {
                // the timer has ticked, get progress and post

                updateProgress();
            } else {
                String arg = evt.getActionCommand();

                if (arg.equals("Cancel")) {
                    aborted = true;
                    if (dbUpload) {
                        DatabaseAPI dbAPI = PerfExplorerServer.getServer().getSession();
                        dbAPI.cancelUpload();
                    } else {
                        dataSource.cancelLoad();
                    }
                }
            }
        } catch (Exception e) {
        	System.err.println(e.getMessage());
        	e.printStackTrace();
        }
    }

    public void finishDatabase(boolean success) {

        if (success && !aborted) {
            progressBar.setValue(100);
        }
        jTimer.stop();

        this.dispose();
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    //Close the window when the close box is clicked
    private void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    void closeThisWindow() {
        try {
            aborted = true;
            if (dbUpload) {
                DatabaseAPI dbAPI = PerfExplorerServer.getServer().getSession();
                dbAPI.cancelUpload();
            } else {
                dataSource.cancelLoad();
            }
            jTimer.stop();
            this.setVisible(false);
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

}