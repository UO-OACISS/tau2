/*
 * LoadTrialPanel.java
 * 
 * Title: ParaProf Author: Robert Bell Description:
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

import edu.uoregon.tau.dms.dss.*;
import java.util.*;

public class LoadTrialProgressWindow extends JFrame implements ActionListener, ParaProfObserver {

    
    class UpdateRunner implements Runnable {
        
        LoadTrialProgressWindow ltpw;
        
        UpdateRunner(LoadTrialProgressWindow ltpw) {
            this.ltpw = ltpw;
            java.lang.Thread thread = new java.lang.Thread(this);
            thread.start();
        }
        public void run() {

            ppTrial.update(dataSource);
            
            
                //Need to notify observers that we are done. Be careful here.
                //It is likely that they will modify swing elements. Make sure
                //to dump request onto the event dispatch thread to ensure
                //safe update of said swing elements. Remember, swing is not thread
                //safe for the most part.
                EventQueue.invokeLater(new Runnable() {
                    public void run() {
                        ltpw.update(this);
                    }
                });
        }
    }
    
    
    public LoadTrialProgressWindow(ParaProfManager paraProfManager, DataSource dataSource, ParaProfTrial ppTrial) {
            
        this.dataSource = dataSource;
        this.ppTrial = ppTrial;
        
        this.paraProfManager = paraProfManager;

        //####################################
        //Window Stuff.
        //####################################
        int windowWidth = 300;
        int windowHeight = 120;

        //Grab paraProfManager position and size.
        Point parentPosition = paraProfManager.getLocationOnScreen();
        Dimension parentSize = paraProfManager.getSize();
        int parentWidth = parentSize.width;
        int parentHeight = parentSize.height;

        //Set the window to come up in the center of the screen.
        int xPosition = (parentWidth - windowWidth) / 2;
        int yPosition = (parentHeight - windowHeight) / 2;

        xPosition = (int) parentPosition.getX() + xPosition;
        yPosition = (int) parentPosition.getY() + yPosition;

        this.setLocation(xPosition, yPosition);
        setSize(new java.awt.Dimension(windowWidth, windowHeight));
        setTitle("Loading...");

        
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
        
        
        progressBar = new JProgressBar(0,100);

        //Dimension dim = progressBar.getPreferredSize();
        //System.out.println ("x=" + dim.width + ", y=" + dim.height);

        progressBar.setPreferredSize(new Dimension(200,14));
        gbc.fill = GridBagConstraints.HORIZONTAL;

        addCompItem(progressBar, gbc, 0, 1, 1, 1);

        
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        
        JButton jButton = new JButton("Cancel");
        jButton.addActionListener(this);
        addCompItem(jButton, gbc, 0, 3, 1, 1);

        
        //Create and start the a timer, and then add paraprof to it.
        javax.swing.Timer jTimer = new javax.swing.Timer(200, this);
        jTimer.start();
        
        DataSourceThreadControl dataSourceThreadControl = new DataSourceThreadControl();
        dataSourceThreadControl.setDebug(UtilFncs.debug);
        dataSourceThreadControl.addObserver(this);
        dataSourceThreadControl.initialize(dataSource, true);
    }

    public void actionPerformed(ActionEvent evt) {
        Object EventSrc = evt.getSource();
        if (EventSrc instanceof javax.swing.Timer) {
            
            if (dbUpload) {
                // we are on the db upload phase

                int progress = DatabaseAPI.getProgress();
                progressBar.setValue(progress);
                
                
            } else {
                int progress = (int)dataSource.getProgress();
                if (progress > 99)
                    progress = 99;
                progressBar.setValue(progress);
            }
        } else {
            String arg = evt.getActionCommand();
        
            if (arg.equals("Cancel")) {
                if (!dbUpload) {
                    dataSource.cancelLoad();
                    aborted = true;
                }
            }
        }	
    }
        
    
    
    
    
    
    public void update(Object obj) {
        
        if (obj instanceof Exception) {
            //bad read (exception in datasource load)
            JOptionPane.showMessageDialog(this, "Error loading profile: \n" + ((Exception)obj).toString());
            
            this.dispose();

        } else if (obj instanceof UpdateRunner) {
            progressBar.setValue(100);
            ParaProf.paraProfManager.populateTrialMetrics(ppTrial);

            ppTrial.update(dataSource);
            progressBar.setValue(100);
            ppTrial.showMainWindow();
            this.dispose();


        } else {
            if (aborted) {
                this.dispose();
            } else {
                progressBar.setValue(99);
                
                if (ppTrial.upload()) {
                    label.setText("Uploading Trial");
                    progressBar.setValue(0);
                    dbUpload = true;
                    new UpdateRunner(this);
                } else {
                    ppTrial.update(dataSource);
                    progressBar.setValue(100);
                    ppTrial.showMainWindow();
                    this.dispose();

                }
            }
        }
    }

    public void update() {
    }

    
    
    
    
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        try {
            gbc.gridx = x;
            gbc.gridy = y;
            gbc.gridwidth = w;
            gbc.gridheight = h;

            getContentPane().add(c, gbc);
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "LTP03");
        }
    }

    //Close the window when the close box is clicked
    private void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    void closeThisWindow() {
        dataSource.cancelLoad();
        aborted = true;
        this.setVisible(false);
        dispose();
    }

    //####################################
    //Instance data.
    //####################################
    ParaProfManager paraProfManager = null;
    JProgressBar progressBar = null;
    DataSource dataSource = null;
    ParaProfTrial ppTrial = null;
    boolean aborted = false;
    JLabel label;
    boolean dbUpload = false;
}