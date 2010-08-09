/*  
 DBConfiguration.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;

public class DBConfiguration extends JFrame implements ActionListener {

    /**
	 * 
	 */
	private static final long serialVersionUID = -3357587639366720895L;
	public DBConfiguration(ParaProfManagerWindow paraProfManager) {
        this.paraProfManager = paraProfManager;

        String password = ParaProf.preferences.getDatabasePassword();
        if (password == null)
            passwordField = new JPasswordField(password, 20);
        else
            passwordField = new JPasswordField(20);

        String configFile = ParaProf.preferences.getDatabaseConfigurationFile();
        if (configFile == null) {
            if (ParaProf.paraProfHomeDirectory.exists())
                configFileField = new JTextField(ParaProf.paraProfHomeDirectory.getPath() + "/perfdmf.cfg", 30);
            else
                configFileField = new JTextField(System.getProperty("user.dir"), 30);
        } else
            configFileField = new JTextField(configFile, 30);

        int windowWidth = 400;
        int windowHeight = 200;

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
        setTitle("TAU: ParaProf: Database Configuration");
        ParaProfUtils.setFrameIcon(this);

        
        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Setting up the layout system for the main window.
        Container contentPane = getContentPane();
        GridBagLayout gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(new JLabel("Password:"), gbc, 0, 0, 1, 1);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(passwordField, gbc, 1, 0, 1, 1);

        JButton jButton = new JButton("Config File");
        jButton.addActionListener(this);
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(jButton, gbc, 0, 1, 1, 1);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(configFileField, gbc, 1, 1, 2, 1);

        jButton = new JButton("Cancel");
        jButton.addActionListener(this);
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(jButton, gbc, 0, 2, 1, 1);

        jButton = new JButton("Ok");
        jButton.addActionListener(this);
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(jButton, gbc, 2, 2, 1, 1);
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            //Object EventSrc = evt.getSource();
            String arg = evt.getActionCommand();
            if (arg.equals("Config File")) {
                JFileChooser jFileChooser = null;
                if (ParaProf.paraProfHomeDirectory.exists())
                    jFileChooser = new JFileChooser(ParaProf.paraProfHomeDirectory.getPath());
                else
                    jFileChooser = new JFileChooser(System.getProperty("user.dir"));
                jFileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                jFileChooser.setMultiSelectionEnabled(false);
                jFileChooser.setDialogTitle("Select");
                jFileChooser.setApproveButtonText("Select");
                if ((jFileChooser.showOpenDialog(this)) != JFileChooser.APPROVE_OPTION) {
                    return;
                }
                //User clicked the approve option.
                configFileField.setText((jFileChooser.getSelectedFile()).getCanonicalPath());
            } else if (arg.equals("Cancel")) {
                closeThisWindow();
            } else if (arg.equals("Ok")) {
                String password = (new String(passwordField.getPassword())).trim();
                if (password.equals(""))
                    ParaProf.preferences.setDatabasePassword(null);
                else
                    ParaProf.preferences.setDatabasePassword(password);
                ParaProf.preferences.setDatabaseConfigurationFile(configFileField.getText().trim());
                closeThisWindow();
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
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
        this.setVisible(false);
        dispose();
    }

    //Instance data.
    ParaProfManagerWindow paraProfManager = null;
    JPasswordField passwordField = null;
    JTextField configFileField = null;
}
