/*
 HelpWindow.java
 
 
 Title:      ParaProf
 Author:     Robert Bell
 Description:  This class provides detailed help information for the user.
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.border.*;
import edu.uoregon.tau.dms.dss.*;

public class HelpWindow extends JFrame implements ActionListener, Observer {

    //General.
    int windowWidth = 750;
    int windowHeight = 500;

    //Text area stuff.
    JTextArea helpJTextArea;
    Document helpJTextAreaDocument;


    public HelpWindow() {

        //Set the preferend initial size for this window.
        setSize(new java.awt.Dimension(windowWidth, windowHeight));
        setTitle("ParaProf Help Window");

        JMenuBar mainMenu = new JMenuBar();

        //******************************
        //File menu.
        //******************************
        JMenu fileMenu = new JMenu("File");

        //Add a menu item.
        JMenuItem generalHelpItem = new JMenuItem("Display General Help");
        generalHelpItem.addActionListener(this);
        fileMenu.add(generalHelpItem);

        //Add a menu item.
        JMenuItem closeItem = new JMenuItem("Close ParaProf Help Window");
        closeItem.addActionListener(this);
        fileMenu.add(closeItem);

        //Add a menu item.
        JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
        exitItem.addActionListener(this);
        fileMenu.add(exitItem);
        //******************************
        //End - File menu.
        //******************************

        //Now, add all the menus to the main menu.
        mainMenu.add(fileMenu);

        setJMenuBar(mainMenu);
        //******************************
        //End - Code to generate the menus.
        //******************************

        //Create the text area and get its document.
        helpJTextArea = new JTextArea();
        helpJTextArea.setLineWrap(true);
        helpJTextArea.setWrapStyleWord(true);
        helpJTextArea.setSize(new java.awt.Dimension(windowWidth, windowHeight));
        helpJTextAreaDocument = helpJTextArea.getDocument();

        //Setting up the layout system for the main window.
        Container contentPane = getContentPane();
        GridBagLayout gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //Create a borders.
        Border mainloweredbev = BorderFactory.createLoweredBevelBorder();

        //The scroll panes into which the list shall be placed.
        JScrollPane scrollPane = new JScrollPane(helpJTextArea);
        scrollPane.setBorder(mainloweredbev);
        scrollPane.setPreferredSize(new Dimension(windowWidth, windowHeight));

        //Add the componants.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        addCompItem(scrollPane, gbc, 0, 0, 2, 1);

        writeText("Welcome to ParaProf!");
        writeText("");
        writeText("For general help, please select display general help from the file menu.");

    }

    public void clearText() {
        try {
            helpJTextAreaDocument.remove(0, helpJTextAreaDocument.getLength());
        } catch (BadLocationException e) {
            // ???
        }
    }

    public void writeText(String inText) {
        helpJTextArea.append(inText);
        helpJTextArea.append("\n");
    }

    //Observer functionProfiles.
    public void update(Observable o, Object arg) {
        String tmpString = (String) arg;
        if (tmpString.equals("subWindowCloseEvent")) {
            setVisible(false);
        }
    }

    //Helper function for adding componants.
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    //ActionListener code.
    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();

                if (arg.equals("Display General Help")) {
                    clearText();
                    writeText("Welcome to ParaProf!");
                    writeText("");
                } else if (arg.equals("Close ParaProf Help Window")) {
                    setVisible(false);
                } else if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

}
