/*
 HelpWindow.java
 
 
 Title:      ParaProf
 Author:     Robert Bell
 Description:  This class provides detailed help information for the user.
 */

package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Observable;
import java.util.Observer;

import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.border.Border;
import javax.swing.text.BadLocationException;
import javax.swing.text.Document;

public class HelpWindow extends JFrame implements ActionListener, Observer {

    /**
	 * 
	 */
	private static final long serialVersionUID = -3689599495979927735L;
	//General.
    int windowWidth = 750;
    int windowHeight = 500;

    //Text area stuff.
    JTextArea helpJTextArea;
    Document helpJTextAreaDocument;
    private JScrollPane scrollPane;

    public JScrollPane getScrollPane() {
        return scrollPane;
    }

    public HelpWindow() {

        //Set the preferend initial size for this window.
        setSize(new java.awt.Dimension(windowWidth, windowHeight));
        setTitle("TAU: ParaProf: Help Window");
        ParaProfUtils.setFrameIcon(this);


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
        helpJTextArea.setMargin(new Insets(3, 3, 3, 3));
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
        scrollPane = new JScrollPane(helpJTextArea);
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
