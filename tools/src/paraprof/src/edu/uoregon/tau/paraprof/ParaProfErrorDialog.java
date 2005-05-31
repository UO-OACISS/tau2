package edu.uoregon.tau.paraprof;

import edu.uoregon.tau.dms.dss.*;
import java.awt.*;
import java.awt.event.*;

import javax.swing.*;
import javax.swing.event.*;
import java.io.*;

public class ParaProfErrorDialog extends JFrame implements ActionListener {

    public ParaProfErrorDialog(Exception obj) {

        String errorString = null;

        if (obj instanceof DataSourceException) {
            DataSourceException dse = (DataSourceException) obj;
            Exception e = dse.getException();

            if (e != null) {
                e.printStackTrace();
                StringWriter sw = new StringWriter();
                PrintWriter pw = new PrintWriter(sw);
                e.printStackTrace(pw);
                pw.close();
                errorString = sw.toString();
            } else {
                JOptionPane.showMessageDialog(this, dse.getMessage());
                this.dispose();
                return;
            }

        } else if (obj instanceof DatabaseException) {
            DatabaseException dbe = (DatabaseException) obj;
            Exception e = dbe.getException();

            e.printStackTrace();

            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.close();
            errorString = dbe.getMessage() + "\n\n" + sw.toString();

        } else {
            Exception e = obj;
            e.printStackTrace();

            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            pw.close();
            errorString = sw.toString();
        }

        this.setTitle("ParaProf Error");
        this.setSize(500, 300);

        int windowWidth = 500;
        int windowHeight = 300;

        //      Grab the screen size.
        Toolkit tk = Toolkit.getDefaultToolkit();
        Dimension screenDimension = tk.getScreenSize();
        int screenHeight = screenDimension.height;
        int screenWidth = screenDimension.width;

        //Set the window to come up in the center of the screen.
        int xPosition = (screenWidth - windowWidth) / 2;
        int yPosition = (screenHeight - windowHeight) / 2;

        setLocation(xPosition, yPosition);

        //JTextArea headerTextArea = new JTextArea("<html>An unexpected error has occurred.<br>Please email us at: tau-bugs@cs.uoregon.edu with the message given below.  If possible, please also send the profile files that caused this error as well as a brief description of your sequence of actions.<br>Thanks for your help!</html>");

        JTextArea headerTextArea;
        if (obj instanceof DataSourceException) {
            headerTextArea = new JTextArea(
                    "\nAn error occurred loading the profile.\nThis most likely means that the data is corrupt.\nBelow is the full error message.\n");
        } else if (obj instanceof DatabaseException) {
            headerTextArea = new JTextArea(
                    "\nAn error occurred during a database transaction.\nBelow is the full error message.\n");
        } else {
            headerTextArea = new JTextArea(
                    "\nAn unexpected error has occurred.\nPlease email us at: tau-bugs@cs.uoregon.edu with the message given below.\nIf possible, please also send the profile files that caused this error as well as a brief description of your sequence of actions.\nThanks for your help!\n");
        }

        headerTextArea.setLineWrap(true);
        headerTextArea.setWrapStyleWord(true);
        headerTextArea.setEditable(false);
        //jTextArea.setFont(new Font(p.getParaProfFont(), p.getFontStyle(), p.getFontSize()));

        JLabel lbl = new JLabel();

        headerTextArea.setBackground(lbl.getBackground());

        JTextArea errorTextArea = new JTextArea("Version: " + ParaProf.getVersionString() + "\n" + errorString);

        JButton closeButton = new JButton("Close");
        closeButton.addActionListener(this);

        JPanel panel = new JPanel(new BorderLayout());
        JPanel headerPanel = new JPanel();
        JPanel errorPanel = new JPanel();
        JPanel buttonPanel = new JPanel();

        //errorPanel.setBorder(BorderFactory.createLineBorder(Color.black));
        //errorPanel.setBorder(BorderFactory.createLineBorder(Color.black));

        JScrollPane sp = new JScrollPane(errorTextArea);

        // sp.setBorder(BorderFactory.createRaisedBevelBorder());

        headerPanel.add(headerTextArea);
        buttonPanel.add(closeButton);

        panel.add(headerTextArea, BorderLayout.NORTH);

        panel.add(sp, BorderLayout.CENTER);
        panel.add(buttonPanel, BorderLayout.SOUTH);

        //        this.setLayout(new GridBagLayout());
        //        
        //        GridBagConstraints gbc = new GridBagConstraints();
        //        gbc.insets = new Insets(5, 5, 5, 5);
        //
        //        gbc.fill = GridBagConstraints.BOTH;
        //        gbc.anchor = GridBagConstraints.CENTER;
        //        gbc.weightx = 0;
        //        gbc.weighty = 0;
        //        addCompItem(panel, gbc, 0, 0, 1, 1);

        getContentPane().add(panel);

        //        getContentPane().setLayout(new FlowLayout());
        //        getContentPane().add(headerTextArea);
        //        getContentPane().add(new JSeparator(JSeparator.VERTICAL));
        //        getContentPane().add(sp);
        //        getContentPane().add(closeButton);

        //        
        //        GridBagLayout gbl = new GridBagLayout();
        //        this.getContentPane().setLayout(gbl);
        //
        //        GridBagConstraints gbc = new GridBagConstraints();
        //
        //        gbc.fill = GridBagConstraints.BOTH;
        //        gbc.anchor = GridBagConstraints.CENTER;
        //        gbc.weightx = 1.0;
        //        gbc.weighty = 0.0;
        //        addCompItem(sp, gbc, 0, 0, 1, 1);
        //
        //        
        //        gbc.fill = GridBagConstraints.NONE;
        //        gbc.anchor = GridBagConstraints.SOUTH;
        //        gbc.weightx = 1.0;
        //        gbc.weighty = 1.0;
        //        addCompItem(new JButton("Close"), gbc, 0, 1, 1, 1);

        //this.getContentPane().add(new JButton("asdf"), BorderLayout.SOUTH);
        //this.pack();

        this.show();

    }

    public Insets getInsets() {

        return new Insets(10, 10, 10, 10);

    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JButton) {
                this.dispose();
            }
        } catch (Exception e) {
            // whatever
            this.dispose();
        }
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

}
