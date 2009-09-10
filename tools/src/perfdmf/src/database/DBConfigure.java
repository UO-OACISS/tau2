/* 
  Title:      DBConfigure.java
  Author:     Robert Bell
  Description: A GUI wrapper around Kevin's configure program.
*/

package edu.uoregon.tau.perfdmf.database;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

import javax.swing.*;

public class DBConfigure extends JFrame implements ActionListener {

    //Some statics.
    static boolean debugIsOn = false;
    private static String USAGE = "DBConfigure (help | debug)";

    public DBConfigure() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {}

        //******************************
        //Window Stuff.
        //******************************
        setTitle("DBConfigure");

        int windowWidth = 640;
        int windowHeight = 480;
        setSize(new java.awt.Dimension(windowWidth, windowHeight));

        //There is really no need to resize this window.
        setResizable(false);

        //Grab the screen size.
        Toolkit tk = Toolkit.getDefaultToolkit();
        Dimension screenDimension = tk.getScreenSize();
        int screenHeight = screenDimension.height;
        int screenWidth = screenDimension.width;

        //Set the window to come up in the center of the screen.
        int xPosition = 0;
        int yPosition = 0;

        //Center the window if required.
        if ((screenHeight > windowHeight) && (screenWidth > windowWidth)) {
            xPosition = (int) ((screenWidth - windowWidth) / 2);
            yPosition = (int) ((screenHeight - windowHeight) / 2);
        }

        setLocation(xPosition, yPosition);

        //******************************
        //End - Window Stuff.
        //******************************

        //******************************
        //Code to generate the menus.
        //******************************

        JMenuBar mainMenu = new JMenuBar();

        //******************************
        //File menu.
        //******************************
        JMenu fileMenu = new JMenu("File");

        //Add a menu item.
        JMenuItem saveItem = new JMenuItem("Save");
        saveItem.addActionListener(this);
        fileMenu.add(saveItem);

        //Add a menu item.
        JMenuItem loadItem = new JMenuItem("Load");
        loadItem.addActionListener(this);
        fileMenu.add(loadItem);

        //Add a menu item.
        JMenuItem exitItem = new JMenuItem("Exit DBConfigure!");
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

        //******************************
        //Setting up the layout system.
        //******************************
        Container contentPane = getContentPane();
        GridBagLayout gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        //******************************
        //End - Setting up the layout system.
        //******************************

        //******************************
        //Create and add the componants.
        //******************************

        JButton button = null;
        JLabel label = null;

        button = new JButton("PerfDMF Home");
        button.addActionListener(this);
        perfDMFHome = new JTextField("unset", 30);
        //Add button.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(button, gbc, 0, 0, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(perfDMFHome, gbc, 1, 0, 1, 1);

        button = new JButton("JDBC Jar File");
        button.addActionListener(this);
        jDBCJarfile = new JTextField("unset", 30);
        //Add button.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(button, gbc, 0, 1, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(jDBCJarfile, gbc, 1, 1, 1, 1);

        label = new JLabel("JDBC Driver");
        jDBCDriver = new JTextField("unset", 30);
        //Add label.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(label, gbc, 0, 2, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(jDBCDriver, gbc, 1, 2, 1, 1);

        label = new JLabel("JDBC Type");
        jDBCType = new JTextField("unset", 30);
        //Add label.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(label, gbc, 0, 3, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(jDBCType, gbc, 1, 3, 1, 1);

        label = new JLabel("Hostname");
        dBHostname = new JTextField("unset", 30);
        //Add label.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(label, gbc, 0, 4, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(dBHostname, gbc, 1, 4, 1, 1);

        label = new JLabel("Port Number");
        dBPortNum = new JTextField("unset", 30);
        //Add label.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(label, gbc, 0, 5, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(dBPortNum, gbc, 1, 5, 1, 1);

        label = new JLabel("DB Name");
        dBName = new JTextField("unset", 30);
        //Add label.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(label, gbc, 0, 6, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(dBName, gbc, 1, 6, 1, 1);

        label = new JLabel("Username");
        dBUsername = new JTextField("unset", 30);
        //Add label.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(label, gbc, 0, 7, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(dBUsername, gbc, 1, 7, 1, 1);

        label = new JLabel("Password");
        dBPassword = new JPasswordField("unset", 30);
        //Add label.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(label, gbc, 0, 8, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(dBPassword, gbc, 1, 8, 1, 1);

        button = new JButton("Schema File");
        button.addActionListener(this);
        //Add button.
        dBSchemaFile = new JTextField("unset", 30);
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(button, gbc, 0, 9, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(dBSchemaFile, gbc, 1, 9, 1, 1);

        button = new JButton("XML Parser");
        button.addActionListener(this);
        //Add button.
        xMLPaser = new JTextField("unset", 30);
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(button, gbc, 0, 10, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(xMLPaser, gbc, 1, 10, 1, 1);

        button = new JButton("Config File");
        button.addActionListener(this);
        //Add button.
        configFileName = new JTextField("unset", 30);
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(button, gbc, 0, 11, 1, 1);
        //Add field.
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(configFileName, gbc, 1, 11, 1, 1);

        //Create buttons to click for saving and adding.
        //Can be done from the menu, but it is nice to put it here too.
        button = new JButton("Load");
        button.addActionListener(this);
        //Add button.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(button, gbc, 0, 12, 1, 1);

        button = new JButton("Save");
        button.addActionListener(this);
        //Add button.
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(button, gbc, 1, 12, 1, 1);

        //******************************
        //End - Create and add the componants.
        //******************************
    }

    public void actionPerformed(ActionEvent evt) {

        Object EventSrc = evt.getSource();
        String arg = evt.getActionCommand();

        if (EventSrc instanceof JMenuItem) {
            if (arg.equals("Save")) {} else if (arg.equals("Load")) {} else if (arg.equals("Exit DBConfigure!")) {
                setVisible(false);
                dispose();
                System.exit(0);
            }
        } else if (EventSrc instanceof JButton) {
            if (arg.equals("PerfDMF Home")) {
                JFileChooser tmpFileChooser = new JFileChooser();
                tmpFileChooser.setDialogTitle("PerfDMF Home");

                //Set the directory to the current directory.
                tmpFileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
                tmpFileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                int resultValue = tmpFileChooser.showOpenDialog(this);

                if (resultValue == JFileChooser.APPROVE_OPTION) {
                    //Try and get the file.
                    File file = tmpFileChooser.getSelectedFile();

                    try {
                        String tmpString = file.getCanonicalPath();
                        perfDMFHome.setText(tmpString);
                    } catch (IOException exp) {
                        System.out.println("An error occurred while getting the directory name!");
                        System.out.println("Please reprot this bug to:tau-bugs.cs.uoregon.edu");
                    }
                }
            } else if (arg.equals("JDBC Jar File")) {
                JFileChooser tmpFileChooser = new JFileChooser();
                tmpFileChooser.setDialogTitle("JDBC Jar File");
                tmpFileChooser.setApproveButtonText("Select");

                //Set the directory.
                if ((perfDMFHome.getText().trim()).equals(unsetString))
                    tmpFileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
                else
                    tmpFileChooser.setCurrentDirectory(new File(perfDMFHome.getText().trim()));

                tmpFileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                int resultValue = tmpFileChooser.showOpenDialog(this);

                if (resultValue == JFileChooser.APPROVE_OPTION) {
                    //Try and get the file.
                    File file = tmpFileChooser.getSelectedFile();

                    try {
                        String tmpString = file.getCanonicalPath();
                        jDBCJarfile.setText(tmpString);
                    } catch (IOException exp) {
                        System.out.println("An error occurred while getting the jar file!");
                        System.out.println("Please reprot this bug to:tau-bugs.cs.uoregon.edu");
                    }
                }
            } else if (arg.equals("Schema File")) {
                JFileChooser tmpFileChooser = new JFileChooser();
                tmpFileChooser.setDialogTitle("Schema File");
                tmpFileChooser.setApproveButtonText("Select");

                //Set the directory.
                if ((perfDMFHome.getText().trim()).equals(unsetString))
                    tmpFileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
                else
                    tmpFileChooser.setCurrentDirectory(new File(perfDMFHome.getText().trim()));

                tmpFileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                int resultValue = tmpFileChooser.showOpenDialog(this);

                if (resultValue == JFileChooser.APPROVE_OPTION) {
                    //Try and get the file.
                    File file = tmpFileChooser.getSelectedFile();

                    try {
                        String tmpString = file.getCanonicalPath();
                        dBSchemaFile.setText(tmpString);
                    } catch (IOException exp) {
                        System.out.println("An error occurred while getting the directory name!");
                        System.out.println("Please reprot this bug to:tau-bugs.cs.uoregon.edu");
                    }
                }
            } else if (arg.equals("XML Parser")) {
                JFileChooser tmpFileChooser = new JFileChooser();
                tmpFileChooser.setDialogTitle("XML Parser");
                tmpFileChooser.setApproveButtonText("Select");

                //Set the directory.
                if ((perfDMFHome.getText().trim()).equals(unsetString))
                    tmpFileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
                else
                    tmpFileChooser.setCurrentDirectory(new File(perfDMFHome.getText().trim()));

                tmpFileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                int resultValue = tmpFileChooser.showOpenDialog(this);

                if (resultValue == JFileChooser.APPROVE_OPTION) {
                    //Try and get the file.
                    File file = tmpFileChooser.getSelectedFile();

                    try {
                        String tmpString = file.getCanonicalPath();
                        xMLPaser.setText(tmpString);
                    } catch (IOException exp) {
                        System.out.println("An error occurred while getting the directory name!");
                        System.out.println("Please reprot this bug to:tau-bugs.cs.uoregon.edu");
                    }
                }
            } else if (arg.equals("Config File")) {
                JFileChooser tmpFileChooser = new JFileChooser();
                tmpFileChooser.setDialogTitle("Config File");

                //Set the directory.
                if ((perfDMFHome.getText().trim()).equals(unsetString))
                    tmpFileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
                else
                    tmpFileChooser.setCurrentDirectory(new File(perfDMFHome.getText().trim()));

                tmpFileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                int resultValue = tmpFileChooser.showSaveDialog(this);

                if (resultValue == JFileChooser.APPROVE_OPTION) {
                    //Try and get the file.
                    File file = tmpFileChooser.getSelectedFile();

                    try {
                        String tmpString = file.getCanonicalPath();
                        configFileName.setText(tmpString);
                    } catch (IOException exp) {
                        System.out.println("An error occurred while getting the directory name!");
                        System.out.println("Please reprot this bug to:tau-bugs.cs.uoregon.edu");
                    }
                }
            } else if (arg.equals("Save")) {
                System.out.println("Writing the following values to file: " + configFileName.getText().trim());
                System.out.println("perfDMFHome: " + perfDMFHome.getText().trim());
                System.out.println("jDBCJarfile: " + jDBCJarfile.getText().trim());
                System.out.println("jDBCDriver: " + jDBCDriver.getText().trim());
                System.out.println("jDBCType: " + jDBCType.getText().trim());
                System.out.println("dBHostname: " + dBHostname.getText().trim());
                System.out.println("dBPortNum: " + dBPortNum.getText().trim());
                System.out.println("dBName: " + dBName.getText().trim());
                System.out.println("dBUsername: " + dBUsername.getText().trim());
                System.out.println("dBPassword: " + new String(dBPassword.getPassword()));
                System.out.println("dBSchemaFile: " + dBSchemaFile.getText().trim());
                System.out.println("dBSchemaFile: " + dBSchemaFile.getText().trim());
                System.out.println("xMLPaser: " + xMLPaser.getText().trim());
            }
        }
    }

    //Adds the components.
    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        this.getContentPane().add(c, gbc);
    }

    // Main entry point
    static public void main(String[] args) {
        int numberOfArguments = 0;
        String argument;

        while (numberOfArguments < args.length) {
            argument = args[numberOfArguments++];
            if (argument.equalsIgnoreCase("HELP")) {
                System.err.println(USAGE);
                System.exit(-1);
            }
            if (argument.equalsIgnoreCase("DEBUG")) {
                DBConfigure.debugIsOn = true;
                continue;
            }
        }

        DBConfigure dBConfigure = new DBConfigure();
        dBConfigure.setVisible(true);
    }

    //Instance data.
    //This object does all the actual non-gui work.
    //Configure = new Configure();
    private JTextField perfDMFHome = null;
    private JTextField jDBCJarfile = null;
    private JTextField jDBCDriver = null;
    private JTextField jDBCType = null;
    private JTextField dBHostname = null;
    private JTextField dBPortNum = null;
    private JTextField dBName = null;
    private JTextField dBUsername = null;
    private JPasswordField dBPassword = null;
    private JTextField dBSchemaFile = null;
    private JTextField xMLPaser = null;
    private JTextField configFileName = null;

    private String unsetString = new String("unset");
}
