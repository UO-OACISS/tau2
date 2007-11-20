package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.*;

import edu.uoregon.tau.perfdmf.DataSource;

/**
 * A window that lets the user select a profile format and launch a JFileChooser
 * 
 * <P>CVS $Id: LoadTrialWindow.java,v 1.5 2007/11/20 21:29:30 amorris Exp $</P>
 * @author  Robert Bell, Alan Morris
 * @version $Revision: 1.5 $
 */
public class LoadTrialWindow extends JFrame implements ActionListener {

    private static int defaultIndex;

    static String lastDirectory;

    private ParaProfManagerWindow paraProfManagerWindow = null;
    private ParaProfApplication application = null;
    private ParaProfExperiment experiment = null;
    private boolean newExperiment;
    private boolean newApplication;

    private JTextField dirLocationField = new JTextField(lastDirectory, 30);
    private JComboBox trialTypes = null;
    private File selectedFiles[];
    private JButton selectButton = null;

    private JCheckBox monitorTrialCheckBox = new JCheckBox("Monitor Trial");

    public LoadTrialWindow(ParaProfManagerWindow paraProfManager, ParaProfApplication application, ParaProfExperiment experiment,
            boolean newApplication, boolean newExperiment) {
        this.paraProfManagerWindow = paraProfManager;
        this.application = application;
        this.experiment = experiment;
        this.newApplication = newApplication;
        this.newExperiment = newExperiment;

        if (lastDirectory == null) {
            lastDirectory = System.getProperty("user.dir");
            dirLocationField.setText(lastDirectory);
        }

        //Window Stuff.
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
        setTitle("TAU: ParaProf: Load Trial");
        ParaProfUtils.setFrameIcon(this);


        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        selectButton = new JButton("Select Directory");
        selectButton.addActionListener(this);

        trialTypes = new JComboBox(DataSource.formatTypeStrings);
        trialTypes.setMaximumRowCount(13);
        trialTypes.addActionListener(this);
        // must be after action listener
        trialTypes.setSelectedIndex(defaultIndex);

        Container contentPane = getContentPane();
        GridBagLayout gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(new JLabel("Trial Type"), gbc, 0, 0, 1, 1);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(trialTypes, gbc, 1, 0, 1, 1);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(selectButton, gbc, 0, 1, 1, 1);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 100;
        gbc.weighty = 0;
        addCompItem(dirLocationField, gbc, 1, 1, 2, 1);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;

        if (!experiment.dBExperiment()) {
            monitorTrialCheckBox.addActionListener(this);
            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;

            addCompItem(monitorTrialCheckBox, gbc, 1, 2, 1, 1);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;

            JButton jButton = new JButton("Cancel");
            jButton.addActionListener(this);
            addCompItem(jButton, gbc, 0, 3, 1, 1);

            jButton = new JButton("Ok");
            jButton.addActionListener(this);
            addCompItem(jButton, gbc, 2, 3, 1, 1);
        } else {

            JButton jButton = new JButton("Cancel");
            jButton.addActionListener(this);
            addCompItem(jButton, gbc, 0, 2, 1, 1);

            jButton = new JButton("Ok");
            jButton.addActionListener(this);
            addCompItem(jButton, gbc, 2, 2, 1, 1);

        }

    }

    public void actionPerformed(ActionEvent evt) {
        try {
            String arg = evt.getActionCommand();
            if (arg.equals("Select Directory")) {
                JFileChooser jFileChooser = new JFileChooser(lastDirectory);
                jFileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                jFileChooser.setMultiSelectionEnabled(false);
                jFileChooser.setDialogTitle("Select Directory");
                jFileChooser.setApproveButtonText("Select");
                if ((jFileChooser.showOpenDialog(this)) != JFileChooser.APPROVE_OPTION) {
                    return;
                }
                lastDirectory = jFileChooser.getSelectedFile().getParent();
                dirLocationField.setText(jFileChooser.getSelectedFile().getCanonicalPath());

            } else if (arg.equals("  Select File(s)  ")) {
                JFileChooser jFileChooser = new JFileChooser(lastDirectory);
                jFileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

                if (trialTypes.getSelectedIndex() == DataSource.PPK || trialTypes.getSelectedIndex() == DataSource.MPIP
                        || trialTypes.getSelectedIndex() == DataSource.PPROF || trialTypes.getSelectedIndex() == DataSource.CUBE) {
                    // These formats are in a single file only
                    jFileChooser.setMultiSelectionEnabled(false);
                } else {
                    // others may have multiple files
                    jFileChooser.setMultiSelectionEnabled(true);
                }
                if (trialTypes.getSelectedIndex() == DataSource.PPK) {
                    jFileChooser.setFileFilter(new ParaProfFileFilter(ParaProfFileFilter.PPK));
                }
                jFileChooser.setDialogTitle("Select File(s)");
                jFileChooser.setApproveButtonText("Select");
                if ((jFileChooser.showOpenDialog(this)) != JFileChooser.APPROVE_OPTION) {
                    return;
                }

                selectedFiles = jFileChooser.getSelectedFiles();
                lastDirectory = jFileChooser.getSelectedFile().getParent();
                if (!jFileChooser.isMultiSelectionEnabled()) {
                    selectedFiles = new File[1];
                    selectedFiles[0] = jFileChooser.getSelectedFile();
                }

                if (selectedFiles.length > 1) {
                    dirLocationField.setText("<Multiple Files Selected>");
                    dirLocationField.setEditable(false);
                } else {
                    dirLocationField.setText(selectedFiles[0].toString());
                    dirLocationField.setEditable(true);
                }
            } else if (arg.equals("Cancel")) {
                // note that these are null if they're not top level (so this won't delete an application that has other experiments)

                if (newExperiment) {
                    paraProfManagerWindow.handleDelete(experiment);
                }
                if (newApplication) {
                    paraProfManagerWindow.handleDelete(application);
                }
                closeThisWindow();
            } else if (arg.equals("Ok")) {
                if (trialTypes.getSelectedIndex() == 0) {
                    File files[] = new File[1];
                    files[0] = new File(dirLocationField.getText().trim());
                    if (!files[0].exists()) {
                        JOptionPane.showMessageDialog(this, dirLocationField.getText().trim() + " does not exist");
                        return;
                    }
                    paraProfManagerWindow.addTrial(application, experiment, files, trialTypes.getSelectedIndex(), false,
                            monitorTrialCheckBox.isSelected());
                } else {
                    if (selectedFiles == null) {
                        selectedFiles = new File[1];
                        selectedFiles[0] = new File(dirLocationField.getText().trim());
                        if (!selectedFiles[0].exists()) {
                            JOptionPane.showMessageDialog(this, dirLocationField.getText().trim() + " does not exist");
                            return;
                        }
                    }
                    paraProfManagerWindow.addTrial(application, experiment, selectedFiles, trialTypes.getSelectedIndex(), false,
                            monitorTrialCheckBox.isSelected());
                }
                closeThisWindow();
            } else if (arg.equals("comboBoxChanged")) {
                if (trialTypes.getSelectedIndex() == 0) {
                    selectButton.setText("Select Directory");
                    dirLocationField.setEditable(true);
                    monitorTrialCheckBox.setEnabled(true);
                } else {
                    selectButton.setText("  Select File(s)  ");
                    monitorTrialCheckBox.setSelected(false);
                    monitorTrialCheckBox.setEnabled(false);
                }
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

    private void closeThisWindow() {
        this.setVisible(false);
        dispose();
    }

    public static void setDefaultIndex(int index) {
        defaultIndex = index;
    }

}