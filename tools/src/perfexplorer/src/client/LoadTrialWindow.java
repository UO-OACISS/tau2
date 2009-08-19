package edu.uoregon.tau.perfexplorer.client;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.net.URL;

import javax.swing.*;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.*;

/**
 * A window that lets the user select a profile format and launch a JFileChooser
 * 
 * <P>CVS $Id: LoadTrialWindow.java,v 1.2 2009/08/19 13:59:42 khuck Exp $</P>
 * @author  Robert Bell, Alan Morris
 * @version $Revision: 1.2 $
 */
public class LoadTrialWindow extends JFrame implements ActionListener {

    private static int defaultIndex;

    static String lastDirectory;

    private PerfExplorerClient mainWindow = null;
    private Application application = null;
    private Experiment experiment = null;
    private boolean newExperiment;
    private boolean newApplication;

    private JTextField dirLocationField = new JTextField(lastDirectory, 30);
    private JComboBox trialTypes = null;
    private File selectedFiles[];
    private JButton selectButton = null;
    private PerfExplorerActionListener listener = null;

    private JCheckBox monitorTrialCheckBox = new JCheckBox("Monitor Trial");

    public LoadTrialWindow(PerfExplorerClient mainWindow, PerfExplorerActionListener listener, Application application, Experiment experiment,
            boolean newApplication, boolean newExperiment) {
        this.mainWindow = mainWindow;
        this.listener = listener;
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
        setTitle("TAU: ParaProf: Load Trial");
    	URL url = Utility.getResource("tau32x32.gif");
    	if (url != null)
    		setIconImage(Toolkit.getDefaultToolkit().getImage(url));

        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        selectButton = new JButton("Select Directory");
        selectButton.addActionListener(this);

        trialTypes = new JComboBox(DataSource.formatTypeStrings);
        trialTypes.setMaximumRowCount(DataSource.formatTypeStrings.length);
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

/*        if (!experiment.dBExperiment()) {
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
*/
            JButton jButton = new JButton("Cancel");
            jButton.addActionListener(this);
            addCompItem(jButton, gbc, 0, 2, 1, 1);

            jButton = new JButton("Ok");
            jButton.addActionListener(this);
            addCompItem(jButton, gbc, 2, 2, 1, 1);

//        }

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
                    jFileChooser.setFileFilter(new FileFilter(FileFilter.PPK));
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
                    listener.handleDelete(experiment);
                }
                if (newApplication) {
                    listener.handleDelete(application);
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
                    addTrial(application, experiment, files, trialTypes.getSelectedIndex(), false,
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
                    addTrial(application, experiment, selectedFiles, trialTypes.getSelectedIndex(), false,
                            monitorTrialCheckBox.isSelected());
                }
                closeThisWindow();
            } else if (arg.equals("comboBoxChanged")) {
                if (trialTypes.getSelectedIndex() == DataSource.TAUPROFILE) {
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
        	System.err.print(e.getMessage());
        	e.printStackTrace(System.err);
        }
    }

    private void addTrial(Application application, Experiment experiment, File files[], int fileType,
            boolean fixGprofNames, boolean monitorProfiles) {
		// TODO Auto-generated method stub
        Trial ppTrial = null;
        DataSource dataSource = null;

        try {
            dataSource = UtilFncs.initializeDataSource(files, fileType, fixGprofNames);
            if (dataSource == null) {
                throw new RuntimeException("Error creating dataSource!");
            }
            dataSource.setGenerateIntermediateCallPathData(true);
        } catch (DataSourceException e) {
            if (files == null || files.length != 0) {
            	System.err.print(e.getMessage());
            	e.printStackTrace();
            }
            return;
        }

        ppTrial = new Trial();
        // this must be done before setting the monitored flag
        ppTrial.setDataSource(dataSource);
        //ppTrial.setLoading(true);
        dataSource.setMonitored(monitorProfiles);
        //ppTrial.setMonitored(monitorProfiles);

        ppTrial.setApplicationID(experiment.getApplicationID());
        ppTrial.setExperimentID(experiment.getID());
//        if (files.length != 0) {
//            ppTrial.setPaths(files[0].getPath());
//        } else {
//            ppTrial.setPaths(System.getProperty("user.dir"));
//        }
        if (files.length == 1) {
            ppTrial.setName(files[0].toString());
        } else {
            ppTrial.setName(FileList.getPathReverse(files[0].getPath()));
        }
//        if (experiment.dBExperiment()) {
//            loadedDBTrials.add(ppTrial);
//            ppTrial.setUpload(true); // This trial is not set to a db trial until after it has finished loading.
//        } else {
//            experiment.addTrial(ppTrial);
//        }

        LoadTrialProgressWindow lpw = new LoadTrialProgressWindow(PerfExplorerClient.getMainFrame(), dataSource, ppTrial, false);
		PerfExplorerModel.getModel().setCurrentSelection(ppTrial);
        lpw.show();		
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