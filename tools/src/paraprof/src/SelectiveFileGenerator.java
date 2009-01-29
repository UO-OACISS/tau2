package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.Iterator;

import javax.swing.*;
import javax.swing.border.TitledBorder;

import edu.uoregon.tau.common.TauRuntimeException;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Group;
import edu.uoregon.tau.perfdmf.UtilFncs;

public class SelectiveFileGenerator extends JFrame {

    private JCheckBox excludeThrottled = new JCheckBox("Exclude Throttled Routines", true);
    private JCheckBox excludeLightweight = new JCheckBox("Exclude Lightweight Routines", true);
    private JTextField percall = new JTextField("10");
    private JTextField numcalls = new JTextField("100000");
    private JTextArea includedFunctions = new JTextArea();
    private JTextArea excludedFunctions = new JTextArea();

    private JLabel location = new JLabel("Output File:");
    private JLabel percallLabel = new JLabel("Microseconds per call:");
    private JLabel numcallsLabel = new JLabel("Number of calls:");
    private JTextField fileLocation = new JTextField();
    private JButton chooseFileButton = new JButton("...");
    private JButton saveButton = new JButton("save");
    private JButton closeButton = new JButton("close");
    private JCheckBox mergeFile = new JCheckBox("Merge", true);

    private String lastDirectory;

    private ParaProfTrial ppTrial;

    public static void showWindow(ParaProfTrial ppTrial, JFrame owner) {
        SelectiveFileGenerator sfg = new SelectiveFileGenerator(ppTrial, owner);
        sfg.setVisible(true);
        return;
    }

    private void updateExcluded() {
        double numcalls_value = 0;
        double percall_value = 0;
        try {
            numcalls_value = Double.parseDouble(numcalls.getText());
            percall_value = Double.parseDouble(percall.getText());
        } catch (NumberFormatException nfe) {
            // silent
        }

        StringBuffer buffer = new StringBuffer();

        Group throttledGroup = ppTrial.getGroup("TAU_DISABLE");

        for (Iterator it = ppTrial.getMeanThread().getFunctionProfiles().iterator(); it.hasNext();) {
            FunctionProfile fp = (FunctionProfile) it.next();
            if (excludeThrottled.isSelected() && fp.getFunction().isGroupMember(throttledGroup)) {
                buffer.append(ParaProfUtils.removeSourceLocation(fp.getName()) + "\n");
            } else if (excludeLightweight.isSelected() == true) {
                if (fp.getNumCalls() >= numcalls_value) {
                    if (fp.getInclusivePerCall(0) <= percall_value) {
                        buffer.append(ParaProfUtils.removeSourceLocation(fp.getName()) + "\n");
                    }
                }
            }
        }

        excludedFunctions.setText(buffer.toString());

    }

    public SelectiveFileGenerator(ParaProfTrial ppTrial, final JFrame owner) {
        this.ppTrial = ppTrial;
        setTitle("TAU: ParaProf: Selective Instrumentation File Generator");
        ParaProfUtils.setFrameIcon(this);
        int windowWidth = 650;
        int windowHeight = 520;
        setSize(new java.awt.Dimension(windowWidth, windowHeight));
        setResizable(true);

        excludeThrottled.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                updateExcluded();
            }
        });

        excludeLightweight.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                updateExcluded();
            }
        });

        Container contentPane = getContentPane();
        contentPane.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weighty = 0.0;
        gbc.weightx = 0.0;
        ParaProfUtils.addCompItem(getContentPane(), location, gbc, 0, 0, 1, 1);
        gbc.weightx = 1.0;
        ParaProfUtils.addCompItem(getContentPane(), fileLocation, gbc, 1, 0, 1, 1);
        gbc.weightx = 0.0;
        ParaProfUtils.addCompItem(getContentPane(), chooseFileButton, gbc, 2, 0, 1, 1);

        gbc.weightx = 1.0;
        ParaProfUtils.addCompItem(getContentPane(), excludeThrottled, gbc, 0, 1, 3, 1);

        ParaProfUtils.addCompItem(getContentPane(), excludeLightweight, gbc, 0, 2, 3, 1);

        final JPanel lightweightPanel = new JPanel();
        lightweightPanel.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(),
                "Lightweight Routine Exclusion Rules"));
        lightweightPanel.setLayout(new GridBagLayout());

        percall.addKeyListener(new KeyListener() {

            public void keyPressed(KeyEvent e) {}

            public void keyReleased(KeyEvent e) {
                updateExcluded();

            }

            public void keyTyped(KeyEvent e) {}
        });

        numcalls.addKeyListener(new KeyListener() {

            public void keyPressed(KeyEvent e) {}

            public void keyReleased(KeyEvent e) {
                updateExcluded();
            }

            public void keyTyped(KeyEvent e) {}
        });

        gbc.fill = GridBagConstraints.BOTH;
        ParaProfUtils.addCompItem(lightweightPanel, percallLabel, gbc, 0, 0, 1, 1);
        ParaProfUtils.addCompItem(lightweightPanel, percall, gbc, 1, 0, 1, 1);
        ParaProfUtils.addCompItem(lightweightPanel, numcallsLabel, gbc, 0, 1, 1, 1);
        ParaProfUtils.addCompItem(lightweightPanel, numcalls, gbc, 1, 1, 1, 1);

        ParaProfUtils.addCompItem(getContentPane(), lightweightPanel, gbc, 0, 3, 3, 1);

        excludeLightweight.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                setEnabledRecursively(lightweightPanel, excludeLightweight.isSelected());
            }
        });

        lastDirectory = System.getProperty("user.dir");
        fileLocation.setText(lastDirectory + "/select.tau");

        chooseFileButton.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                JFileChooser jFileChooser = new JFileChooser(lastDirectory);
                jFileChooser.setSelectedFile(new File("select.tau"));
                jFileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                jFileChooser.setMultiSelectionEnabled(false);
                jFileChooser.setDialogTitle("Choose Output File");
                jFileChooser.setApproveButtonText("Select");
                if ((jFileChooser.showOpenDialog(SelectiveFileGenerator.this)) != JFileChooser.APPROVE_OPTION) {
                    return;
                }
                lastDirectory = jFileChooser.getSelectedFile().getParent();
                fileLocation.setText(jFileChooser.getSelectedFile().getAbsolutePath());
            }
        });

        saveButton.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                try {
                    File file = new File(fileLocation.getText());
                    FileOutputStream out = new FileOutputStream(file,mergeFile.isSelected());
                    OutputStreamWriter outWriter = new OutputStreamWriter(out);
                    BufferedWriter bw = new BufferedWriter(outWriter);

                    bw.write("\nBEGIN_EXCLUDE_LIST\n");
                    bw.write(excludedFunctions.getText());
                    bw.write("END_EXCLUDE_LIST\n\n");

                    bw.close();
                    outWriter.close();
                    out.close();

                    JOptionPane.showMessageDialog(owner, "Selective Instrumentation file written to \"" + fileLocation.getText()
                            + "\"");

                } catch (FileNotFoundException fnfe) {
                    throw new TauRuntimeException(fnfe);
                } catch (IOException ioe) {
                    throw new TauRuntimeException(ioe);
                }
            }
        });

        closeButton.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                setVisible(false);
            }
        });

        final JPanel excludedPanel = new JPanel();
        excludedPanel.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), "Excluded Routines"));
        excludedPanel.setLayout(new GridBagLayout());
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;

        ParaProfUtils.addCompItem(excludedPanel, new JScrollPane(excludedFunctions), gbc, 0, 0, 1, 1);

        ParaProfUtils.addCompItem(getContentPane(), excludedPanel, gbc, 0, 4, 3, 1);

        gbc.weightx = 0.0;
        gbc.weighty = 0.0;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        ParaProfUtils.addCompItem(getContentPane(), saveButton, gbc, 0, 5, 1, 1);
        ParaProfUtils.addCompItem(getContentPane(), mergeFile, gbc, 1, 5, 1, 1);
        ParaProfUtils.addCompItem(getContentPane(), closeButton, gbc, 2, 5, 1, 1);

        //setEnabledRecursively(lightweightPanel, false);

        updateExcluded();
    }

    public static void setEnabledRecursively(Container root, boolean enabled) {
        root.setEnabled(enabled);

        if (root instanceof JPanel) {
            Color color;
            if (enabled) {
                color = new Color(0, 0, 0);
            } else {
                color = new Color(128, 128, 128);
            }
            JPanel panel = (JPanel) root;
            ((TitledBorder) panel.getBorder()).setTitleColor(color);
        }

        for (int i = 0; i < root.getComponentCount(); i++) {
            Component c = root.getComponent(i);
            if (c instanceof Container) {
                setEnabledRecursively((Container) c, enabled);
            } else {
                c.setEnabled(enabled);
            }

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
