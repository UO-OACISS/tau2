package edu.uoregon.tau.paraprof.sourceview;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.*;
import java.util.List;

import javax.swing.*;

import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.WindowPlacer;
import edu.uoregon.tau.perfdmf.SourceRegion;

public class SourceManager extends JFrame {

    private DefaultListModel listModel;
    private JList dirList;
    private SourceRegion toFind;

    private Map<File, SourceViewer> sourceViewers = new TreeMap<File, SourceViewer>();

    public ArrayList<Object> getCurrentElements() {
        ArrayList<Object> list = new ArrayList<Object>();
        for (int i = 0; i < listModel.getSize(); i++) {
            list.add(listModel.getElementAt(i));
        }
        return list;
    }

    private boolean match(String s1, String s2) {
        //System.out.println("comparing " + s1 + " to " + s2);
        if (s1.equals(s2)) {
            return true;
        }

        return false;
    }

    private boolean searchLocations(SourceRegion region, File[] list, boolean recurse) {
        for (int j = 0; j < list.length; j++) {
            if (match(region.getFilename(), list[j].getName())) {
                //System.out.println("found it");
                SourceViewer sourceViewer = sourceViewers.get(list[j]);
                if (sourceViewer == null) {
                    sourceViewer = new SourceViewer(list[j]);
                    sourceViewers.put(list[j], sourceViewer);
                }
                sourceViewer.highlightRegion(region);
                sourceViewer.setVisible(true);
                return true;
            }
        }

        if (recurse) {
            for (int j = 0; j < list.length; j++) {
                if (list[j].isDirectory()) {
                    if (searchLocations(region, list[j].listFiles(), recurse)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    public void showSourceCode(SourceRegion region) {

        String filename = region.getFilename();

        File cwd = new File(".");

        if (searchLocations(region, cwd.listFiles(), false)) {
            return;
        }

        for (int i = 0; i < listModel.getSize(); i++) {
            String directory = (String) listModel.getElementAt(i);
            File file = new File(directory);
            File[] children = file.listFiles();
            if (children != null) {
                if (searchLocations(region, children, true)) {
                    return;
                }
            }
        }

        //        JOptionPane.showMessageDialog(this, "ParaProf could not find \"" + filename
        //                + "\", please add the containing directory, or a parent to the search list.");
        int hr = JOptionPane.showOptionDialog(this, "ParaProf could not find \"" + filename
                + "\", would you like to add the containing directory to the search list?", "Looking for \"" + filename + "\"",
                JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE, null, null, null);

        toFind = null;
        if (hr == JOptionPane.YES_OPTION) {
            toFind = region;
            display(null);
        }
    }

    public SourceManager(List<Object> initialElements) {

        Container contentPane = getContentPane();
        contentPane.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;

        // First add the label.
        JLabel titleLabel = new JLabel("Current Source Directories (directories are search recursively)");
        titleLabel.setFont(new Font("SansSerif", Font.PLAIN, 14));
        addCompItem(titleLabel, gbc, 0, 0, 1, 1);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 0.1;
        gbc.weighty = 0.1;

        // Create and add color list.
        listModel = new DefaultListModel();

        if (initialElements != null) {
            for (int i = 0; i < initialElements.size(); i++) {
                listModel.addElement(initialElements.get(i));
            }
        }

        dirList = new JList(listModel);
        dirList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        dirList.setSize(500, 300);
        // colorList.addMouseListener(this);
        JScrollPane sp = new JScrollPane(dirList);
        addCompItem(sp, gbc, 0, 1, 1, 3);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        JButton button = new JButton("Add");
        button.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent evt) {
                JFileChooser jFileChooser = new JFileChooser();
                jFileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                jFileChooser.setMultiSelectionEnabled(false);
                jFileChooser.setDialogTitle("Select Directory");
                jFileChooser.setApproveButtonText("Select");
                if ((jFileChooser.showOpenDialog(SourceManager.this)) != JFileChooser.APPROVE_OPTION) {
                    return;
                }
                // lastDirectory = jFileChooser.getSelectedFile().getParent();
                try {
                    listModel.addElement(jFileChooser.getSelectedFile().getCanonicalPath());
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }

            }
        });
        addCompItem(button, gbc, 1, 1, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        button = new JButton("Remove");
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                int index = dirList.getSelectedIndex();
                if (index >= 0) {
                    listModel.removeElementAt(index);
                }
            }
        });
        addCompItem(button, gbc, 1, 2, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.SOUTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        button = new JButton("Close");
        button.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                SourceManager.this.setVisible(false);
                if (toFind != null) {
                    showSourceCode(toFind);
                }

            }
        });
        addCompItem(button, gbc, 1, 3, 1, 1);

    }

    public void display(Component invoker) {
        setSize(new Dimension(855, 450));
        setLocation(WindowPlacer.getNewLocation(this, invoker));
        setTitle("TAU: ParaProf: Source Directory Manager");
        ParaProfUtils.setFrameIcon(this);
        setVisible(true);
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

}
