package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Group;

public class GroupChangerWindow extends JFrame implements ParaProfWindow, ActionListener {

    private ParaProfTrial ppTrial;
    private DefaultListModel listModel;
    private JList regionList;

    private DefaultListModel currentGroupListModel = new DefaultListModel();;
    private JList currentGroupList;

    private DefaultListModel availableGroupListModel = new DefaultListModel();;
    private JList availableGroupList;

    private JTextField filterTextField;

    public static class GroupListBlob {
        public Group group;
        public boolean allMembers;

        public String toString() {
            return group.getName();
        }
    }

    private class RegionListener implements ListSelectionListener {
        public void valueChanged(ListSelectionEvent e) {
            updateGroupLists();
        }
    }

    private GroupChangerWindow(ParaProfTrial ppTrial, JFrame parent) {
        this.ppTrial = ppTrial;

        setLocation(WindowPlacer.getNewLocation(this, parent));
        ParaProfUtils.setFrameIcon(this);
        setSize(new Dimension(800, 400));
        setTitle("TAU: ParaProf: Group Changer: " + ppTrial);

        Container contentPane = getContentPane();
        contentPane.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0.1;
        gbc.weighty = 0.1;

        JLabel titleLabel = new JLabel("Region");
        gbc.weighty = 0;
        Utility.addCompItem(contentPane, titleLabel, gbc, 0, 0, 2, 1);

        filterTextField = new JTextField();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        filterTextField.addKeyListener(new KeyListener() {

            public void keyTyped(KeyEvent e) {}

            public void keyReleased(KeyEvent e) {
                updateFunctionList();
            }

            public void keyPressed(KeyEvent e) {}
        });
        Utility.addCompItem(contentPane, filterTextField, gbc, 1, 1, 1, 1);
        JLabel filterLabel = new JLabel("filter:");
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0.0;
        Utility.addCompItem(contentPane, filterLabel, gbc, 0, 1, 1, 1);

        listModel = new DefaultListModel();
        updateFunctionList();

        regionList = new JList(listModel);
        regionList.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
        regionList.setSize(500, 300);
        regionList.addListSelectionListener(new RegionListener());
        //        regionList.addMouseListener(this);
        JScrollPane sp = new JScrollPane(regionList);
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 0.5;
        gbc.weighty = 0.5;
        Utility.addCompItem(contentPane, sp, gbc, 0, 2, 2, 2);

        JLabel currentLabel = new JLabel("Current");
        currentLabel.setHorizontalAlignment(SwingConstants.CENTER);
        currentLabel.setVerticalAlignment(SwingConstants.TOP);

        gbc.weightx = 0;
        gbc.weighty = 0;
        Utility.addCompItem(contentPane, currentLabel, gbc, 2, 0, 1, 2);

        currentGroupList = new JList(currentGroupListModel);
        currentGroupList.setCellRenderer(new GroupListCellRenderer());
        currentGroupList.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);

        DefaultListCellRenderer foo = new DefaultListCellRenderer();

        JScrollPane currentListSP = new JScrollPane(currentGroupList);
        gbc.weightx = 0.3;
        gbc.weighty = 0.1;
        Utility.addCompItem(contentPane, currentListSP, gbc, 2, 2, 1, 2);

        gbc.fill = GridBagConstraints.NONE;

        gbc.weightx = 0.0;
        JButton leftButton = new JButton("<--");
        leftButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                addGroups();
            }
        });
        Utility.addCompItem(contentPane, leftButton, gbc, 3, 2, 1, 1);

        JButton rightButton = new JButton("-->");
        rightButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                removeGroups();
            }
        });
        Utility.addCompItem(contentPane, rightButton, gbc, 3, 3, 1, 1);

        gbc.weightx = 0.1;

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;

        JLabel availableLabel = new JLabel("Available");
        availableLabel.setHorizontalAlignment(SwingConstants.CENTER);
        gbc.weighty = 0;
        Utility.addCompItem(contentPane, availableLabel, gbc, 4, 0, 2, 1);

        final JTextField newGroupTextField = new JTextField();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        Utility.addCompItem(contentPane, newGroupTextField, gbc, 4, 1, 1, 1);
        JButton newGroupButton = new JButton("new group");
        newGroupButton.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                GroupChangerWindow.this.ppTrial.getDataSource().addGroup(newGroupTextField.getText());
                updateGroupLists();
            }
        });
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0;
        Utility.addCompItem(contentPane, newGroupButton, gbc, 5, 1, 1, 1);

        gbc.weighty = 0.2;
        gbc.weightx = 0.2;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;

        availableGroupList = new JList(availableGroupListModel);
        availableGroupList.setCellRenderer(new GroupListCellRenderer());
        availableGroupList.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
        JScrollPane availableListSP = new JScrollPane(availableGroupList);
        Utility.addCompItem(contentPane, availableListSP, gbc, 4, 2, 2, 2);

    }

    private void addGroups() {
        Object[] selectedGroups = availableGroupList.getSelectedValues();
        for (int i = 0; i < selectedGroups.length; i++) {
            GroupListBlob blob = (GroupListBlob) selectedGroups[i];
            Group group = blob.group;
            Object[] selectedFunctions = regionList.getSelectedValues();
            for (int j = 0; j < selectedFunctions.length; j++) {
                Function function = (Function) selectedFunctions[j];
                function.addGroup(group);
            }
        }
        updateGroupLists();
        ppTrial.updateRegisteredObjects("dataEvent");
    }

    private void removeGroups() {
        Object[] selectedGroups = currentGroupList.getSelectedValues();
        for (int i = 0; i < selectedGroups.length; i++) {
            GroupListBlob blob = (GroupListBlob) selectedGroups[i];
            Group group = blob.group;
            Object[] selectedFunctions = regionList.getSelectedValues();
            for (int j = 0; j < selectedFunctions.length; j++) {
                Function function = (Function) selectedFunctions[j];
                function.removeGroup(group);
            }
        }
        updateGroupLists();
        ppTrial.updateRegisteredObjects("dataEvent");
    }

    private void updateFunctionList() {
        listModel.clear();
        int idx = 0;

        try {
            String text = filterTextField.getText();
            text = text.replaceAll("\\*", ".\\*");
            text = text.replaceAll("\\(", "\\\\(");
            text = text.replaceAll("\\)", "\\\\)");
            text = text + ".*";

            Pattern pattern = Pattern.compile(text);

            for (Iterator it = ppTrial.getFunctions(); it.hasNext();) {
                Function function = (Function) it.next();
                if (filterTextField.getText().equals("")) {
                    listModel.add(idx++, function);
                } else {

                    Matcher matcher = pattern.matcher(function.getName().trim());
                    if (matcher.matches()) {
                        listModel.add(idx++, function);
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
            // ignore pattern exceptions
        }
    }

    private void updateGroupLists() {
        // create a map of groups to ints, which is the count of the number of functions in each group
        Map<Group, Integer> map = new HashMap<Group, Integer>();
        Object[] selectedValues = regionList.getSelectedValues();
        for (int i = 0; i < selectedValues.length; i++) {
            Function function = (Function) selectedValues[i];
            for (Iterator grIt = function.getGroups().iterator(); grIt.hasNext();) {
                Group group = (Group) grIt.next();
                Integer count = map.get(group);
                if (count == null) {
                    map.put(group, new Integer(1));
                } else {
                    map.put(group, new Integer(count.intValue() + 1));
                }
            }
        }

        // clear both lists
        currentGroupListModel.clear();
        availableGroupListModel.clear();

        // now add to each group list accordingly
        int idx = 0;
        int avail_idx = 0;
        for (Iterator<Group> it = map.keySet().iterator(); it.hasNext();) {
            Group group = it.next();
            int numMembers = map.get(group).intValue();
            GroupListBlob blob = new GroupListBlob();
            blob.group = group;
            if (numMembers == selectedValues.length) {
                blob.allMembers = true;
            } else {
                blob.allMembers = false;
                availableGroupListModel.add(avail_idx++, blob);
            }
            currentGroupListModel.add(idx++, blob);
        }

        for (Iterator it = ppTrial.getDataSource().getGroups(); it.hasNext();) {
            Group group = (Group) it.next();
            if (map.get(group) == null) {
                GroupListBlob blob = new GroupListBlob();
                blob.group = group;
                blob.allMembers = true;
                availableGroupListModel.add(avail_idx++, blob);
            }
        }

    }

    public static GroupChangerWindow createGroupChangerWindow(ParaProfTrial ppTrial, JFrame parent) {
        GroupChangerWindow gcw = new GroupChangerWindow(ppTrial, parent);
        return gcw;
    }

    public void closeThisWindow() {
    // TODO Auto-generated method stub

    }

    public JFrame getFrame() {
        // TODO Auto-generated method stub
        return null;
    }

    public void help(boolean display) {
    // TODO Auto-generated method stub

    }

    public void actionPerformed(ActionEvent e) {
    // TODO Auto-generated method stub

    }

}
