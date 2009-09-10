package edu.uoregon.tau.paraprof.treetable;

import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;

import javax.swing.*;

import edu.uoregon.tau.paraprof.ParaProfMetric;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.perfdmf.Metric;

public class ColumnChooser extends JFrame {

    private ParaProfTrial ppTrial;

 
    private List statistics = new ArrayList();

    private ParaProfMetric numCalls;
    private ParaProfMetric numSubr;
    private CheckBoxCellRenderer checkBoxCellRenderer = new CheckBoxCellRenderer();

    private JList metricJList;
    private DefaultListModel metricModel;

    private JList valueJList;
    private DefaultListModel valueModel;

    private JList statsJList;
    private DefaultListModel statsModel;

    private TreeTableWindow ttWindow;

    public ColumnChooser(TreeTableWindow owner, ParaProfTrial ppTrial) {
        this.ttWindow = owner;

        this.setTitle("Choose Columns");
        this.setSize(500, 400);

        this.ppTrial = ppTrial;

        numCalls = new ParaProfMetric();
        numCalls.setName("Calls");
        numSubr = new ParaProfMetric();
        numSubr.setName("Child Calls");

        //metrics.add(numCalls);
        //metrics.add(numSubr);

        // create the value JList
        valueModel = new DefaultListModel();
        valueModel.addElement(new CheckBoxListItem("Exclusive Value", true));
        valueModel.addElement(new CheckBoxListItem("Inclusive Value", true));
        valueModel.addElement(new CheckBoxListItem("Exclusive Percent Value", false));
        valueModel.addElement(new CheckBoxListItem("Inclusive Percent Value", false));
        valueModel.addElement(new CheckBoxListItem("Exclusive Value Per Call", false));
        valueModel.addElement(new CheckBoxListItem("Inclusive Value Per Call", false));

        statistics.add("Standard Deviation");
        statistics.add("Mini Histogram");

        Metric selectedMetric = null;
        // if greater than 5, don't start with all of them on, instead do the ppTrial's "default metrics"
        if (ppTrial.getMetrics().size() > 3) {
            selectedMetric = ppTrial.getDefaultMetric();
        }

        List metrics = ppTrial.getMetrics();
        
        // create the metric JList
        metricModel = new DefaultListModel();
        for (int i = 0; i < metrics.size(); i++) {
            boolean selected = true;
            if (selectedMetric != null) {
                if (metrics.get(i) != selectedMetric) {
                    selected = false;
                }
            }
            metricModel.addElement(new CheckBoxListItem(((ParaProfMetric) metrics.get(i)), selected));
        }

        metricModel.addElement(new CheckBoxListItem("Calls", true));
        metricModel.addElement(new CheckBoxListItem("Child Calls", true));
        metricJList = new JList(metricModel);
        metricJList.setCellRenderer(checkBoxCellRenderer);
        metricJList.addMouseListener(new MouseController(metricJList, metricModel));

        valueJList = new JList(valueModel);
        valueJList.setCellRenderer(checkBoxCellRenderer);
        valueJList.addMouseListener(new MouseController(valueJList, valueModel));

        // create the stats JList
        statsModel = new DefaultListModel();
        for (int i = 0; i < statistics.size(); i++) {
            statsModel.addElement(new CheckBoxListItem(statistics.get(i), true));
        }
        statsJList = new JList(statsModel);
        statsJList.setCellRenderer(checkBoxCellRenderer);
        statsJList.addMouseListener(new MouseController(statsJList, statsModel));

        Container panel = this.getContentPane();

        this.getContentPane().setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 0.0;
        gbc.weighty = 0.0;
        ParaProfUtils.addCompItem(this.getContentPane(), new JLabel("Metrics"), gbc, 0, 0, 1, 1);
        ParaProfUtils.addCompItem(this.getContentPane(), new JLabel("Values"), gbc, 1, 0, 1, 1);
        //ParaProfUtils.addCompItem(this.getContentPane(), new JLabel("Statistics (over all threads)"), gbc, 0, 2, 2, 1);
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 0.5;
        gbc.weighty = 0.5;
        ParaProfUtils.addCompItem(this.getContentPane(), new JScrollPane(metricJList), gbc, 0, 1, 1, 1);
        ParaProfUtils.addCompItem(this.getContentPane(), new JScrollPane(valueJList), gbc, 1, 1, 1, 1);
        gbc.weighty = 0.1;
        //ParaProfUtils.addCompItem(this.getContentPane(), new JScrollPane(statsJList), gbc, 0, 3, 2, 1);

        JButton okButton = new JButton("close");
        okButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    //selected = true;
                    ttWindow.updateColumns();
                    //ttWindow.update(null,null);
                    dispose();
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }
        });
        //        JButton cancelButton = new JButton("cancel");
        //
        //        cancelButton.addActionListener(new ActionListener() {
        //            public void actionPerformed(ActionEvent evt) {
        //                try {
        //                    dispose();
        //                } catch (Exception e) {
        //                    ParaProfUtils.handleException(e);
        //                }
        //            }
        //        });

        JPanel buttonPanel = new JPanel();
        buttonPanel.add(okButton);
        //        buttonPanel.add(cancelButton);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        ParaProfUtils.addCompItem(panel, buttonPanel, gbc, 1, 2, 1, 1);

    }

    // center the frame in the owner 
    private void center(JFrame owner) {
        int centerOwnerX = owner.getX() + (owner.getWidth() / 2);
        int centerOwnerY = owner.getY() + (owner.getHeight() / 2);
        int posX = centerOwnerX - (this.getWidth() / 2);
        int posY = centerOwnerY - (this.getHeight() / 2);
        posX = Math.max(posX, 0);
        posY = Math.max(posY, 0);
        this.setLocation(posX, posY);
    }

    public ListModel getMetricModel() {
        return metricModel;
    }

    public ListModel getValueModel() {
        return valueModel;
    }

    public void showDialog(JFrame owner, boolean modal) {
        this.center(owner);
        this.setVisible(true);
    }

    static class CheckBoxListItem {
        private Object userObject;
        private boolean selected;

        public CheckBoxListItem(Object userObject, boolean selected) {
            this.userObject = userObject;
            this.selected = selected;
        }

        public boolean getSelected() {
            return selected;
        }

        public void setSelected(boolean selected) {
            this.selected = selected;
        }

        public Object getUserObject() {
            return userObject;
        }

    }

    static class MouseController implements MouseListener {

        private JList list;
        private ListModel model;

        public MouseController(JList list, ListModel model) {
            this.list = list;
            this.model = model;
        }

        public void mouseClicked(MouseEvent e) {
            Point p = e.getPoint();
            int index = list.locationToIndex(p);

            CheckBoxListItem checkBox = (CheckBoxListItem) model.getElementAt(index);
            if (checkBox.getSelected()) {
                checkBox.setSelected(false);
            } else {
                checkBox.setSelected(true);
            }
            list.repaint();
        }

        public void mouseEntered(MouseEvent e) {}

        public void mouseExited(MouseEvent e) {}

        public void mousePressed(MouseEvent e) {}

        public void mouseReleased(MouseEvent e) {}
    }

    static class CheckBoxCellRenderer extends JCheckBox implements ListCellRenderer {
        public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected,
                boolean cellHasFocus) {
            setBackground(list.getBackground());
            CheckBoxListItem checkBox = (CheckBoxListItem) value;
            setText(checkBox.getUserObject().toString());
            setSelected(checkBox.getSelected());
            return this;
        }
    }
}
