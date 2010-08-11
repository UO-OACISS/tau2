package edu.uoregon.tau.paraprof;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;

import javax.swing.*;

import edu.uoregon.tau.vis.VisTools;

public class ThreeDeeScalePanel {

    private String min[], max[];
    private String labels[], unitLabels[];
    private JPanel panel = new JPanel();
    private ScaleBar scaleBars[];
    
    public static ThreeDeeScalePanel CreateScalePanel() {
        ThreeDeeScalePanel scalePanel = new ThreeDeeScalePanel();
        return scalePanel;
    }

    public void setRanges(String min[], String max[], String labels[], String unitLabels[]) {
        this.min = min;
        this.max = max;
        this.labels = labels;
        this.unitLabels = unitLabels;
        this.scaleBars = new ScaleBar[labels.length];
        if (panel != null) {
            createPanel();
            panel.updateUI(); 
            panel.repaint();
        }
    }

    public void setPosition(int idx, float value) {
        scaleBars[idx].setPosition(value);
    }

    public JPanel getJPanel() {
        return panel;
    }

    private void createPanel() {
        panel.removeAll();
        panel.setBorder(BorderFactory.createLoweredBevelBorder());
        panel.setLayout(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();
//        gbc.insets = new Insets(10, 5, 0, 5);
        gbc.insets = new Insets(25, 5, 0, 5);
        gbc.weighty = 0.1;
        gbc.weightx = 0.1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;

        int y = 0;
        for (int i = 0; i < labels.length; i++) {

            VisTools.addCompItem(panel, new JLabel(labels[i] + ":"), gbc, 0, y, 1, 2);
            VisTools.addCompItem(panel, new JLabel("" + min[i]), gbc, 1, y, 1, 2);

            JLabel unitLabel = new JLabel(unitLabels[i]);
            //            unitLabel.setBorder(BorderFactory.createLoweredBevelBorder());
            gbc.weightx = 1.0;

            scaleBars[i] = new ScaleBar();
            scaleBars[i].setPosition(-1);
            scaleBars[i].setBorder(BorderFactory.createLoweredBevelBorder());
            //JTextField textField = new JTextField("");

            gbc.weightx = 1.0;

            gbc.fill = GridBagConstraints.HORIZONTAL;
            //          VisTools.addCompItem(panel, textField, gbc, 2, y, 1, 1);
            VisTools.addCompItem(panel, scaleBars[i], gbc, 2, y, 1, 1);
            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.weighty = 0.1;
            gbc.weightx = 0.1;
            VisTools.addCompItem(panel, new JLabel("" + max[i]), gbc, 3, y, 1, 2);

            gbc.insets = new Insets(0, 5, 0, 5);
            gbc.weightx = 0.0;
            gbc.weighty = 1.0;
            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.NORTH;
            VisTools.addCompItem(panel, unitLabel, gbc, 2, y + 1, 1, 1);
            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.weightx = 0.1;
            gbc.weighty = 0.1;

            y += 2;

        }
    }
}

//int y = 0;
//for (int i = 0; i < 4; i++) {
//
//    VisTools.addCompItem(panel, new JLabel(labels[i] + ":"), gbc, 0, y, 1, 2);
//    VisTools.addCompItem(panel, new JLabel("0"), gbc, 1, y, 1, 2);
//
//    JPanel barPanel = new JPanel();
////    barPanel.setBorder(BorderFactory.createLoweredBevelBorder());
//    barPanel.setLayout(new GridBagLayout());
//    GridBagConstraints bargbc = new GridBagConstraints();
//    bargbc.anchor = GridBagConstraints.NORTH;
//    bargbc.fill = GridBagConstraints.HORIZONTAL;
//    JTextField textField = new JTextField("");
//    JLabel unitLabel = new JLabel("seconds");
//    bargbc.weightx = 1.0;
//    VisTools.addCompItem(barPanel, textField, bargbc, 0, 0, 1, 1);
//    bargbc.fill = GridBagConstraints.NONE;
//    VisTools.addCompItem(barPanel, unitLabel, bargbc, 0, 1, 1, 1);
//    
//    gbc.weightx = 1.0;
//    VisTools.addCompItem(panel, barPanel, gbc, 2, y, 1, 1);
//    gbc.weightx = 0.1;
//    gbc.fill = GridBagConstraints.NONE;
//    gbc.fill = GridBagConstraints.HORIZONTAL;
//    VisTools.addCompItem(panel, new JLabel("100"), gbc, 3, y, 1, 2);
//
//    y += 2;
//
//}

