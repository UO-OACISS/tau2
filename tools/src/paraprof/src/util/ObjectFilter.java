package edu.uoregon.tau.paraprof.util;

import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Observable;

import javax.swing.*;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.ParaProfUtils;

public class ObjectFilter extends Observable {
    private Object objects[];

    private boolean filter[];

    private int numShown;
    private int numHidden;

    private JFrame frame;

    public ObjectFilter(List objects) {
        this.objects = objects.toArray();
        filter = new boolean[this.objects.length];
        showAll();
    }

    public List getFilteredObjects() {
        List list = new ArrayList();
        for (int i = 0; i < objects.length; i++) {
            if (filter[i]) {
                list.add(objects[i]);
            }
        }
        return list;
    }

    public void hide(Object object) {
        // maybe use a hash?
        for (int i = 0; i < objects.length; i++) {
            if (object.equals(objects[i])) {
                filter[i] = false;
                numShown--;
                numHidden++;
            }
        }
    }

    public void show(Object object) {
        // maybe use a hash?
        for (int i = 0; i < objects.length; i++) {
            if (object.equals(objects[i])) {
                filter[i] = true;
                numShown++;
                numHidden--;
            }
        }
    }

    public void showAll() {
        for (int i = 0; i < filter.length; i++) {
            filter[i] = true;
        }
        numShown = filter.length;
        numHidden = 0;
    }

    public void closeWindow() {
        if (frame != null) {
            frame.setVisible(false);
            frame.dispose();
            frame = null;
        }
    }

    public void showFrame(String title) {

        if (frame == null) {
            frame = new JFrame();

            final JCheckBox[] boxes = new JCheckBox[objects.length];
            for (int i = 0; i < objects.length; i++) {
                boxes[i] = new JCheckBox(objects[i].toString());
                boxes[i].setSelected(filter[i]);
            }

            CheckBoxList checkBoxList = new CheckBoxList(boxes);

            JScrollPane scrollpane = new JScrollPane(checkBoxList);

            frame.setLayout(new GridBagLayout());

            GridBagConstraints gbc = new GridBagConstraints();

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.NORTH;
            gbc.weightx = 0.5;
            gbc.weighty = 0.5;

            JButton applyButton = new JButton("Apply");

            applyButton.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    for (int i = 0; i < objects.length; i++) {
                        filter[i] = boxes[i].isSelected();
                    }
                    setChanged();
                    notifyObservers();
                }
            });

            JButton cancelButton = new JButton("Cancel");
            cancelButton.addActionListener(new ActionListener() {

                public void actionPerformed(ActionEvent e) {
                    frame.setVisible(false);
                    frame.dispose();
                    frame = null;
                }
            });
            Utility.addCompItem(frame, scrollpane, gbc, 0, 0, 2, 1);
            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.SOUTHEAST;
            gbc.weightx = 0.0;
            gbc.weighty = 0.0;

            Utility.addCompItem(frame, applyButton, gbc, 0, 1, 1, 1);
            Utility.addCompItem(frame, cancelButton, gbc, 1, 1, 1, 1);

            frame.pack();
            frame.setSize(new Dimension(300, 800));
            ParaProfUtils.setFrameIcon(frame);
        }

        frame.setTitle(title);
        frame.setVisible(true);
    }

}
