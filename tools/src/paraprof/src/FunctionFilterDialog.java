package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.*;

import edu.uoregon.tau.common.Utility;


public class FunctionFilterDialog extends JDialog {

    private static String CASE_SENSITIVE = "case sensitive";
    private ButtonGroup group = new ButtonGroup();

    private JRadioButton hideExcept = createRadioButton("hide all except matching", true, group);
    private JRadioButton showExcept = createRadioButton("show all except matching", false, group);
    private JRadioButton hide = createRadioButton("hide matching", false, group);
    private JRadioButton show = createRadioButton("show matching", false, group);
    private JCheckBox caseSensitive = new JCheckBox(CASE_SENSITIVE, false);
    private JTextField textField = new JTextField();

    private JButton apply = new JButton("Apply");
    private JButton cancel = new JButton("Dismiss");

    public FunctionFilterDialog(JFrame owner, final ParaProfTrial ppTrial) {
        super(owner);

        this.setTitle("Advanced Filtering");

        this.setSize(350, 200);

        center(owner);

        Container panel = this.getContentPane();

        panel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();

        gbc.anchor = GridBagConstraints.WEST;
        gbc.insets = new Insets(4, 4, 4, 4);
        gbc.weightx = 0;
        gbc.weighty = 0;

        gbc.fill = GridBagConstraints.HORIZONTAL;
        Utility.addCompItem(panel, textField, gbc, 0, 0, 1, 1);
        gbc.fill = GridBagConstraints.NONE;
        Utility.addCompItem(panel, caseSensitive, gbc, 1, 0, 1, 1);

        Utility.addCompItem(panel, hideExcept, gbc, 0, 1, 1, 1);
        Utility.addCompItem(panel, showExcept, gbc, 0, 2, 1, 1);
        Utility.addCompItem(panel, hide, gbc, 1, 1, 1, 1);
        Utility.addCompItem(panel, show, gbc, 1, 2, 1, 1);

        gbc.anchor = GridBagConstraints.EAST;
        Utility.addCompItem(panel, apply, gbc, 0, 3, 1, 1);
        Utility.addCompItem(panel, cancel, gbc, 1, 3, 1, 1);

        apply.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                if (hideExcept.isSelected() || hide.isSelected()) {
                    ppTrial.hideMatching(textField.getText(), caseSensitive.isSelected(), hideExcept.isSelected());
                } else {
                    ppTrial.showMatching(textField.getText(), caseSensitive.isSelected(), showExcept.isSelected());
                }
            }
        });

        cancel.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                dispose();
            }
        });

    }

    private JRadioButton createRadioButton(String name, boolean value, ButtonGroup group) {

        JRadioButton button = new JRadioButton(name, value);
        group.add(button);
        return button;
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
}
