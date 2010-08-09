/*  
 ParaProfImageOptionsPanel.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.common;

import java.awt.Component;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class ImageOptionsPanel extends JPanel implements PropertyChangeListener {

    /**
	 * 
	 */
	private static final long serialVersionUID = -4236392653580823777L;
	private JCheckBox fullScreen = new JCheckBox("Full Window", true);
    private JCheckBox prependHeader = new JCheckBox("Show Meta-Data", true);
    private JLabel imageQualityLabel = new JLabel("Image Quality");
    private String imageQualityStrings[] = { "1.0", "0.75", "0.5", "0.25", "0.15", "0.1" };
    private JComboBox imageQuality = new JComboBox(imageQualityStrings);

    private JCheckBox textAsShapes = new JCheckBox("Draw text as shapes", true);
    private boolean imageQualityEnabled = true;

    public ImageOptionsPanel(Component component, boolean dumbControls, boolean vector) {

        //Window Stuff.
        int windowWidth = 200;
        int windowHeight = 500;
        setSize(new java.awt.Dimension(windowWidth, windowHeight));

        this.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        if (dumbControls) {
            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0;
            gbc.weighty = 0;
            addCompItem(fullScreen, gbc, 0, 0, 1, 1);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0;
            gbc.weighty = 0;
            addCompItem(prependHeader, gbc, 0, 1, 1, 1);
        }

        if (vector) {
            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0;
            gbc.weighty = 0;
            addCompItem(textAsShapes, gbc, 0, 2, 1, 1);

        } else {

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0;
            gbc.weighty = 0;
            addCompItem(imageQualityLabel, gbc, 0, 2, 1, 1);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 100;
            gbc.weighty = 0;
            addCompItem(imageQuality, gbc, 1, 2, 1, 1);
        }
    }

    public boolean isFullScreen() {
        return fullScreen.isSelected();
    }

    public boolean isPrependHeader() {
        return prependHeader.isSelected();
    }

    public float getImageQuality() {
        return Float.valueOf((String) imageQuality.getSelectedItem()).floatValue();
    }

    public boolean imageQualityEnabled() {
        return imageQualityEnabled;
    }

    public boolean getTextAsShapes() {
        return textAsShapes.isSelected();
    }

    public void propertyChange(PropertyChangeEvent evt) {
        if (evt.getPropertyName().equals(JFileChooser.FILE_FILTER_CHANGED_PROPERTY)) {
            Object obj = evt.getSource();
            if (obj instanceof JFileChooser) {
                JFileChooser fileChooser = (JFileChooser) obj;
                javax.swing.filechooser.FileFilter fileFilter = fileChooser.getFileFilter();
                if (fileFilter instanceof ImageFormatFileFilter) {
                    String extension = ((ImageFormatFileFilter) fileFilter).getExtension();
                    if (extension.equals(ImageFormatFileFilter.PNG)) {
                        imageQuality.setEnabled(false);
                        imageQualityEnabled = false;
                    } else {
                        imageQuality.setEnabled(true);
                        imageQualityEnabled = true;
                    }
                }
            }
        }
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        this.add(c, gbc);
    }

}
