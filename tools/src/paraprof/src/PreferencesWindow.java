/*
 * Preferences.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 * 
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.*;

import javax.swing.*;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.DataSource;

public class PreferencesWindow extends JFrame implements ActionListener, Observer {

    private boolean mShown = false;

    //References for some of the components for this frame.
    private PrefSpacingPanel prefSpacingPanel;
    private JCheckBox bold;
    private JCheckBox italic;

    private JComboBox fontComboBox;
    private JLabel barHeightLabel = new JLabel("Size");
    private JSlider barHeightSlider = new JSlider(SwingConstants.HORIZONTAL, 0, 40, 0);

    private Preferences preferences;

    private String fontName = "SansSerif";
    private int fontStyle = Font.PLAIN;
    private int fontSize = 12;
    private Font font;

    private JComboBox unitsBox;
    private JCheckBox autoLabelsBox = new JCheckBox("Auto label node/context/threads");
    private JCheckBox showValuesAsPercentBox = new JCheckBox("Show Values as Percent");
    private JCheckBox showPathTitleInReverseBox = new JCheckBox("Show Path Title in Reverse");
    private JCheckBox reverseCallPathsBox = new JCheckBox("Reverse Call Paths");
    private JCheckBox meanIncludeNullBox = new JCheckBox(
            "<html>Interpret threads that do not call a given function as a 0 value for statistics computation</html>");
    private JCheckBox generateIntermediateCallPathDataBox = new JCheckBox(
            "<html>Generate data for reverse calltree<br>(requires lots of memory)<br>(does not apply to currently loaded profiles)</html>");

    private JCheckBox showSourceLocationsBox = new JCheckBox("<html>Show Source Locations</html>");

    public PreferencesWindow(Preferences preferences) {
        this.preferences = preferences;

        String[] items = new String[4];
        items[0] = "Microseconds";
        items[1] = "Milliseconds";
        items[2] = "Seconds";
        items[3] = "hr:mm:ss";
        unitsBox = new JComboBox(items);

        meanIncludeNullBox.setToolTipText("<html>There are two methods for computing the mean for a given function over all threads:<br>1. Add up the values for that function for each thread and divide by the total number of threads.<br>2. Add up the values for that function for each thread and divide by the number of threads that actually called that function.<br><br>This has the effect that if a particular function is only called by 2 threads:<br>The first method will show the mean value as 1/N where N is the total number of threads.<br>The second method will only divide by 2.<br><br>This option also affects standard deviation computation.");
        reverseCallPathsBox.setToolTipText("<html>If this option is enabled, call path names will be shown in reverse<br>(e.g. \"C &lt;= B &lt;= A\" vs. \"A =&gt; B =&gt; C\")");
        generateIntermediateCallPathDataBox.setToolTipText("<html>If this option is enabled, then the reverse calltree will be available.<br>However, it requires additional memory and should be disabled if the JVM<br>runs out of memory on large callpath datasets.</html>");
        autoLabelsBox.setToolTipText("<html>If this option is enabled, execution thread labels \"n,c,t 0,0,0\" will be shortened based on the execution type (MPI, threaded, hybrid)</html>");

        // Set preferences based on saved values.
        fontName = preferences.getFontName();
        fontStyle = preferences.getFontStyle();
        fontSize = preferences.getFontSize();
        unitsBox.setSelectedIndex(preferences.getUnits());
        showValuesAsPercentBox.setSelected(preferences.getShowValuesAsPercent());
        showPathTitleInReverseBox.setSelected(preferences.getShowPathTitleInReverse());
        reverseCallPathsBox.setSelected(preferences.getReversedCallPaths());
        autoLabelsBox.setSelected(preferences.getAutoLabels());
        meanIncludeNullBox.setSelected(!preferences.getComputeMeanWithoutNulls());
        generateIntermediateCallPathDataBox.setSelected(preferences.getGenerateIntermediateCallPathData());
        showSourceLocationsBox.setSelected(preferences.getShowSourceLocation());

        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Get available fonts and initialize the fontComboBox..
        GraphicsEnvironment gE = GraphicsEnvironment.getLocalGraphicsEnvironment();
        String[] fontFamilyNames = gE.getAvailableFontFamilyNames();
        fontComboBox = new JComboBox(fontFamilyNames);

        fontComboBox.addActionListener(this);

        //Now initialize the panels.
        prefSpacingPanel = new PrefSpacingPanel();

        //Window Stuff.
        setTitle("TAU: ParaProf: Preferences");
        ParaProfUtils.setFrameIcon(this);

        int windowWidth = 650;
        int windowHeight = 520;
        setSize(new java.awt.Dimension(windowWidth, windowHeight));

        //There is really no need to resize this window.
        setResizable(true);

        //Set the window to come up in the center of the screen.
        int xPosition = 0;
        int yPosition = 0;

        setLocation(xPosition, yPosition);

        //End - Window Stuff.

        setupMenus();

        //####################################
        //Create and add the components
        //####################################

        //Setup the layout system for the main window.
        Container contentPane = getContentPane();
        contentPane.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        JScrollPane jScrollPane = new JScrollPane(prefSpacingPanel);
        jScrollPane.setPreferredSize(new Dimension(300, 200));

        bold = new JCheckBox("Bold");
        bold.addActionListener(this);
        italic = new JCheckBox("Italic");
        italic.addActionListener(this);

        setControls();

        barHeightSlider.setPaintTicks(true);
        barHeightSlider.setMajorTickSpacing(10);
        barHeightSlider.setMinorTickSpacing(5);
        barHeightSlider.setPaintLabels(true);
        barHeightSlider.addChangeListener(prefSpacingPanel);

        JPanel fontPanel = new JPanel();
        fontPanel.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), "Font"));
        fontPanel.setLayout(new GridBagLayout());

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;
        Utility.addCompItem(fontPanel, fontComboBox, gbc, 0, 0, 2, 1);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        Utility.addCompItem(fontPanel, bold, gbc, 0, 1, 1, 1);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        Utility.addCompItem(fontPanel, italic, gbc, 0, 2, 1, 1);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;
        Utility.addCompItem(fontPanel, barHeightLabel, gbc, 1, 1, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0.25;
        gbc.weighty = 0.25;
        Utility.addCompItem(fontPanel, barHeightSlider, gbc, 1, 2, 1, 1);

        JPanel defaultsPanel = new JPanel();
        defaultsPanel.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), "Window defaults"));
        defaultsPanel.setLayout(new GridBagLayout());

        JLabel unitsLabel = new JLabel("Units");

        Utility.addCompItem(defaultsPanel, unitsLabel, gbc, 0, 0, 1, 1);
        Utility.addCompItem(defaultsPanel, unitsBox, gbc, 1, 0, 1, 1);
        Utility.addCompItem(defaultsPanel, showValuesAsPercentBox, gbc, 0, 1, 2, 1);

        JPanel settingsPanel = new JPanel();
        settingsPanel.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), "Settings"));
        settingsPanel.setLayout(new GridBagLayout());
        Utility.addCompItem(settingsPanel, showPathTitleInReverseBox, gbc, 0, 2, 2, 1);
        Utility.addCompItem(settingsPanel, reverseCallPathsBox, gbc, 0, 3, 2, 1);
        Utility.addCompItem(settingsPanel, meanIncludeNullBox, gbc, 0, 4, 2, 1);
        Utility.addCompItem(settingsPanel, generateIntermediateCallPathDataBox, gbc, 0, 5, 2, 1);
        Utility.addCompItem(settingsPanel, showSourceLocationsBox, gbc, 0, 6, 2, 1);
        Utility.addCompItem(settingsPanel, autoLabelsBox, gbc, 0, 7, 2, 1);

        gbc.fill = GridBagConstraints.BOTH;

        addCompItem(fontPanel, gbc, 0, 0, 1, 1);
        addCompItem(defaultsPanel, gbc, 0, 1, 1, 1);
        addCompItem(settingsPanel, gbc, 1, 1, 1, 1);

        JButton defaultButton = new JButton("Restore Defaults");
        defaultButton.addActionListener(this);
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(defaultButton, gbc, 0, 2, 1, 1);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0.5;
        gbc.weighty = 0.5;
        addCompItem(jScrollPane, gbc, 1, 0, 1, 1);

        JPanel applyCancelPanel = new JPanel();

        JButton applyButton = new JButton("Apply");
        applyButton.addActionListener(this);
        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(this);

        applyCancelPanel.add(applyButton);
        applyCancelPanel.add(cancelButton);
        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addCompItem(applyCancelPanel, gbc, 1, 2, 1, 1);

        setSavedPreferences();

    }

    void setControls() {
        int tmpInt = fontComboBox.getItemCount();
        int counter = 0;
        //We should always have some fonts available, so this should be safe.
        String tmpString = (String) fontComboBox.getItemAt(counter);
        while ((counter < tmpInt) && (!(fontName.equals(tmpString)))) {
            counter++;
            tmpString = (String) fontComboBox.getItemAt(counter);
        }

        if (counter == tmpInt) {
            //The default font was not available. Indicate an error.
            System.out.println("The default font was not found!  This is not a good thing as it is a default Java font!");
        } else {
            fontComboBox.setSelectedIndex(counter);
        }

        bold.setSelected((fontStyle == Font.BOLD) || (fontStyle == (Font.BOLD | Font.ITALIC)));
        italic.setSelected((fontStyle == (Font.PLAIN | Font.ITALIC)) || (fontStyle == (Font.BOLD | Font.ITALIC)));
        barHeightSlider.setValue(fontSize);
    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        JMenu fileMenu = new JMenu("File");

        JMenuItem jmenuItem;

        jmenuItem = new JMenuItem("Load Preferences...");
        jmenuItem.addActionListener(this);
        fileMenu.add(jmenuItem);
        jmenuItem = new JMenuItem("Save Preferences...");
        jmenuItem.addActionListener(this);
        fileMenu.add(jmenuItem);

        JMenuItem editColorItem = new JMenuItem("Edit Default Colors");
        editColorItem.addActionListener(this);
        fileMenu.add(editColorItem);

        //        JMenuItem saveColorItem = new JMenuItem("Save Color Map");
        //        saveColorItem.addActionListener(this);
        //        fileMenu.add(saveColorItem);
        //
        //        JMenuItem loadColorItem = new JMenuItem("Load Color Map");
        //        loadColorItem.addActionListener(this);
        //        fileMenu.add(loadColorItem);

        JMenuItem newItem = new JMenuItem("Edit Source Directories");
        newItem.addActionListener(this);
        fileMenu.add(newItem);

        newItem = new JMenuItem("Show Color Map");
        newItem.addActionListener(this);
        fileMenu.add(newItem);

        JMenuItem closeItem = new JMenuItem("Apply and Close Window");
        closeItem.addActionListener(this);
        fileMenu.add(closeItem);

        JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
        exitItem.addActionListener(this);
        fileMenu.add(exitItem);

        mainMenu.add(fileMenu);

        setJMenuBar(mainMenu);
    }

    public void showPreferencesWindow(Component invoker) {
        //The path to data might have changed, therefore, reset the title.
        this.setTitle("ParaProf Preferences");
        this.setLocation(WindowPlacer.getNewLocation(this, invoker));
        this.setVisible(true);
    }

    public void loadSavedPreferences() {
        this.preferences = ParaProf.preferences;
        // Set preferences based on saved values.
        fontName = preferences.getFontName();
        fontStyle = preferences.getFontStyle();
        fontSize = preferences.getFontSize();
        font = null;
    }

    public Font getFont() {
        if (font == null) {
            font = new Font(fontName, fontStyle, fontSize);
        }
        return font;
    }

    public void setSavedPreferences() {
        ParaProf.preferences.setFontName(fontName);
        ParaProf.preferences.setFontStyle(fontStyle);
        ParaProf.preferences.setFontSize(fontSize);
        ParaProf.preferences.setUnits(unitsBox.getSelectedIndex());
        ParaProf.preferences.setShowValuesAsPercent(showValuesAsPercentBox.isSelected());
        ParaProf.preferences.setShowPathTitleInReverse(showPathTitleInReverseBox.isSelected());
        ParaProf.preferences.setReversedCallPaths(reverseCallPathsBox.isSelected());
        ParaProf.preferences.setAutoLabels(autoLabelsBox.isSelected());
        ParaProf.preferences.setComputeMeanWithoutNulls(!meanIncludeNullBox.isSelected());
        ParaProf.preferences.setGenerateIntermediateCallPathData(generateIntermediateCallPathDataBox.isSelected());
        ParaProf.preferences.setShowSourceLocation(showSourceLocationsBox.isSelected());
    }

    public String getFontName() {
        return fontName;
    }

    public int getFontStyle() {
        return fontStyle;
    }

    public int getFontSize() {
        return fontSize;
    }

    public void setFontSize(int fontSize) {
        this.fontSize = fontSize;
        font = null;
    }

    public void updateFontSize() {
        fontSize = Math.max(1, barHeightSlider.getValue());
        font = null;
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();
            String arg = evt.getActionCommand();
            if (EventSrc instanceof JMenuItem) {

                if (arg.equals("Show Color Map")) {
                    ParaProf.colorMap.showColorMap(this);
                } else if (arg.equals("Edit Source Directories")) {
                    ParaProf.getDirectoryManager().display(this);
                } else if (arg.equals("Load Preferences...")) {

                    JFileChooser fileChooser = new JFileChooser();

                    //Set the directory to the current directory.
                    fileChooser.setCurrentDirectory(new File("."));

                    //Bring up the file chooser.
                    int resultValue = fileChooser.showOpenDialog(this);

                    if (resultValue == JFileChooser.APPROVE_OPTION) {
                        File file = fileChooser.getSelectedFile();

                        try {
                            ParaProf.loadPreferences(file);
                        } catch (Exception e) {
                            JOptionPane.showMessageDialog(this, "Error loading preferences!", "ParaProf Preferences",
                                    JOptionPane.ERROR_MESSAGE);

                        }
                        loadSavedPreferences();
                        setControls();
                    }

                } else if (arg.equals("Save Preferences...")) {

                    JFileChooser fileChooser = new JFileChooser();
                    fileChooser.setCurrentDirectory(new File("."));
                    int resultValue = fileChooser.showSaveDialog(this);

                    if (resultValue == JFileChooser.APPROVE_OPTION) {

                        File file = fileChooser.getSelectedFile();

                        if (ParaProf.savePreferences(file) == false) {
                            JOptionPane.showMessageDialog(this, "Error Saving preferences!", "ParaProf Preferences",
                                    JOptionPane.ERROR_MESSAGE);
                        }
                    }

                } else if (arg.equals("Edit Default Colors")) {
                    ParaProf.colorChooser.showColorChooser(this);
                } else if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);
                } else if (arg.equals("Apply and Close Window")) {
                    setVisible(false);
                    apply();
                }

            } else if (EventSrc instanceof JCheckBox) {
                if (arg.equals("Bold")) {
                    if (italic.isSelected()) {
                        if (bold.isSelected()) {
                            fontStyle = Font.BOLD | Font.ITALIC;
                        } else {
                            fontStyle = Font.PLAIN | Font.ITALIC;
                        }
                    } else {
                        if (bold.isSelected()) {
                            fontStyle = Font.BOLD;
                        } else {
                            fontStyle = Font.PLAIN;
                        }
                    }

                    prefSpacingPanel.repaint();
                } else if (arg.equals("Italic")) {
                    if (italic.isSelected()) {
                        if (bold.isSelected()) {
                            fontStyle = Font.BOLD | Font.ITALIC;
                        } else {
                            fontStyle = Font.PLAIN | Font.ITALIC;
                        }
                    } else {
                        if (bold.isSelected()) {
                            fontStyle = Font.BOLD;
                        } else {
                            fontStyle = Font.PLAIN;
                        }
                    }

                    prefSpacingPanel.repaint();
                }
            } else if (EventSrc == fontComboBox) {
                fontName = (String) fontComboBox.getSelectedItem();
                prefSpacingPanel.repaint();
            } else if (EventSrc instanceof JButton) {
                if (arg.equals("Apply")) {
                    apply();
                } else if (arg.equals("Cancel")) {
                    setVisible(false);
                    fontName = preferences.getFontName();
                    fontStyle = preferences.getFontStyle();
                    fontSize = preferences.getFontSize();
                    setControls();
                } else if (arg.equals("Restore Defaults")) {
                    fontName = "SansSerif";
                    fontStyle = Font.PLAIN;
                    fontSize = 12;
                    unitsBox.setSelectedIndex(2);
                    showValuesAsPercentBox.setSelected(false);
                    showPathTitleInReverseBox.setSelected(false);
                    reverseCallPathsBox.setSelected(false);
                    meanIncludeNullBox.setSelected(true);
                    autoLabelsBox.setSelected(true);
                    generateIntermediateCallPathDataBox.setSelected(false);
                    showSourceLocationsBox.setSelected(true);
                    setControls();
                }

            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    private void apply() {
        boolean needDataEvent = false;
        if (reverseCallPathsBox.isSelected() != ParaProf.preferences.getReversedCallPaths()) {
            needDataEvent = true;
        }

        if (autoLabelsBox.isSelected() != ParaProf.preferences.getAutoLabels()) {
            needDataEvent = true;
        }

        if (meanIncludeNullBox.isSelected() == ParaProf.preferences.getComputeMeanWithoutNulls()) {
            needDataEvent = true;
            DataSource.setMeanIncludeNulls(meanIncludeNullBox.isSelected());
            ParaProf.paraProfManagerWindow.recomputeStats();
        }

        this.font = null;
        setSavedPreferences();
        Vector trials = ParaProf.paraProfManagerWindow.getLoadedTrials();
        for (Iterator it = trials.iterator(); it.hasNext();) {
            ParaProfTrial ppTrial = (ParaProfTrial) it.next();
            ppTrial.updateRegisteredObjects("prefEvent");
            ppTrial.updateRegisteredObjects("dataEvent");
        }
    }

    public void update(Observable o, Object arg) {
        String tmpString = (String) arg;
        if (tmpString.equals("colorEvent")) {
            //Just need to call a repaint.
            prefSpacingPanel.repaint();
        }
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    public void addNotify() {
        super.addNotify();

        if (mShown)
            return;

        // resize frame to account for menubar
        JMenuBar jMenuBar = getJMenuBar();
        if (jMenuBar != null) {
            int jMenuBarHeight = jMenuBar.getPreferredSize().height;
            Dimension dimension = getSize();
            dimension.height += jMenuBarHeight;
            setSize(dimension);
        }

        mShown = true;
    }

    // Close the window when the close box is clicked
    void thisWindowClosing(java.awt.event.WindowEvent e) {
        setVisible(false);
        fontName = preferences.getFontName();
        fontStyle = preferences.getFontStyle();
        fontSize = preferences.getFontSize();
        setControls();
    }

}