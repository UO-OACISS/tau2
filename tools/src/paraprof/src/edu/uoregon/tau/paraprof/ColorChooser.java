/*
 * ParaProf.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.colorchooser.*;
import edu.uoregon.tau.dms.dss.*;

public class ColorChooser implements WindowListener {
    public ColorChooser(ParaProfTrial trial, SavedPreferences savedPreferences) {
        this.trial = trial;

        if (savedPreferences != null) {
            colors = savedPreferences.getColors();
            groupColors = savedPreferences.getGroupColors();
            functionHighlightColor = savedPreferences.getHighlightColor();
            groupHighlightColor = savedPreferences.getGroupHighlightColor();
            userEventHighlightColor = savedPreferences.getUserEventHightlightColor();
            miscFunctionColor = savedPreferences.getMiscFunctionColor();
        } else {
            //Set the default colours.
            this.setDefaultColors();
            this.setDefaultGroupColors();
        }
    }

    public void showColorChooser() {
        if (!clrChooserFrameShowing) {
            //Bring up the color chooser frame.
            clrChooserFrame = new ColorChooserFrame(trial, this);
            clrChooserFrame.addWindowListener(this);
            clrChooserFrame.show();
            clrChooserFrameShowing = true;
        } else {
            //Just bring it to the foreground.
            clrChooserFrame.show();
        }
    }

    public void setSavedColors() {
        ParaProf.savedPreferences.setColors(colors);
        ParaProf.savedPreferences.setGroupColors(groupColors);
        ParaProf.savedPreferences.setHighlightColor(functionHighlightColor);
        ParaProf.savedPreferences.setGroupHighlightColor(groupHighlightColor);
        ParaProf.savedPreferences.setMiscFunctionColor(miscFunctionColor);
    }

    public int getNumberOfColors() {
        int tmpInt = -1;
        tmpInt = colors.size();

        return tmpInt;
    }

    public int getNumberOfGroupColors() {
        int tmpInt = -1;
        tmpInt = groupColors.size();
        return tmpInt;
    }

    public void addColor(Color color) {
        colors.add(color);
    }

    public void setColor(Color color, int location) {
        colors.setElementAt(color, location);
    }

    public Color getColor(int location) {
        Color color = null;
        color = (Color) colors.elementAt(location);

        return color;
    }

    public Vector getColors() {
        return colors;
    }

    public void addGroupColor(Color color) {
        groupColors.add(color);
    }

    public void setGroupColor(Color color, int location) {
        groupColors.setElementAt(color, location);
    }

    public Color getGroupColor(int location) {
        return (Color) groupColors.elementAt(location);
    }

    public Vector getGroupColors() {
        return groupColors;
    }

    public void setHighlightColor(Color highlightColor) {
        this.functionHighlightColor = highlightColor;
    }

    public Color getHighlightColor() {
        return functionHighlightColor;
    }

    public void setHighlightedFunction(Function func) {
        this.highlightedFunction = func;
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    public Function getHighlightedFunction() {
        return this.highlightedFunction;
    }

    public void toggleHighlightedFunction(Function function) {
        if (highlightedFunction == function)
            highlightedFunction = null;
        else
            highlightedFunction = function;
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    public void setGroupHighlightColor(Color groupHighlightColor) {
        this.groupHighlightColor = groupHighlightColor;
    }

    public Color getGroupHighlightColor() {
        return groupHighlightColor;
    }

    public void setHighlightedGroup(Group group) {
        this.highlightedGroup = group;
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    public Group getHighlightedGroup() {
        return highlightedGroup;
    }

    public void toggleHighlightedGroup(Group group) {
        if (highlightedGroup == group)
            highlightedGroup = null;
        else
            highlightedGroup = group;
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    // User Event Colors
    public void setUserEventHightlightColor(Color userEventHighlightColor) {
        this.userEventHighlightColor = userEventHighlightColor;
    }

    public Color getUserEventHighlightColor() {
        return userEventHighlightColor;
    }

    public void setHighlightedUserEvent(UserEvent userEvent) {
        this.highlightedUserEvent = userEvent;
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    public UserEvent getHighlightedUserEvent() {
        return highlightedUserEvent;
    }

    public void toggleHighlightedUserEvent(UserEvent userEvent) {
        if (highlightedUserEvent == userEvent)
            highlightedUserEvent = null;
        else
            highlightedUserEvent = userEvent;
        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
    }

    // User Event Colors

    public void setMiscFunctionColor(Color miscFunctionColor) {
        this.miscFunctionColor = miscFunctionColor;
    }

    public Color getMiscFunctionColor() {
        return miscFunctionColor;
    }

    //A function which sets the colors vector to be the default set.
    public void setDefaultColors() {
        //Clear the colors vector.
        colors.clear();

        //Add the default colours.
        addColor(new Color(70, 156, 168));
        addColor(new Color(255, 153, 0));

        addColor(new Color(0, 51, 255));

        addColor(new Color(102, 0, 51));
        addColor(new Color(221, 232, 30));
        addColor(new Color(0, 255, 0));
        addColor(new Color(121, 196, 144));
        addColor(new Color(86, 88, 112));

        addColor(new Color(151, 204, 255));
        addColor(new Color(102, 102, 255));
        addColor(new Color(0, 102, 102));
        addColor(new Color(204, 255, 51));
        addColor(new Color(102, 132, 25));
        addColor(new Color(255, 204, 153));
        addColor(new Color(204, 0, 204));
        addColor(new Color(0, 102, 102));
        addColor(new Color(204, 204, 255));
        addColor(new Color(61, 104, 63));
        addColor(new Color(102, 255, 255));
        addColor(new Color(255, 102, 102));
        addColor(new Color(119, 71, 145));
        addColor(new Color(255, 204, 204));
        addColor(new Color(240, 97, 159));
        addColor(new Color(0, 102, 153));
    }

    //A function which sets the groupColors vector to be the default set.
    public void setDefaultGroupColors() {
        //Clear the globalColors vector.
        groupColors.clear();

        //Add the default colours.
        addGroupColor(new Color(102, 0, 102));
        addGroupColor(new Color(51, 51, 0));
        addGroupColor(new Color(204, 0, 51));
        addGroupColor(new Color(0, 102, 102));
        addGroupColor(new Color(255, 255, 102));
        addGroupColor(new Color(0, 0, 102));
        addGroupColor(new Color(153, 153, 255));
        addGroupColor(new Color(255, 51, 0));
        addGroupColor(new Color(255, 153, 0));
        addGroupColor(new Color(255, 102, 102));
        addGroupColor(new Color(51, 0, 51));
        addGroupColor(new Color(255, 255, 102));
    }

    //Sets the colors of the given TrialData.
    //If the selection is equal to -1, then set the colors in all the
    // sets,
    //otherwise, just set the ones for the specified set.
    public void setColors(ParaProfTrial ppTrial, int selection) {

        TrialData trialData = ppTrial.getTrialData();
        if ((selection == -1) || (selection == 0)) {
            int numberOfColors = this.getNumberOfColors();

            DataSorter dataSorter = new DataSorter(ppTrial);
            Vector list = dataSorter.getFunctionProfiles(-1, -1, -1, 20);

            for (int i = 0; i < list.size(); i++) {
                Function func = ((PPFunctionProfile) list.get(i)).getFunction();
                func.setColor(this.getColor(i % numberOfColors));
            }
        }

        if ((selection == -1) || (selection == 1)) {
            int numberOfColors = this.getNumberOfGroupColors();
            for (Iterator i = trialData.getGroups(); i.hasNext();) {
                Group group = (Group) i.next();
                group.setColor(this.getGroupColor((group.getID()) % numberOfColors));
            }
        }

        if ((selection == -1) || (selection == 2)) {
            int numberOfColors = this.getNumberOfColors();
            for (Iterator i = trialData.getUserEvents(); i.hasNext();) {
                UserEvent userEvent = (UserEvent) i.next();
                userEvent.setColor(this.getColor((userEvent.getID()) % numberOfColors));
            }
        }
    }

    //Window Listener code.
    public void windowClosed(WindowEvent winevt) {
    }

    public void windowIconified(WindowEvent winevt) {
    }

    public void windowOpened(WindowEvent winevt) {
    }

    public void windowClosing(WindowEvent winevt) {
        if (winevt.getSource() == clrChooserFrame) {
            clrChooserFrameShowing = false;
        }
    }

    public void windowDeiconified(WindowEvent winevt) {
    }

    public void windowActivated(WindowEvent winevt) {
    }

    public void windowDeactivated(WindowEvent winevt) {
    }

    //####################################
    //Instance Data.
    //####################################
    private ParaProfTrial trial = null;
    private Vector colors = new Vector();
    private Vector groupColors = new Vector();
    //    private Color highlightColor = Color.red;
    //    private int highlightColorID = -1;

    private Color functionHighlightColor = Color.red;
    private Function highlightedFunction = null;

    //private Color groupHighlightColor = new Color(0, 255, 255);
    // private int groupHighlightColorID = -1;

    private Color groupHighlightColor = new Color(0, 255, 255);
    private Group highlightedGroup = null;

    //private Color userEventHighlightColor = new Color(255, 255, 0);
    //private int userEventHighlightColorID = -1;

    private Color userEventHighlightColor = new Color(255, 255, 0);
    private UserEvent highlightedUserEvent = null;

    private Color miscFunctionColor = Color.black;
    private boolean clrChooserFrameShowing = false; //For determining whether
    // the clrChooserFrame is
    // showing.
    private ColorChooserFrame clrChooserFrame;
    //####################################
    //End - Instance Data.
    //####################################
}

class ColorChooserFrame extends JFrame implements ActionListener {
    public ColorChooserFrame(ParaProfTrial trial, ColorChooser colorChooser) {
        this.trial = trial;
        this.colorChooser = colorChooser;
        numberOfColors = trial.getColorChooser().getNumberOfColors();

        //Window Stuff.
        setLocation(new Point(100, 100));
        setSize(new Dimension(850, 450));

        //####################################
        //Code to generate the menus.
        //####################################
        JMenuBar mainMenu = new JMenuBar();

        //######
        //File menu.
        //######
        JMenu fileMenu = new JMenu("File");

        JMenuItem closeItem = new JMenuItem("Close This Window");
        closeItem.addActionListener(this);
        fileMenu.add(closeItem);

        JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
        exitItem.addActionListener(this);
        fileMenu.add(exitItem);
        //######
        //File menu.
        //######

        //######
        //Help menu.
        //######
        /*
         * JMenu helpMenu = new JMenu("Help");
         * 
         * //Add a menu item. JMenuItem aboutItem = new JMenuItem("About
         * Racy"); helpMenu.add(aboutItem);
         * 
         * //Add a menu item. JMenuItem showHelpWindowItem = new
         * JMenuItem("Show Help Window");
         * showHelpWindowItem.addActionListener(this);
         * helpMenu.add(showHelpWindowItem);
         */
        //######
        //Help menu.
        //######
        //Now, add all the menus to the main menu.
        mainMenu.add(fileMenu);
        //mainMenu.add(helpMenu);
        setJMenuBar(mainMenu);
        //####################################
        //Code to generate the menus.
        //####################################

        //####################################
        //Create and add the components.
        //####################################
        //Setting up the layout system for the main window.
        Container contentPane = getContentPane();
        GridBagLayout gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //Create a new ColorChooser.
        clrChooser = new JColorChooser();
        clrModel = clrChooser.getSelectionModel();

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;

        //First add the label.
        JLabel titleLabel = new JLabel("ParaProf Color Set.");
        titleLabel.setFont(new Font("SansSerif", Font.ITALIC, 14));
        addCompItem(titleLabel, gbc, 0, 0, 1, 1);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 0;
        gbc.weighty = 0;

        //Create and add color list.
        listModel = new DefaultListModel();
        colorList = new JList(listModel);
        colorList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        colorList.setCellRenderer(new CustomCellRenderer(trial));
        JScrollPane sp = new JScrollPane(colorList);
        addCompItem(sp, gbc, 0, 1, 1, 5);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addColorButton = new JButton("Add Color");
        addColorButton.addActionListener(this);
        addCompItem(addColorButton, gbc, 1, 1, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addGroupColorButton = new JButton("Add Group Color");
        addGroupColorButton.addActionListener(this);
        addCompItem(addGroupColorButton, gbc, 1, 2, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        deleteColorButton = new JButton("Delete Selected Color");
        deleteColorButton.addActionListener(this);
        addCompItem(deleteColorButton, gbc, 1, 3, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        updateColorButton = new JButton("Update Selected Color");
        updateColorButton.addActionListener(this);
        addCompItem(updateColorButton, gbc, 1, 4, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        restoreDefaultsButton = new JButton("Restore Defaults");
        restoreDefaultsButton.addActionListener(this);
        addCompItem(restoreDefaultsButton, gbc, 1, 5, 1, 1);

        //Add the JColorChooser.
        addCompItem(clrChooser, gbc, 2, 0, 1, 6);
        //####################################
        //End - Create and add the components.
        //####################################

        //Now populate the colour list.
        populateColorList();
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();
            String arg = evt.getActionCommand();

            if (EventSrc instanceof JMenuItem) {
                if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);
                } else if (arg.equals("Close This Window")) {
                    setVisible(false);
                }
            } else if (EventSrc instanceof JButton) {
                if (arg.equals("Add Color")) {
                    Color color = clrModel.getSelectedColor();
                    (colorChooser.getColors()).add(color);
                    listModel.clear();
                    populateColorList();
                    //Update the TrialData.
                    colorChooser.setColors(trial, 0);
                    //Update the listeners.
                    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                } else if (arg.equals("Add Group Color")) {
                    Color color = clrModel.getSelectedColor();
                    (colorChooser.getGroupColors()).add(color);
                    listModel.clear();
                    populateColorList();
                    //Update the TrialData.
                    colorChooser.setColors(trial, 1);
                    //Update the listeners.
                    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                } else if (arg.equals("Delete Selected Color")) {
                    //Get the currently selected items and cycle through them.
                    int[] values = colorList.getSelectedIndices();
                    for (int i = 0; i < values.length; i++) {
                        if ((values[i]) < trial.getColorChooser().getNumberOfColors()) {
                            System.out.println("The value being deleted is: " + values[i]);
                            listModel.removeElementAt(values[i]);
                            (colorChooser.getColors()).removeElementAt(values[i]);
                            //Update the TrialData.
                            colorChooser.setColors(trial, 0);
                        } else if ((values[i]) < (trial.getColorChooser().getNumberOfColors())
                                + (trial.getColorChooser().getNumberOfGroupColors())) {
                            System.out.println("The value being deleted is: " + values[i]);
                            listModel.removeElementAt(values[i]);
                            (colorChooser.getGroupColors()).removeElementAt(values[i]
                                    - (trial.getColorChooser().getNumberOfColors()));
                            //Update the TrialData.
                            colorChooser.setColors(trial, 1);
                        }
                    }

                    //Update the listeners.
                    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                } else if (arg.equals("Update Selected Color")) {
                    Color color = clrModel.getSelectedColor();
                    //Get the currently selected items and cycle through them.
                    int[] values = colorList.getSelectedIndices();
                    for (int i = 0; i < values.length; i++) {
                        listModel.setElementAt(color, values[i]);
                        int totalNumberOfColors = (trial.getColorChooser().getNumberOfColors())
                                + (trial.getColorChooser().getNumberOfGroupColors());
                        if ((values[i]) == (totalNumberOfColors)) {
                            trial.getColorChooser().setHighlightColor(color);
                        } else if ((values[i]) == (totalNumberOfColors + 1)) {
                            trial.getColorChooser().setGroupHighlightColor(color);
                        } else if ((values[i]) == (totalNumberOfColors + 2)) {
                            trial.getColorChooser().setUserEventHightlightColor(color);
                        } else if ((values[i]) == (totalNumberOfColors + 3)) {
                            trial.getColorChooser().setMiscFunctionColor(color);
                        } else if ((values[i]) < trial.getColorChooser().getNumberOfColors()) {
                            colorChooser.setColor(color, values[i]);
                            //Update the TrialData.
                            colorChooser.setColors(trial, 0);
                        } else {
                            colorChooser.setGroupColor(color,
                                    (values[i] - trial.getColorChooser().getNumberOfColors()));
                            //Update the TrialData.
                            colorChooser.setColors(trial, 1);
                        }
                    }
                    //Update the listeners.
                    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                } else if (arg.equals("Restore Defaults")) {
                    colorChooser.setDefaultColors();
                    colorChooser.setDefaultGroupColors();
                    colorChooser.setHighlightColor(Color.red);
                    colorChooser.setGroupHighlightColor(new Color(0, 255, 255));
                    colorChooser.setUserEventHightlightColor(new Color(255, 255, 0));
                    colorChooser.setMiscFunctionColor(Color.black);
                    listModel.clear();
                    populateColorList();
                    //Update the TrialData.
                    colorChooser.setColors(trial, 0);
                    colorChooser.setColors(trial, 1);
                    //Update the listeners.
                    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                }
            }

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }

    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    void populateColorList() {
        Color color;
        for (Enumeration e = (colorChooser.getColors()).elements(); e.hasMoreElements();) {
            color = (Color) e.nextElement();
            listModel.addElement(color);
        }

        for (Enumeration e = (colorChooser.getGroupColors()).elements(); e.hasMoreElements();) {
            color = (Color) e.nextElement();
            listModel.addElement(color);
        }

        color = trial.getColorChooser().getHighlightColor();
        listModel.addElement(color);

        color = trial.getColorChooser().getGroupHighlightColor();
        listModel.addElement(color);

        color = trial.getColorChooser().getUserEventHighlightColor();
        listModel.addElement(color);

        color = trial.getColorChooser().getMiscFunctionColor();
        listModel.addElement(color);
    }

    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    private ColorChooser colorChooser;
    private ColorSelectionModel clrModel;
    private JColorChooser clrChooser;
    private DefaultListModel listModel;
    private JList colorList;
    private JButton addColorButton;
    private JButton addGroupColorButton;
    private JButton deleteColorButton;
    private JButton updateColorButton;
    private JButton restoreDefaultsButton;
    private int numberOfColors = -1;
    //####################################
    //End - Instance data.
    //####################################
}

class CustomCellRenderer implements ListCellRenderer {
    CustomCellRenderer(ParaProfTrial trial) {
        this.trial = trial;
    }

    public Component getListCellRendererComponent(final JList list, final Object value, final int index,
            final boolean isSelected, final boolean cellHasFocus) {
        return new JPanel() {
            public void paintComponent(Graphics g) {
                super.paintComponent(g);
                Color inColor = (Color) value;

                int xSize = 0;
                int ySize = 0;
                int maxXNumFontSize = 0;
                int maxXFontSize = 0;
                int maxYFontSize = 0;
                int thisXFontSize = 0;
                int thisYFontSize = 0;
                int barHeight = 0;

                //For this, I will not allow changes in font size.
                barHeight = 12;

                //Create font.
                Font font = new Font(trial.getPreferences().getParaProfFont(), Font.PLAIN, barHeight);
                g.setFont(font);
                FontMetrics fmFont = g.getFontMetrics(font);

                maxXFontSize = fmFont.getAscent();
                maxYFontSize = fmFont.stringWidth("0000,0000,0000");

                xSize = getWidth();
                ySize = getHeight();

                String tmpString1 = new String("00" + (trial.getColorChooser().getNumberOfColors()));
                maxXNumFontSize = fmFont.stringWidth(tmpString1);

                String tmpString2 = new String(inColor.getRed() + "," + inColor.getGreen() + ","
                        + inColor.getBlue());
                thisXFontSize = fmFont.stringWidth(tmpString2);
                thisYFontSize = maxYFontSize;

                g.setColor(isSelected ? list.getSelectionBackground() : list.getBackground());
                g.fillRect(0, 0, (5 + maxXNumFontSize + 5), ySize);

                int xStringPos1 = 5;
                int yStringPos1 = (ySize - 5);
                g.setColor(isSelected ? list.getSelectionForeground() : list.getForeground());

                int totalNumberOfColors = (trial.getColorChooser().getNumberOfColors())
                        + (trial.getColorChooser().getNumberOfGroupColors());

                if (index == totalNumberOfColors) {
                    g.drawString(("" + ("MHC")), xStringPos1, yStringPos1);
                } else if (index == (totalNumberOfColors + 1)) {
                    g.drawString(("" + ("GHC")), xStringPos1, yStringPos1);
                } else if (index == (totalNumberOfColors + 2)) {
                    g.drawString(("" + ("UHC")), xStringPos1, yStringPos1);
                } else if (index == (totalNumberOfColors + 3)) {
                    g.drawString(("" + ("MPC")), xStringPos1, yStringPos1);
                } else if (index < (trial.getColorChooser().getNumberOfColors())) {
                    g.drawString(("" + (index + 1)), xStringPos1, yStringPos1);
                } else {
                    g.drawString(("G" + (index - (trial.getColorChooser().getNumberOfColors()) + 1)),
                            xStringPos1, yStringPos1);
                }

                g.setColor(inColor);
                g.fillRect((5 + maxXNumFontSize + 5), 0, 50, ySize);

                //Just a sanity check.
                if ((xSize - 50) > 0) {
                    g.setColor(isSelected ? list.getSelectionBackground() : list.getBackground());
                    g.fillRect((5 + maxXNumFontSize + 5 + 50), 0, (xSize - 50), ySize);
                }

                int xStringPos2 = 50 + (((xSize - 50) - thisXFontSize) / 2);
                int yStringPos2 = (ySize - 5);

                g.setColor(isSelected ? list.getSelectionForeground() : list.getForeground());
                g.drawString(tmpString2, xStringPos2, yStringPos2);
            }

            public Dimension getPreferredSize() {
                int xSize = 0;
                int ySize = 0;
                int maxXNumFontSize = 0;
                int maxXFontSize = 0;
                int maxYFontSize = 0;
                int barHeight = 12;

                //Create font.
                Font font = new Font(trial.getPreferences().getParaProfFont(), Font.PLAIN, barHeight);
                Graphics g = getGraphics();
                FontMetrics fmFont = g.getFontMetrics(font);

                String tmpString = new String("00" + (trial.getColorChooser().getNumberOfColors()));
                maxXNumFontSize = fmFont.stringWidth(tmpString);

                maxXFontSize = fmFont.stringWidth("0000,0000,0000");
                maxYFontSize = fmFont.getAscent();

                xSize = (maxXNumFontSize + 10 + 50 + maxXFontSize + 20);
                ySize = (10 + maxYFontSize);

                return new Dimension(xSize, ySize);
            }
        };
    }

    //Instance data.
    private ParaProfTrial trial = null;
}