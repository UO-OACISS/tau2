/*
 * StaticMainWindow.java
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
import javax.swing.event.*;
import java.awt.print.*;
import edu.uoregon.tau.dms.dss.*;

public class StaticMainWindow extends JFrame implements ActionListener, MenuListener, Observer, ChangeListener {

    
    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        JMenu subMenu = null;
        JMenuItem menuItem = null;

        //File menu.
        JMenu fileMenu = new JMenu("File");

        //Save menu.
        subMenu = new JMenu("Save ...");

        menuItem = new JMenuItem("Save Image");
        menuItem.addActionListener(this);
        subMenu.add(menuItem);

        fileMenu.add(subMenu);
        //End - Save menu.

        menuItem = new JMenuItem("Preferences...");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Print");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Close This Window");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        menuItem = new JMenuItem("Exit ParaProf!");
        menuItem.addActionListener(this);
        fileMenu.add(menuItem);

        fileMenu.addMenuListener(this);

        //Options menu.
        optionsMenu = new JMenu("Options");

        nameCheckBox = new JCheckBoxMenuItem("Sort By Name", false);
        nameCheckBox.addActionListener(this);
        optionsMenu.add(nameCheckBox);

        normalizeCheckBox = new JCheckBoxMenuItem("Normalize Bars", true);
        normalizeCheckBox.addActionListener(this);
        optionsMenu.add(normalizeCheckBox);

        orderByMeanCheckBox = new JCheckBoxMenuItem("Order By Mean", true);
        orderByMeanCheckBox.addActionListener(this);
        optionsMenu.add(orderByMeanCheckBox);

        orderCheckBox = new JCheckBoxMenuItem("Descending Order", true);
        orderCheckBox.addActionListener(this);
        optionsMenu.add(orderCheckBox);

        stackBarsCheckBox = new JCheckBoxMenuItem("Stack Bars Together", true);
        stackBarsCheckBox.addActionListener(this);
        optionsMenu.add(stackBarsCheckBox);

        slidersCheckBox = new JCheckBoxMenuItem("Display Sliders", false);
        slidersCheckBox.addActionListener(this);
        optionsMenu.add(slidersCheckBox);

        pathTitleCheckBox = new JCheckBoxMenuItem("Show Path Title in Reverse", true);
        pathTitleCheckBox.addActionListener(this);
        optionsMenu.add(pathTitleCheckBox);

        metaDataCheckBox = new JCheckBoxMenuItem("Show Meta Data in Panel", true);
        metaDataCheckBox.addActionListener(this);
        optionsMenu.add(metaDataCheckBox);

        //Windows menu
        windowsMenu = new JMenu("Windows");

        menuItem = new JMenuItem("Show ParaProf Manager");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Function Ledger");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Group Ledger");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show User Event Ledger");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Call Path Relations");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Show Full Call Graph");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        menuItem = new JMenuItem("Close All Sub-Windows");
        menuItem.addActionListener(this);
        windowsMenu.add(menuItem);

        windowsMenu.addMenuListener(this);

        //Help menu.
        JMenu helpMenu = new JMenu("Help");

        menuItem = new JMenuItem("Show Help Window");
        menuItem.addActionListener(this);
        helpMenu.add(menuItem);

        menuItem = new JMenuItem("About ParaProf");
        menuItem.addActionListener(this);
        helpMenu.add(menuItem);

        helpMenu.addMenuListener(this);

        //Now, add all the menus to the main menu.
        mainMenu.add(fileMenu);
        mainMenu.add(optionsMenu);
        mainMenu.add(windowsMenu);
        mainMenu.add(helpMenu);

        setJMenuBar(mainMenu);
    }
    
    public StaticMainWindow(ParaProfTrial ppTrial, boolean debug) {
        //This window needs to maintain a reference to its trial.
        this.trial = ppTrial;
        
        //Window Stuff.
        setTitle("ParaProf: " + ppTrial.getTrialIdentifier(true));

        int windowWidth = 750;
        int windowHeight = 400;
        setSize(new java.awt.Dimension(windowWidth, windowHeight));

        dataSorter = new DataSorter(ppTrial);

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Grab the screen size.
        Toolkit tk = Toolkit.getDefaultToolkit();
        Dimension screenDimension = tk.getScreenSize();
        int screenHeight = screenDimension.height;
        int screenWidth = screenDimension.width;

        int xPosition = (screenWidth - windowWidth) / 2;
        //Set the window to come up in the center of the screen.
        int yPosition = (screenHeight - windowHeight) / 2;

        setLocation(xPosition, yPosition);


        setupMenus();
        

        //Setting up the layout system for the main window.
        contentPane = getContentPane();
        gbl = new GridBagLayout();
        contentPane.setLayout(gbl);
        gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        //Panel and ScrollPane definition.
        panel = new StaticMainWindowPanel(ppTrial, this);
        sp = new JScrollPane(panel);
        this.setHeader();

        //Slider setup.
        //Do the slider stuff, but don't add. By default, sliders are off.
        String sliderMultipleStrings[] = { "1.00", "0.75", "0.50", "0.25", "0.10" };
        sliderMultiple = new JComboBox(sliderMultipleStrings);
        sliderMultiple.addActionListener(this);

        barLengthSlider.setPaintTicks(true);
        barLengthSlider.setMajorTickSpacing(5);
        barLengthSlider.setMinorTickSpacing(1);
        barLengthSlider.setPaintLabels(true);
        barLengthSlider.setSnapToTicks(true);
        barLengthSlider.addChangeListener(this);
        //End - Slider setup.

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 1;
        gbc.weighty = 1;
        addCompItem(sp, gbc, 0, 0, 1, 1);

        dataSorter = new DataSorter(ppTrial);

        //Sort the data for the main window.
        sortLocalData();

        //Call a repaint of the panel
        panel.repaint();

    }

    public void actionPerformed(ActionEvent evt) {

        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();

                if (arg.equals("Print")) {
                    ParaProfUtils.print(panel);

                } else if (arg.equals("Show ParaProf Manager")) {
                    (new ParaProfManagerWindow()).show();
                } else if (arg.equals("Preferences...")) {
                    trial.getPreferences().showPreferencesWindow();
                }
                //		
                //		else if(arg.equals("Save to XML File")){
                //		    //Ask the user for a filename and location.
                //		    JFileChooser fileChooser = new JFileChooser();
                //		    fileChooser.setDialogTitle("Save XML File");
                //		    //Set the directory.
                //		    fileChooser.setCurrentDirectory(new
                // File(System.getProperty("user.dir")));
                //		    fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                //		    int resultValue = fileChooser.showSaveDialog(this);
                //		    if(resultValue == JFileChooser.APPROVE_OPTION){
                //			System.out.println("Saving XML file ...");
                //			//Get both the file.
                //			File file = fileChooser.getSelectedFile();
                //			XMLSupport xMLSupport = new XMLSupport(trial);
                //			xMLSupport.writeXmlFiles(trial.getSelectedMetricID(),file);
                //			System.out.println("Done saving XML file ...");
                //		    }
                //		}
                //		else if(arg.equals("Save to txt File")){
                //		    //Ask the user for a filename and location.
                //		    JFileChooser fileChooser = new JFileChooser();
                //		    fileChooser.setDialogTitle("Save txt File");
                //		    //Set the directory.
                //		    fileChooser.setCurrentDirectory(new
                // File(System.getProperty("user.dir")));
                //		    fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                //		    int resultValue = fileChooser.showSaveDialog(this);
                //		    if(resultValue == JFileChooser.APPROVE_OPTION){
                //			System.out.println("Saving txt file ...");
                //			//Get both the file.
                //			File file = fileChooser.getSelectedFile();
                //			UtilFncs.outputData(trial.getDataSource(),file,this);
                //			System.out.println("Done saving txt file ...");
                //		    }
                //		}
                else if (arg.equals("Save Image")) {
                    ParaProfImageOutput imageOutput = new ParaProfImageOutput();
                    imageOutput.saveImage((ParaProfImageInterface) panel);
                } else if (arg.equals("Close This Window")) {
                    closeThisWindow();
                } else if (arg.equals("Exit ParaProf!")) {
                    setVisible(false);
                    dispose();
                    ParaProf.exitParaProf(0);
                } else if (arg.equals("Sort By Name")) {
                    if (nameCheckBox.isSelected())
                        name = true;
                    else
                        name = false;
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Normalize Bars")) {
                    if (normalizeCheckBox.isSelected())
                        normalizeBars = true;
                    else
                        normalizeBars = false;
                    panel.repaint();

                } else if (arg.equals("Stack Bars Together")) {
                    if (stackBarsCheckBox.isSelected()) {

                        normalizeCheckBox.setEnabled(true);
                        orderByMeanCheckBox.setEnabled(true);

                        stackBars = true;
                    } else {
                        stackBars = false;

                        normalizeCheckBox.setSelected(false);
                        normalizeCheckBox.setEnabled(false);
                        normalizeBars = false;
                        orderByMeanCheckBox.setSelected(true);
                        orderByMeanCheckBox.setEnabled(false);
                        orderByMean = true;
                    }
                    panel.repaint();
                }

                else if (arg.equals("Order By Mean")) {
                    if (orderByMeanCheckBox.isSelected())
                        orderByMean = true;
                    else
                        orderByMean = false;
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Descending Order")) {
                    if (orderCheckBox.isSelected())
                        order = 0;
                    else
                        order = 1;
                    sortLocalData();
                    panel.repaint();
                } else if (arg.equals("Display Sliders")) {
                    if (slidersCheckBox.isSelected())
                        displaySliders(true);
                    else
                        displaySliders(false);
                } else if (arg.equals("Show Path Title in Reverse"))
                    this.setTitle("ParaProf: " + trial.getTrialIdentifier(pathTitleCheckBox.isSelected()));
                else if (arg.equals("Show Meta Data in Panel"))
                    this.setHeader();
                else if (arg.equals("Show Function Ledger")) {
                    (new LedgerWindow(trial, 0)).show();
                } else if (arg.equals("Show Group Ledger")) {
                    (new LedgerWindow(trial, 1)).show();
                } else if (arg.equals("Show User Event Ledger")) {
                    (new LedgerWindow(trial, 2)).show();
                } else if (arg.equals("Show Call Path Relations")) {
                    CallPathTextWindow tmpRef = new CallPathTextWindow(trial, -1, -1, -1, this.getDataSorter(),
                            2);
                    trial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();
                } else if (arg.equals("Show Full Call Graph")) {
                    CallGraphWindow tmpRef = new CallGraphWindow(trial, -1, -1, -1, this.getDataSorter());
                    trial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();
                } else if (arg.equals("Close All Sub-Windows")) {
                    //Close the all subwindows.
                    trial.getSystemEvents().updateRegisteredObjects("subWindowCloseEvent");
                } else if (arg.equals("About ParaProf")) {
                    JOptionPane.showMessageDialog(this, ParaProf.getInfoString());
                } else if (arg.equals("Show Help Window")) {
                    ParaProf.helpWindow.show();
                }
            } else if (EventSrc == sliderMultiple) {
                panel.changeInMultiples();
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void stateChanged(ChangeEvent event) {
        try {
            panel.changeInMultiples();
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuSelected(MenuEvent evt) {
        try {
            if (trial.groupNamesPresent())
                ((JMenuItem) windowsMenu.getItem(2)).setEnabled(true);
            else
                ((JMenuItem) windowsMenu.getItem(2)).setEnabled(false);

            if (trial.userEventsPresent())
                ((JMenuItem) windowsMenu.getItem(3)).setEnabled(true);
            else
                ((JMenuItem) windowsMenu.getItem(3)).setEnabled(false);
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void menuDeselected(MenuEvent evt) {
    }

    public void menuCanceled(MenuEvent evt) {
    }

    public void update(Observable o, Object arg) {
        String tmpString = (String) arg;
        if (tmpString.equals("prefEvent")) {
            this.setHeader();
            panel.repaint();
        } else if (tmpString.equals("colorEvent")) {
            panel.repaint();
        } else if (tmpString.equals("dataEvent")) {
            sortLocalData();
            this.setHeader();
            panel.repaint();
        }
    }

    public Dimension getViewportSize() {
        return sp.getViewport().getExtentSize();
    }

    public Rectangle getViewRect() {
        return sp.getViewport().getViewRect();
    }

    public void setVerticalScrollBarPosition(int position) {
        JScrollBar scrollBar = sp.getVerticalScrollBar();
        scrollBar.setValue(position);
    }

    //######
    //Panel header.
    //######
    //This process is separated into two functionProfiles to provide the option
    //of obtaining the current header string being used for the panel
    //without resetting the actual header. Printing and image generation
    //use this functionality for example.
    public void setHeader() {
        if (metaDataCheckBox.isSelected()) {
            JTextArea jTextArea = new JTextArea();
            jTextArea.setLineWrap(true);
            jTextArea.setWrapStyleWord(true);
            jTextArea.setEditable(false);
            Preferences p = trial.getPreferences();
            jTextArea.setFont(new Font(p.getParaProfFont(), p.getFontStyle(), p.getFontSize()));
            jTextArea.append(this.getHeaderString());
            sp.setColumnHeaderView(jTextArea);
        } else
            sp.setColumnHeaderView(null);
    }

    public String getHeaderString() {
        return "Metric Name: " + (trial.getMetricName(trial.getSelectedMetricID())) + "\n" + "Value Type: "
                + UtilFncs.getValueTypeString(2) + "\n";
    }


    public double getSliderValue() {
        return (double) barLengthSlider.getValue();
    }

    public double getSliderMultiple() {
        String tmpString = null;
        tmpString = (String) sliderMultiple.getSelectedItem();
        return Double.parseDouble(tmpString);

    }

    private void displaySliders(boolean displaySliders) {
        if (displaySliders) {
            contentPane.remove(sp);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            addCompItem(sliderMultipleLabel, gbc, 0, 0, 1, 1);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            addCompItem(sliderMultiple, gbc, 1, 0, 1, 1);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            addCompItem(barLengthLabel, gbc, 2, 0, 1, 1);

            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0.70;
            gbc.weighty = 0.01;
            addCompItem(barLengthSlider, gbc, 3, 0, 1, 1);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1.0;
            gbc.weighty = 0.99;
            addCompItem(sp, gbc, 0, 1, 4, 1);
        } else {
            contentPane.remove(sliderMultipleLabel);
            contentPane.remove(sliderMultiple);
            contentPane.remove(barLengthLabel);
            contentPane.remove(barLengthSlider);
            contentPane.remove(sp);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1;
            gbc.weighty = 1;
            addCompItem(sp, gbc, 0, 0, 1, 1);
        }

        //Now call validate so that these component changes are displayed.
        validate();
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        contentPane.add(c, gbc);
    }

    public DataSorter getDataSorter() {
        return dataSorter;
    }

    //Updates the sorted lists after a change of sorting method takes place.
    private void sortLocalData() {

        if (name) {
            list = dataSorter.getAllFunctionProfiles(0 + order);
        } else {

            if (orderByMean) {
                list = dataSorter.getAllFunctionProfiles(20 + order);

            } else {
                list = dataSorter.getAllFunctionProfiles(2 + order);
            }
        }

    }

    public Vector getData() {
        return list;
    }

    private boolean mShown = false;

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

    //Close the window when the close box is clicked
    void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    void closeThisWindow() {
        try {
            setVisible(false);
            
            // don't do this!
            //trial.getSystemEvents().deleteObserver(this);
            ParaProf.decrementNumWindows();
        } catch (Exception e) {
            // do nothing
        }
        dispose();
    }

    public boolean getNormalizeBars() {
        return this.normalizeBars;
    }

    public boolean getStackBars() {
        return this.stackBars;
    }

    //Instance data.
    ParaProfTrial trial = null;

    //Create a file chooser to allow the user to select files for loading data.
    JFileChooser fileChooser = new JFileChooser();

    //References for some of the components for this frame.
    private StaticMainWindowPanel panel = null;
    private DataSorter dataSorter = null;

    private JMenu optionsMenu = null;
    private JMenu windowsMenu = null;
    private JCheckBoxMenuItem nameCheckBox = null;
    private JCheckBoxMenuItem normalizeCheckBox = null;
    private JCheckBoxMenuItem stackBarsCheckBox = null;
    private JCheckBoxMenuItem orderByMeanCheckBox = null;
    private JCheckBoxMenuItem orderCheckBox = null;
    private JCheckBoxMenuItem slidersCheckBox = null;
    private JCheckBoxMenuItem pathTitleCheckBox = null;
    private JCheckBoxMenuItem metaDataCheckBox = null;

    private JLabel sliderMultipleLabel = new JLabel("Slider Multiple");
    private JComboBox sliderMultiple;
    private JLabel barLengthLabel = new JLabel("Bar Multiple");
    private JSlider barLengthSlider = new JSlider(0, 40, 1);

    private Container contentPane = null;
    private GridBagLayout gbl = null;
    private GridBagConstraints gbc = null;
    private JScrollPane sp;

    private boolean name = false; //true: sort by name,false: sort by value.
    private int order = 0; //0: descending order,1: ascending order.
    private boolean normalizeBars = true;
    private boolean orderByMean = true;
    private boolean stackBars = true;

    boolean displaySliders = false;

    //    private Vector[] list = new Vector[2]; //list[0]:The result of a call to
    // getSMWGeneralData in
    // DataSorter
    //list[1]:The result of a call to getMeanData in DataSorter

    private Vector list;

}