package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.ListCellRenderer;
import javax.swing.ListSelectionModel;
import javax.swing.colorchooser.ColorSelectionModel;

/**
 * @author Robert Bell, Alan Morris
 *
 * TODO ...
 */
public class ColorDefaultsWindow extends JFrame implements ActionListener, MouseListener {

    /**
	 * 
	 */
	private static final long serialVersionUID = -552939062619209145L;
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
   
    public ColorDefaultsWindow(ColorChooser colorChooser, Component invoker) {
        this.colorChooser = colorChooser;

        //Window Stuff.
        setSize(new Dimension(855, 450));
        setLocation(WindowPlacer.getNewLocation(this, invoker));
        setTitle("TAU: ParaProf: Edit Default Colors");
        ParaProfUtils.setFrameIcon(this);

        setupMenus();
        

        //Setting up the layout system for the main window.
        Container contentPane = getContentPane();
        contentPane.setLayout(new GridBagLayout());
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
        JLabel titleLabel = new JLabel("Default Color Set");
        titleLabel.setFont(new Font("SansSerif", Font.PLAIN, 14));
        addCompItem(titleLabel, gbc, 0, 0, 1, 1);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 0.1;
        gbc.weighty = 0.1;

        //Create and add color list.
        listModel = new DefaultListModel();
        colorList = new JList(listModel);
        colorList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        colorList.setCellRenderer(new CustomCellRenderer());
        colorList.setSize(500, 300);
        colorList.addMouseListener(this);
        JScrollPane sp = new JScrollPane(colorList);
        addCompItem(sp, gbc, 0, 1, 1, 5);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;
        addColorButton = new JButton("Add Function Color");
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

        populateColorList();
    }
    
    
    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();

        //File menu.
        JMenu fileMenu = new JMenu("File");

        JMenuItem closeItem = new JMenuItem("Close This Window");
        closeItem.addActionListener(this);
        fileMenu.add(closeItem);

        JMenuItem exitItem = new JMenuItem("Exit ParaProf!");
        exitItem.addActionListener(this);
        fileMenu.add(exitItem);

        //Now, add all the menus to the main menu.
        mainMenu.add(fileMenu);
        //mainMenu.add(helpMenu);
        setJMenuBar(mainMenu);
    }
    

    private void updateTrials() {
        Vector<ParaProfTrial> trials = ParaProf.paraProfManagerWindow.getLoadedTrials();
        
        for (Iterator<ParaProfTrial> it = trials.iterator(); it.hasNext();) {
            ParaProfTrial ppTrial = it.next();
            
            colorChooser.setColors(ppTrial, -1);
            //Update the listeners.
            ppTrial.updateRegisteredObjects("colorEvent");
        }
   
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
                if (arg.equals("Add Function Color")) {
                    Color color = clrModel.getSelectedColor();
                    (colorChooser.getColors()).add(color);
                    listModel.clear();
                    populateColorList();
                    updateTrials();
                } else if (arg.equals("Add Group Color")) {
                    Color color = clrModel.getSelectedColor();
                    (colorChooser.getGroupColors()).add(color);
                    listModel.clear();
                    populateColorList();
                    updateTrials();
                } else if (arg.equals("Delete Selected Color")) {
                    //Get the currently selected items and cycle through them.
                    int[] values = colorList.getSelectedIndices();
                    for (int i = 0; i < values.length; i++) {
                        if ((values[i]) < ParaProf.colorChooser.getNumberOfColors()) {
                            if (ParaProf.colorChooser.getNumberOfColors() > 2) {
                                listModel.removeElementAt(values[i]);
                                (colorChooser.getColors()).removeElementAt(values[i]);
                                updateTrials();
                            }
                        } else if ((values[i]) < (ParaProf.colorChooser.getNumberOfColors())
                                + (ParaProf.colorChooser.getNumberOfGroupColors())) {
                            if (ParaProf.colorChooser.getNumberOfGroupColors() > 2) {
                                listModel.removeElementAt(values[i]);
                                (colorChooser.getGroupColors()).removeElementAt(values[i]
                                        - (ParaProf.colorChooser.getNumberOfColors()));
                                
                                updateTrials();
                            }
                        }
                    }
                    updateTrials();

                } else if (arg.equals("Update Selected Color")) {
                    Color color = clrModel.getSelectedColor();
                
                    //Get the currently selected items and cycle through them.
                    int[] values = colorList.getSelectedIndices();
                    for (int i = 0; i < values.length; i++) {
                        listModel.setElementAt(color, values[i]);
                        int totalNumberOfColors = (colorChooser.getNumberOfColors())
                                + (colorChooser.getNumberOfGroupColors());
                        if ((values[i]) == (totalNumberOfColors)) {
                            colorChooser.setHighlightColor(color);
                        } else if ((values[i]) == (totalNumberOfColors + 1)) {
                            colorChooser.setGroupHighlightColor(color);
                        } else if ((values[i]) == (totalNumberOfColors + 2)) {
                            colorChooser.setUserEventHighlightColor(color);
                        } else if ((values[i]) == (totalNumberOfColors + 3)) {
                            colorChooser.setMiscFunctionColor(color);
                        } else if ((values[i]) < colorChooser.getNumberOfColors()) {
                            colorChooser.setColor(color, values[i]);
                            updateTrials();
                        } else {
                            colorChooser.setGroupColor(color,
                                    (values[i] - colorChooser.getNumberOfColors()));
                            updateTrials();
                        }
                    }
                    updateTrials();
                } else if (arg.equals("Restore Defaults")) {
                    colorChooser.setDefaultColors();
                    colorChooser.setDefaultGroupColors();
                    colorChooser.setHighlightColor(Color.red);
                    colorChooser.setGroupHighlightColor(new Color(0, 255, 255));
                    colorChooser.setUserEventHighlightColor(new Color(255, 255, 0));
                    colorChooser.setMiscFunctionColor(Color.black);
                    listModel.clear();
                    populateColorList();
                    updateTrials();
                }
            }

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }

    }

    
    public void mouseClicked(MouseEvent evt) {
        try {
            JList jList = (JList) evt.getSource();
        
            int index = jList.locationToIndex(evt.getPoint());

            Color color = (Color) listModel.getElementAt(index);
            
            clrModel.setSelectedColor(color);
            
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void mousePressed(MouseEvent evt) {
    }

    public void mouseReleased(MouseEvent evt) {
    }

    public void mouseEntered(MouseEvent evt) {
    }

    public void mouseExited(MouseEvent evt) {
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
        for (Enumeration<Color> e = (colorChooser.getColors()).elements(); e.hasMoreElements();) {
            color = e.nextElement();
            listModel.addElement(color);
        }

        for (Enumeration<Color> e = (colorChooser.getGroupColors()).elements(); e.hasMoreElements();) {
            color = e.nextElement();
            listModel.addElement(color);
        }

        color = colorChooser.getHighlightColor();
        listModel.addElement(color);

        color = colorChooser.getGroupHighlightColor();
        listModel.addElement(color);

        color = colorChooser.getUserEventHighlightColor();
        listModel.addElement(color);

        color = colorChooser.getMiscFunctionColor();
        listModel.addElement(color);
    }

  
}

class CustomCellRenderer implements ListCellRenderer {

    public Component getListCellRendererComponent(final JList list, final Object value, final int index,
            final boolean isSelected, final boolean cellHasFocus) {
        return new JPanel() {
            /**
			 * 
			 */
			private static final long serialVersionUID = -1614728767463124603L;

			public void paintComponent(Graphics g) {
                super.paintComponent(g);
                Color inColor = (Color) value;

                int xSize = 0;
                int ySize = 0;
                int maxXNumFontSize = 0;
                //int maxXFontSize = 0;
                //int maxYFontSize = 0;
                //int thisXFontSize = 0;
                //int thisYFontSize = 0;
                int barHeight = 0;

                //For this, I will not allow changes in font size.
                barHeight = 12;

                //Create font.
                Font font = new Font("SansSerif", Font.PLAIN, barHeight);
                g.setFont(font);
                FontMetrics fmFont = g.getFontMetrics(font);

                //maxYFontSize = fmFont.getAscent();
                //maxXFontSize = fmFont.stringWidth("0000,0000,0000");

                xSize = getWidth();
                ySize = getHeight();

                String tmpString1 = new String("00" + (ParaProf.colorChooser.getNumberOfColors()));
                maxXNumFontSize = fmFont.stringWidth(tmpString1);

                //String tmpString2 = new String(inColor.getRed() + "," + inColor.getGreen() + "," + inColor.getBlue());
                //thisXFontSize = fmFont.stringWidth(tmpString2);
                //thisYFontSize = maxYFontSize;

                g.setColor(isSelected ? list.getSelectionBackground() : list.getBackground());
                g.fillRect(0, 0, xSize, ySize);

                g.setColor(inColor);
                g.fillRect(5, 1, 50, ySize - 1);

                //Just a sanity check.
                if ((xSize - 50) > 0) {
                    g.setColor(isSelected ? list.getSelectionBackground() : list.getBackground());
                    g.fillRect((5 + maxXNumFontSize + 5 + 50), 0, (xSize - 50), ySize);
                }

                int xStringPos1 = 60;
                int yStringPos1 = (ySize - 5);

                //int xStringPos1 = 5;
                //int yStringPos1 = (ySize - 5);
                g.setColor(isSelected ? list.getSelectionForeground() : list.getForeground());

                int totalNumberOfColors = (ParaProf.colorChooser.getNumberOfColors())
                        + (ParaProf.colorChooser.getNumberOfGroupColors());

                String id = null;

                if (index == totalNumberOfColors) {
                    id = "Func. Highlight";
                } else if (index == (totalNumberOfColors + 1)) {
                    id = "Group Highlight";
                } else if (index == (totalNumberOfColors + 2)) {
                    id = "User Event Highlight";
                } else if (index == (totalNumberOfColors + 3)) {
                    id = "Misc. Func. Color";
                } else if (index < (ParaProf.colorChooser.getNumberOfColors())) {
                    id = "Function " + (index + 1);
                } else {
                    id = "Group " + (index - (ParaProf.colorChooser.getNumberOfColors()) + 1);
                }

                g.drawString(id, xStringPos1, yStringPos1);

            }

            public Dimension getPreferredSize() {
                int xSize = 0;
                int ySize = 0;
                int maxXNumFontSize = 0;
                int maxXFontSize = 0;
                int maxYFontSize = 0;
                int barHeight = 12;

                //Create font.
                Font font = new Font(ParaProf.preferencesWindow.getFontName(), Font.PLAIN, barHeight);
                Graphics g = getGraphics();
                FontMetrics fmFont = g.getFontMetrics(font);

                String tmpString = new String("00" + (ParaProf.colorChooser.getNumberOfColors()));
                maxXNumFontSize = fmFont.stringWidth(tmpString);

                maxXFontSize = fmFont.stringWidth("0000,0000,0000");
                maxYFontSize = fmFont.getAscent();

                xSize = (maxXNumFontSize + 10 + 50 + maxXFontSize + 20);
                ySize = (10 + maxYFontSize);

                return new Dimension(xSize, ySize);
            }
        };
    }

}