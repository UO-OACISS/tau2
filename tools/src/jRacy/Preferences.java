/*	
	Preferences.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import javax.swing.colorchooser.*;

public class Preferences extends JFrame implements ActionListener, Observer
{	
	//******************************
	//Instance data.
	//******************************
		
	//References for some of the components for this frame.
	private PrefColorPanel pCPanel;
	private PrefSpacingPanel pSPanel;
	
	private JCheckBox loadPprofDat;
	
	
	private JRadioButton normal;
	private JRadioButton bold;
	private JRadioButton italic;
	
	private ButtonGroup buttonGroup;
	
	private JLabel fontLabel = new JLabel("Font Selection");
	
	private JComboBox fontComboBox;
	
	private JLabel barSpacingLabel = new JLabel("Adjust Bar Spacing");
	private JLabel barHeightLabel = new JLabel("Adjust Bar Height");
	
	private JSlider barSpacingSlider = new JSlider(SwingConstants.VERTICAL, 0, 100, 0);
	private JSlider barHeightSlider = new JSlider(SwingConstants.VERTICAL, 0, 100, 0);
	
	
	private JButton colorButton;
	private JButton closeButton;
	
	int fontStyle;
	
	private boolean barDetailsSet = false;
	private int barSpacing = -1;
	private int barHeight = -1;
	
	String jRacyFont;
	
	//Whether we are doing inclusive or exclusive.
	String inExValue;
	
	//Variable to determine which sorting paradigm has been chosen.
	String sortBy;	//Possible values are:
					//functionID
					//millDes
					//millAsc
	
	//******************************
	//End - Instance data.
	//******************************
	public Preferences(SavedPreferences inSavedPreferences)
	{	
		
		
		if(inSavedPreferences != null)
		{	
			//******************************
			//Set the saved values.
			//******************************
			jRacyFont = inSavedPreferences.getJRacyFont();
			barSpacing = inSavedPreferences.getBarSpacing();
			barHeight = inSavedPreferences.getBarHeight();
			inExValue = inSavedPreferences.getInExValue();
			sortBy = inSavedPreferences.getSortBy();
			
			fontStyle = inSavedPreferences.getFontStyle();
			
			barDetailsSet = inSavedPreferences.getBarDetailsSet();
			//******************************
			//End - Set the saved values.
			//******************************
		}
		else
		{
		
			//******************************
			//Set the default values.
			//******************************	
			//Set inExValue ... exclusive by default.
			inExValue = new String("Exclusive");
			//Set sortBy ... functionID by default.
			String sortBy = new String("functionID");
			
			
			jRacyFont = "SansSerif";
			barHeight = 0;
			barSpacing =  0;
			
			fontStyle = Font.PLAIN;
			//******************************
			//End - Set the default values.
			//******************************
		}
		
		//Get available fonts and initialize the fontComboBox..
		GraphicsEnvironment gE = GraphicsEnvironment.getLocalGraphicsEnvironment();
		String[] fontFamilyNames = gE.getAvailableFontFamilyNames();
		fontComboBox = new JComboBox(fontFamilyNames);
		
		int tmpInt = fontComboBox.getItemCount();
		int counter = 0;
		//We should always have some fonts available, so this should be safe.
		String tmpString = (String) fontComboBox.getItemAt(counter);
		while((counter < tmpInt) && (!(jRacyFont.equals(tmpString))))
		{
			counter++;
			tmpString = (String) fontComboBox.getItemAt(counter);
		}
		
		if(counter == tmpInt)
		{
			//The default font was not available.  Indicate an error.
			System.out.println("The default font was not found!!  This is not a good thing as it is a default Java font!!");
		}
		else
		{
			fontComboBox.setSelectedIndex(counter);
		}
		
		//Set the sliders.
		barHeightSlider.setValue(barHeight);
		barSpacingSlider.setValue(barSpacing);
		
		fontComboBox.addActionListener(this);
		
		//Now initialize the panels.
		pCPanel = new PrefColorPanel();
		pSPanel = new PrefSpacingPanel();
		
		//Window Stuff.
		setTitle("jRacy Preferences: " + jRacy.profilePathName);
		
		int windowWidth = 900;
		int windowHeight = 450;
		setSize(new java.awt.Dimension(windowWidth, windowHeight));
		
		//There is really no need to resize this window.
		setResizable(false);
		
		
		//Grab the screen size.
		Toolkit tk = Toolkit.getDefaultToolkit();
		Dimension screenDimension = tk.getScreenSize();
		int screenHeight = screenDimension.height;
		int screenWidth = screenDimension.width;
		
		//Set the window to come up in the center of the screen.
		int xPosition = 0;
		int yPosition = 0;
		
		setLocation(xPosition, yPosition);
		
		//End - Window Stuff.

		//******************************
		//Code to generate the menus.
		//******************************
		
		
		JMenuBar mainMenu = new JMenuBar();
		
		//******************************
		//File menu.
		//******************************
		JMenu fileMenu = new JMenu("File");
		
		//Add a menu item.
		JMenuItem closeItem = new JMenuItem("Close This Window");
		closeItem.addActionListener(this);
		fileMenu.add(closeItem);
		
		//Add a menu item.
		JMenuItem exitItem = new JMenuItem("Exit jRacy!");
		exitItem.addActionListener(this);
		fileMenu.add(exitItem);
		//******************************
		//End - File menu.
		//******************************
		
		//******************************
		//Help menu.
		//******************************
		/*JMenu helpMenu = new JMenu("Help");
		
		//Add a menu item.
		JMenuItem aboutItem = new JMenuItem("About Racy");
		helpMenu.add(aboutItem);
		
		//Add a menu item.
		JMenuItem showHelpWindowItem = new JMenuItem("Show Help Window");
		showHelpWindowItem.addActionListener(this);
		helpMenu.add(showHelpWindowItem);*/
		//******************************
		//End - Help menu.
		//******************************
		
		
		//Now, add all the menus to the main menu.
		mainMenu.add(fileMenu);
		//mainMenu.add(helpMenu);
		
		setJMenuBar(mainMenu);
		
		//******************************
		//End - Code to generate the menus.
		//******************************
		
		
		//******************************
		//Create and add the componants.
		//******************************
		//Setting up the layout system for the main window.
		Container contentPane = getContentPane();
		GridBagLayout gbl = new GridBagLayout();
		contentPane.setLayout(gbl);
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.insets = new Insets(5, 5, 5, 5);
		
		//Create some borders.
		Border mainloweredbev = BorderFactory.createLoweredBevelBorder();
		Border mainraisedbev = BorderFactory.createRaisedBevelBorder();
		Border mainempty = BorderFactory.createEmptyBorder();
		
		//**********
		//Panel and ScrollPane definition.
		//**********
		JScrollPane scrollPaneC = new JScrollPane(pCPanel);
		scrollPaneC.setBorder(mainloweredbev);
		scrollPaneC.setPreferredSize(new Dimension(500, 100));
		
		JScrollPane scrollPaneS = new JScrollPane(pSPanel);
		scrollPaneS.setBorder(mainloweredbev);
		scrollPaneS.setPreferredSize(new Dimension(500, 200));
		//**********
		//End - Panel and ScrollPane definition.
		//**********
		
		//**********
		//Slider Setup
		//**********
		barSpacingSlider.setPaintTicks(true);
		barSpacingSlider.setMajorTickSpacing(20);
		barSpacingSlider.setMinorTickSpacing(5);
		barSpacingSlider.setPaintLabels(true);
		barSpacingSlider.addChangeListener(pSPanel);
		barSpacingSlider.addChangeListener(pCPanel);
		
		barHeightSlider.setPaintTicks(true);
		barHeightSlider.setMajorTickSpacing(20);
		barHeightSlider.setMinorTickSpacing(5);
		barHeightSlider.setPaintLabels(true);
		barHeightSlider.addChangeListener(pSPanel);
		barHeightSlider.addChangeListener(pCPanel);
		//**********
		//End - Slider Setup
		//**********
		
		//**********
		//Button Setup
		//**********
		colorButton = new JButton("Adjust Colours");
		colorButton.addActionListener(this);
		closeButton = new JButton("Close");
		closeButton.addActionListener(this);
		//**********
		//End - Button Setup
		//**********
		
		
		//**********
		//CheckBox Setup
		//**********
		//loadPprofDat = new JCheckBox("Load Pprof.dat File on Startup");
		//loadPprofDat.addActionListener(this);
		//**********
		//End - CheckBox Setup
		//**********
		
		//**********
		//RadioButton and ButtonGroup Setup
		//**********
		normal = new JRadioButton("Plain Font", ((fontStyle == Font.PLAIN) || (fontStyle == (Font.PLAIN|Font.ITALIC))));
		normal.addActionListener(this);
		bold = new JRadioButton("Bold Font", ((fontStyle == Font.BOLD) || (fontStyle == (Font.BOLD|Font.ITALIC))));
		bold.addActionListener(this);
		italic = new JRadioButton("Italic Font", ((fontStyle == (Font.PLAIN|Font.ITALIC)) || (fontStyle == (Font.BOLD|Font.ITALIC))));
		italic.addActionListener(this);
		
		buttonGroup = new ButtonGroup();
		buttonGroup.add(normal);
		buttonGroup.add(bold);
		//**********
		//End - RadioButton and ButtonGroup Setup
		//**********
		
		//gbc.fill = GridBagConstraints.BOTH;
		//gbc.anchor = GridBagConstraints.CENTER;
		//gbc.weightx = 1;
		//gbc.weighty = 1;
		//addCompItem(loadPprofDat, gbc, 0, 0, 1, 1);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.EAST;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(fontLabel, gbc, 2, 0, 1, 1);
		
		gbc.fill = GridBagConstraints.BOTH;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(fontComboBox, gbc, 3, 0, 1, 1);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(colorButton, gbc, 0, 1, 1, 1);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(normal, gbc, 1, 1, 1, 1);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(bold, gbc, 2, 1, 1, 1);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(italic, gbc, 3, 1, 1, 1);
		
		gbc.fill = GridBagConstraints.BOTH;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(scrollPaneS, gbc, 0, 3, 2, 2);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.SOUTH;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(barSpacingLabel, gbc, 2, 3, 1, 1);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(barSpacingSlider, gbc, 2, 4, 1, 1);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.SOUTH;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(barHeightLabel, gbc, 3, 3, 1, 1);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(barHeightSlider, gbc, 3, 4, 1, 1);
		
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 1;
		gbc.weighty = 1;
		addCompItem(closeButton, gbc, 3, 5, 1, 1);
		
		//******************************
		//End - Create and add the componants.
		//******************************
		
		pCPanel.repaint();
		//pSPanel.repaint();
			
	}
	
	public void showPreferencesWindow()
	{
		this.show();
	}
	
	public void setSavedPreferences()
	{
		jRacy.savedPreferences.setJRacyFont(jRacyFont);
		jRacy.savedPreferences.setBarSpacing(barSpacing);
		jRacy.savedPreferences.setBarHeight(barHeight);
		jRacy.savedPreferences.setInExValue(inExValue);
		jRacy.savedPreferences.setSortBy(sortBy);
		jRacy.savedPreferences.setFontStyle(fontStyle);
		jRacy.savedPreferences.setBarDetailsSet(barDetailsSet);
		
	}
		
	
	public boolean areBarDetailsSet()
	{
		return barDetailsSet;
	}
	
	public String getJRacyFont()
	{
		return jRacyFont;
	}
	
	public int getFontStyle()
	{
		return fontStyle;
	}
	
	public void setBarDetails(int inBarHeight, int inBarSpacing)
	{
		barHeight = inBarHeight;
		barSpacing = inBarSpacing;
		
		barDetailsSet = true;
	}
	
	public void setSliders(int inBarHeight, int inBarSpacing)
	{
		//Set the slider values.
		barHeightSlider.setValue(inBarHeight);
		barSpacingSlider.setValue(inBarSpacing);
	}
	
	public int getBarSpacing()
	{
		return barSpacing;
	}
	
	public int getBarHeight()
	{
		return barHeight;
	}
	
	//Setting and returning the inExValue.
	public void setInExValue(String inString)
	{
		inExValue = inString;
	}
	
	public String getInExValue()
	{
		return inExValue;
	}
	
	//Setting and returning sortBy.
	public void setSortBy(String inString)
	{
		sortBy = inString;
	}
	
	public String getSortBy()
	{
		return sortBy;
	}
	
	//******************************
	//Event listener code!!
	//******************************
	
	//Observer functions.
	public void update(Observable o, Object arg)
	{
		String tmpString = (String) arg;
		
		if(tmpString.equals("colorEvent"))
		{			
			//Just need to call a repaint.
			pCPanel.repaint();
			pSPanel.repaint();
		}
	}
	
	//ActionListener code.
	public void actionPerformed(ActionEvent evt)
	{
		Object EventSrc = evt.getSource();
		String arg = evt.getActionCommand();
		
		if(EventSrc instanceof JMenuItem)
		{
			if(arg.equals("Exit jRacy!"))
			{
				setVisible(false);
				dispose();
				System.exit(0);
			}
			else if(arg.equals("Close This Window"))
			{
				setVisible(false);
			}
		}
		else if(EventSrc instanceof JButton)
		{
			if(arg.equals("Close"))
			{
				setVisible(false);
				jRacy.systemEvents.updateRegisteredObjects("prefEvent");
			}
			if(arg.equals("Adjust Colours"))
			{
				jRacy.clrChooser.showColorChooser();
			}
		}
		else if(EventSrc instanceof JRadioButton)
		{
			if(arg.equals("Plain Font"))
			{
				if(italic.isSelected())
					fontStyle = Font.PLAIN|Font.ITALIC;
				else
					fontStyle = Font.PLAIN;
				
				pSPanel.repaint();
			}
			else if(arg.equals("Bold Font"))
			{
				if(italic.isSelected())
					fontStyle = Font.BOLD|Font.ITALIC;
				else
					fontStyle = Font.BOLD;
				
				pSPanel.repaint();
			}
			else if(arg.equals("Italic Font"))
			{
				if(italic.isSelected())
				{
					if(normal.isSelected())
						fontStyle = Font.PLAIN|Font.ITALIC;
					else
						fontStyle = Font.BOLD|Font.ITALIC;
				}
				else
				{
					if(normal.isSelected())
						fontStyle = Font.PLAIN;
					else
						fontStyle = Font.BOLD;
				}
				
				pSPanel.repaint();
			}
		}
		else if(EventSrc == fontComboBox)
		{
			jRacyFont = (String) fontComboBox.getSelectedItem();
			pSPanel.repaint();
		}
	}
	
	public void updateBarDetails()
	{
		barHeight = barHeightSlider.getValue();
		barSpacing = barSpacingSlider.getValue();
	}
		
	private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h)
	{
		gbc.gridx = x;
		gbc.gridy = y;
		gbc.gridwidth = w;
		gbc.gridheight = h;
		
		getContentPane().add(c, gbc);
	}
	
	//******************************
	//End - Event listener code!!
	//******************************
	
	private boolean mShown = false;
	
	public void addNotify() 
	{
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
	void thisWindowClosing(java.awt.event.WindowEvent e)
	{
		setVisible(false);
	}

}
