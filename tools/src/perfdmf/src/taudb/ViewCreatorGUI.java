package edu.uoregon.tau.perfdmf.taudb;

/*
 * Copyright (c) 1995, 2008, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *   - Neither the name of Oracle or the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */ 



/*
 * CardLayoutDemo.java
 *
 */
import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.database.DB;



public class ViewCreatorGUI extends JFrame implements ActionListener{
	/**
	 * 
	 */
	private static final long serialVersionUID = -5859741019248347898L;

	public class ViewCreatorListner implements ItemListener, ActionListener {
    	JPanel cards;

		public ViewCreatorListner(JPanel comparators) {
			super();
			cards = comparators;
		}


		public void itemStateChanged(ItemEvent evt) {
	        CardLayout cl = (CardLayout)(cards.getLayout());
	        cl.show(cards, (String)evt.getItem());
	    }


		public void actionPerformed(ActionEvent evt) {
			if(evt.getSource() instanceof JButton){
				JButton button = (JButton) evt.getSource();
				if(button.getText().equals("+")){
			    	createNewRule();
			    	panel.validate();
				}else if(button.getText().equals("-")){
					rulePane.remove(button.getParent());
					rulePane.getParent().validate();
				}
			}
			
		}

	}


	 static final String STRING_ENDS = "ends with";
	 static final String STRING_CONTAINS = "contains";
	 static final String STRING_EXACTLY = "is exactly";
	 static final String STRING_NOT = "is not";
	 static final String STRING_BEGINS = "beings with";
	
	 static final String NUMBER_EQUAL = "is equal to";
	 static final String NUMBER_NOT = "is not equal to";
	 static final String NUMBER_LESS = "is less than";
	 static final String NUMBER_RANGE = "is in the range";
	 static final String NUMBER_GREATER = "is greater than";
	 
	 static final String DATE_IS = "is";
	 static final String DATE_RANGE = "is between";
	 static final String DATE_BEFORE = "is before";
	 static final String DATE_AFTER = "is after";
	 
	 static final String STRING = "read as a string";
	 static final String NUMBER = "read as a number";
	 static final String DATE = "read as a date";
	 
	 static final String ANY="or";
	 static final String ALL="and";
	 static final String METADATA = "METADATA";
	 static final String READ_TYPE = "Read Type";
	 

	 
	private JPanel panel;
	private JPanel rulePane;
	private List<ViewCreatorRuleListener> ruleListeners;
	private String anyOrAll;
	private TAUdbDatabaseAPI databaseAPI;
	private DB db;
	private int parentID;

	
    
    public ViewCreatorGUI(TAUdbDatabaseAPI databaseAPI) {
    	this(databaseAPI, -1);
    }
    public ViewCreatorGUI(TAUdbDatabaseAPI databaseAPI, int parentID) {
    	super();
    	 try {
    		// UIManager.setLookAndFeel("javax.swing.plaf.metal.MetalLookAndFeel");
             UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
         } catch (Exception e) {}
    	this.databaseAPI = databaseAPI;
    	this.db = databaseAPI.getDb();
    	this.parentID = parentID;
    	this.ruleListeners = new ArrayList<ViewCreatorRuleListener>();
    	this.setTitle("TAUdb View Creator");
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            	
    	panel = new JPanel();
    	panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

    	rulePane = new JPanel();

    	rulePane.setLayout(new BoxLayout(rulePane, BoxLayout.Y_AXIS));


    	JScrollPane scrollRule = new JScrollPane(rulePane);
    	scrollRule.setPreferredSize(new Dimension(800, 200));

    	
    	createNewRule();
    	rulePane.validate();
        //Display the window.
        rulePane.setVisible(true);


    	panel.add(addMatch());
    	panel.add(scrollRule);
    	panel.add(getSaveButtons());
    	panel.validate();
    	
    	this.getContentPane().add(panel);
	}
	private JPanel getSaveButtons() {
		JPanel buttonPanel = new JPanel();
		buttonPanel.setLayout(new BoxLayout(buttonPanel, BoxLayout.X_AXIS));
		
		JButton save = new JButton("Save");
		save.setActionCommand("Save"); 
		save.addActionListener(this);
		
		JButton cancel = new JButton("Cancel");
		cancel.setActionCommand("Cancel");
		cancel.addActionListener(this);
		
		buttonPanel.add(cancel);
		buttonPanel.add(save);
		
    	return buttonPanel;
	}
	public void close(){
		this.dispose();
	}
	public void actionPerformed(ActionEvent e) {
		if("Save".equals(e.getActionCommand())){

			String valid = checkValues();

			if (valid != null) {
				JOptionPane.showMessageDialog(this, valid
						+ " is not a valid numeric value.", "Invalid Value",
						JOptionPane.ERROR_MESSAGE);
				return;
			}

			String saveName = (String)JOptionPane.showInputDialog(
			                    this,
			                    "Please enter the name of this TAUdb View",
			                    "Save TAUdb View",
			                    JOptionPane.PLAIN_MESSAGE);

			//If a string was returned, say so.
			if ((saveName != null) && (saveName.length() > 0)) {
				saveView(saveName);
				close();
			}
			
		}else if ("Cancel".equals(e.getActionCommand())){
			close();
		}else if ("comboBoxChanged".equals(e.getActionCommand())){
			anyOrAll = ((JComboBox<String>) e.getSource()).getSelectedItem()
					.toString();
		}
		
	}

	private void saveView(String saveName) {

		int viewID;
		try {
			viewID = View.saveView(db, saveName, anyOrAll, parentID);

			for (ViewCreatorRuleListener rule : ruleListeners) {

				if (rule.getOperator().equals(NUMBER_RANGE)) {
					View.saveViewParameter(db, viewID, rule.getTable_name(),
							rule.getColumn_name(), ">=", rule.getValue());
					View.saveViewParameter(db, viewID, rule.getTable_name(),
							rule.getColumn_name(), "<=", rule.getValue2());
				} else {
					View.saveViewParameter(db, viewID, rule.getTable_name(),
							rule.getColumn_name(), rule.getOperator(),
							rule.getValue());
				}
			}
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	private static boolean isNumber(String str) {
		try {
			Double.parseDouble(str);
		} catch (NumberFormatException nfe) {
			return false;
		}
		return true;
	}

	private String checkValues() {
		// Set<String> numberOps = new HashSet<String>(Arrays.asList(new
		// String[] {
		// NUMBER_EQUAL, NUMBER_NOT, NUMBER_GREATER, NUMBER_LESS,
		// NUMBER_RANGE }));
		for (ViewCreatorRuleListener rule : ruleListeners) {
			if (rule.type.equals(NUMBER)) {

				String value = rule.getValue();
				if (!isNumber(value)) {
					return value;
				}

				if (rule.getOperator().equals(NUMBER_RANGE)) {
					value = rule.getValue2();
					if (!isNumber(value)) {
						return value;
					}
				}
			}
		}

		return null;
	}

	private JPanel addMatch() {
		JPanel panel = new JPanel();
		String[] comboBoxItems = {ALL,  ANY};
		JComboBox<String> comboBox = new JComboBox<String>(comboBoxItems);
		JLabel label1 = new JLabel("Match ");
		JLabel label2 = new JLabel(" of the following rules.");
		
		comboBox.addActionListener(this);
		comboBox.setSelectedIndex(0);

		panel.add(label1);
		panel.add(comboBox);
		panel.add(label2);
		
		return panel;
	}
//	public enum ComparatorType {
//	    STRING("read as a string"), 
//	    NUMBER(""), 
//	    DATE("");
//	    private final String string;   // in kilograms
//	    ComparatorType(String string) {
//	        this.string = string;
//	    }
//	}

	
	public void createNewRule() {

		//String[] comparatorTypes = {STRING,NUMBER, DATE};
		String[] comparatorTypes = {STRING,NUMBER};

    	JPanel cards;
        JPanel comboBoxPane = new JPanel(); //use FlowLayout
        String comboBoxItems[] = comparatorTypes ;
		JComboBox<String> cb = new JComboBox<String>(comboBoxItems);
        cb.setEditable(false);
        cb.setName(READ_TYPE);
        
        
        cards = new JPanel(new CardLayout());
        ViewCreatorRuleListener listener = new ViewCreatorRuleListener();
        ruleListeners.add(listener);
        cb.addActionListener(listener);
       
        cards.add(addStringField(listener), STRING);
        cards.add(addNumberField(listener),NUMBER);
//        cards.add(addDateField(), DATE);
        
        
        ViewCreatorListner listner = new ViewCreatorListner(cards);
        cb.addItemListener(listner);

        
        JButton plusButton = new JButton("+");
        JButton minusButton = new JButton("-");
        plusButton.addActionListener(listner);
        minusButton.addActionListener(listner);
        
        String metadataList[] = getMetaDataList();
        JComboBox<String> metadataCB = new JComboBox<String>(metadataList);
        metadataCB.addActionListener(listener);
        metadataCB.setEditable(false);
        metadataCB.setName(METADATA);
        if(metadataList.length>0)
        metadataCB.setSelectedIndex(0);
        
        comboBoxPane.add(metadataCB, BorderLayout.WEST);
        comboBoxPane.add(cb, BorderLayout.CENTER);
        comboBoxPane.add(cards, BorderLayout.EAST);
        comboBoxPane.add(minusButton);
        comboBoxPane.add(plusButton);
        comboBoxPane.setAlignmentX(Component.CENTER_ALIGNMENT);
        rulePane.add(comboBoxPane);
	}
    private String[] getMetaDataList() {
    	String[] returnS = new String[0];
    	List<String> names = databaseAPI.getPrimaryMetadataNames();
    	for (String s:TAUdbTrial.TRIAL_COLUMNS )
    		names.add(s);
		return names.toArray(returnS);
	}
    private Component addNumberField(ViewCreatorRuleListener listener){
        //Put the JComboBox in a JPanel to get a nicer look.
        JPanel comboBoxPane = new JPanel(); //use FlowLayout
        String comboBoxItems[] = {NUMBER_EQUAL, NUMBER_NOT, NUMBER_GREATER, NUMBER_LESS, NUMBER_RANGE};
        JComboBox<String> cb = new JComboBox<String>(comboBoxItems);
        cb.addActionListener(listener);
        cb.setEditable(false);
        //cb.setSelectedIndex(0);
        
        
        //Create the "cards".
        JPanel greaterCard = new JPanel();
		JTextField text = new JTextField();
		// text.setValue(0.0);
        text.setPreferredSize(new Dimension(100, 20));
        text.getDocument().addDocumentListener(listener);
        greaterCard.add(text);
        
        
        JPanel lessCard = new JPanel();
		text = new JTextField();
		// text.setValue(0.0);
        text.setPreferredSize(new Dimension(100, 20));
        text.getDocument().addDocumentListener(listener);
        lessCard.add(text);
        
        JPanel equalCard = new JPanel();
		text = new JTextField();
		// text.setValue(0.0);
        text.setPreferredSize(new Dimension(100, 20));
        text.getDocument().addDocumentListener(listener);
        equalCard.add(text);
        
        JPanel notEqualCard = new JPanel();
		text = new JTextField();
		// text.setValue(0.0);
       text.setPreferredSize(new Dimension(100, 20));
       text.getDocument().addDocumentListener(listener);
       notEqualCard.add(text);
        
        JPanel rangeCard = new JPanel();
		text = new JTextField();
		// text.setValue(0.0);
        text.setPreferredSize(new Dimension(100, 20));
        text.getDocument().addDocumentListener(listener);
        text.getDocument().putProperty(NUMBER_RANGE,"begin");
        rangeCard.add(text);

		text = new JTextField();
		// text.setValue(0.0);
        text.setPreferredSize(new Dimension(100, 20));
        text.getDocument().addDocumentListener(listener);
        text.getDocument().putProperty(NUMBER_RANGE,"end");
        rangeCard.add(text);


        
        //Create the panel that contains the "cards".
        JPanel comparators = new JPanel(new CardLayout());
        comparators.add(equalCard, NUMBER_EQUAL);
        comparators.add(notEqualCard, NUMBER_NOT);
        comparators.add(greaterCard, NUMBER_GREATER);
        comparators.add(lessCard, NUMBER_LESS);
        comparators.add(rangeCard, NUMBER_RANGE);

        cb.addItemListener(new ViewCreatorListner(comparators));
       
        
        comboBoxPane.add(cb, BorderLayout.WEST);
        comboBoxPane.add(comparators, BorderLayout.EAST);
        
        return comboBoxPane;
    	
    }
    private Component addDateField(){
        //Put the JComboBox in a JPanel to get a nicer look.
        JPanel comboBoxPane = new JPanel(); //use FlowLayout
        String comboBoxItems[] = {DATE_IS, DATE_AFTER, DATE_BEFORE, DATE_RANGE};
        JComboBox<String> cb = new JComboBox<String>(comboBoxItems);
        cb.setEditable(false);
        cb.setSelectedIndex(0);
        
        //Create the "cards".
        JPanel greaterCard = new JPanel();
        JFormattedTextField date = new JFormattedTextField(new SimpleDateFormat("MM/dd/yyyy"));
        date.setValue(new Date());
        greaterCard.add(date);
        
        
        JPanel lessCard = new JPanel();
        date = new JFormattedTextField(new SimpleDateFormat("MM/dd/yyyy"));
        date.setValue(new Date());
        lessCard.add(date);
        
        JPanel equalCard = new JPanel();
        date = new JFormattedTextField(new SimpleDateFormat("MM/dd/yyyy"));
        date.setValue(new Date());
        equalCard.add(date);
        
        JPanel rangeCard = new JPanel();
        date = new JFormattedTextField(new SimpleDateFormat("MM/dd/yyyy"));
        date.setValue(new Date());
        rangeCard.add(date);
        date = new JFormattedTextField(new SimpleDateFormat("MM/dd/yyyy"));
        date.setValue(new Date());
        rangeCard.add(date);
        
        //Create the panel that contains the "cards".
        JPanel comparators = new JPanel(new CardLayout());
        comparators.add(equalCard, DATE_IS);
        comparators.add(greaterCard, DATE_AFTER);
        comparators.add(lessCard, DATE_BEFORE);
        comparators.add(rangeCard, DATE_RANGE);

        cb.addItemListener(new ViewCreatorListner(comparators));
       
        
        comboBoxPane.add(cb, BorderLayout.WEST);
        comboBoxPane.add(comparators, BorderLayout.EAST);
        
        return comboBoxPane;
    	
    }
	private Component addStringField(ViewCreatorRuleListener listener){
        //Put the JComboBox in a JPanel to get a nicer look.
        JPanel comboBoxPane = new JPanel(); //use FlowLayout
        String comboBoxItems[] = {STRING_EXACTLY,STRING_BEGINS, STRING_ENDS, STRING_CONTAINS};
        JComboBox<String> cb = new JComboBox<String>(comboBoxItems);
        cb.setEditable(false);
      //  cb.setName(STRING);
        cb.addActionListener(listener);
        cb.setSelectedIndex(0);
        
        //Create the "cards".
        JPanel beginCard = new JPanel();
        JTextField begin = new JTextField("", 20);
        begin.getDocument().addDocumentListener(listener);
        beginCard.add(begin);
        
        JPanel containsCard = new JPanel();
        JTextField contains = new JTextField("", 20);
        contains.getDocument().addDocumentListener(listener);
        containsCard.add(contains);
        
        JPanel endCard = new JPanel();
        JTextField end = new JTextField("", 20);
        end.getDocument().addDocumentListener(listener);
        endCard.add(end);
        
        JPanel exactlyCard = new JPanel();
        JTextField exactly = new JTextField("", 20);
        exactly.getDocument().addDocumentListener(listener);
        exactlyCard.add(exactly);
        

        
        //Create the panel that contains the "cards".
        JPanel comparators = new JPanel(new CardLayout());
        comparators.add(exactlyCard, STRING_EXACTLY);
        comparators.add(beginCard, STRING_BEGINS);
        comparators.add(endCard, STRING_ENDS);
        comparators.add(containsCard, STRING_CONTAINS);

        cb.addItemListener(new ViewCreatorListner(comparators));

        comboBoxPane.add(cb, BorderLayout.WEST);
        comboBoxPane.add(comparators, BorderLayout.EAST);
        
        return comboBoxPane;
    	
    }
    


    
    /**
     * Create the GUI and show it.  For thread safety,
     * this method should be invoked from the
     * event dispatch thread.
     */
    private static void createAndShowGUI() {
        //Create and set up the window.
        //Create and set up the content pane.
        ViewCreatorGUI frame = new ViewCreatorGUI(null);
       
        
        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }
    

	public static void main(String[] args) {
        /* Use an appropriate Look and Feel */
        try {
            //UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
            UIManager.setLookAndFeel("javax.swing.plaf.metal.MetalLookAndFeel");
        } catch (UnsupportedLookAndFeelException ex) {
            ex.printStackTrace();
        } catch (IllegalAccessException ex) {
            ex.printStackTrace();
        } catch (InstantiationException ex) {
            ex.printStackTrace();
        } catch (ClassNotFoundException ex) {
            ex.printStackTrace();
        }
        /* Turn off metal's use of bold fonts */
        UIManager.put("swing.boldMetal", Boolean.FALSE);
        
        //Schedule a job for the event dispatch thread:
        //creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
    }

}

