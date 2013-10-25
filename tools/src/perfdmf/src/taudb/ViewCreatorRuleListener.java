package edu.uoregon.tau.perfdmf.taudb;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JComboBox;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.BadLocationException;


public class ViewCreatorRuleListener implements DocumentListener, ActionListener {
	private static final String WILDCARD = "%";
	/*
	 * simple view where the metadata field "Application" is equal to "application" 
INSERT INTO taudb_view (parent, name, conjoin) VALUES (NULL, 'Test View', 'and');
INSERT INTO taudb_view_parameter (taudb_view, table_name, column_name, operator, value) 
VALUES (2, 'primary_metadata', 'Application', '=', 'application');
	 */
	int viewID; //ID for view that this rule applies too
	public int getViewID() {
		return viewID;
	}
	public void setViewID(int viewID) {
		this.viewID = viewID;
	}
	public String getTable_name() {
		return table_name;
	}
	public void setTable_name(String table_name) {
		this.table_name = table_name;
	}
	public String getColumn_name() {
		return column_name;
	}
	public void setColumn_name(String column_name) {
		this.column_name = column_name;
	}
	public String getOperator() {
		if(operator == ViewCreatorGUI.STRING_BEGINS || operator == ViewCreatorGUI.STRING_ENDS
				||operator == ViewCreatorGUI.STRING_CONTAINS){
			return "like";
		}
		return operator;
	}
	public void setOperator(String operator) {
		this.operator = operator;
	}
	public String getValue() {
		if(operator == ViewCreatorGUI.STRING_BEGINS){
			return value+WILDCARD;
		} else if(operator == ViewCreatorGUI.STRING_ENDS){
			return WILDCARD + value;
		}else if(operator == ViewCreatorGUI.STRING_CONTAINS){
			return WILDCARD + value+WILDCARD;
		}

		return value;
	}
	public void setValue(String value) {
		this.value = value;
	}
	public String getValue2() {
		return value2;
	}
	public void setValue2(String value2) {
		this.value2 = value2;
	}
	
	String table_name="primary_metadata"; //primary or secondary metadata
	String column_name=""; //Metadata name
	String operator=""; //= > < 
	String value=""; //value of field
	String value2="";
	String type = "";

	private void change (DocumentEvent e){
		try {			
			 
			if(e.getDocument().getProperty(ViewCreatorGUI.NUMBER_RANGE) != null){
				String range = (String)e.getDocument().getProperty(ViewCreatorGUI.NUMBER_RANGE) ;
				if(range.equals("begin")){
					value = e.getDocument().getText(0, e.getDocument().getLength());
				}else{
					value2 = e.getDocument().getText(0, e.getDocument().getLength());
				}
			}else{
				value = e.getDocument().getText(0, e.getDocument().getLength());
			}
		} catch (BadLocationException e1) {
			e1.printStackTrace();
		}
	}
	public void changedUpdate(DocumentEvent e) {
		change(e);
	}

	public void insertUpdate(DocumentEvent e) {
		change(e);
	}
	public void removeUpdate(DocumentEvent e) {
		change(e);
	}
	public void actionPerformed(ActionEvent arg) {
		if( "comboBoxChanged".equals(arg.getActionCommand())){
			JComboBox<String> combo = (JComboBox<String>) arg.getSource();
			if(combo.getName() == ViewCreatorGUI.NUMBER_RANGE){
				operator = ViewCreatorGUI.NUMBER_RANGE;
			}else if(combo.getName() == ViewCreatorGUI.METADATA){
				column_name = combo.getSelectedItem().toString();
				if(isTrialCol(column_name)){
					table_name="trial"; 
				}else{
					table_name="primary_metadata";
				}
			}else if(combo.getName() == ViewCreatorGUI.READ_TYPE){
				type = combo.getSelectedItem().toString();
			}else {
				operator = combo.getSelectedItem().toString();
			
				if(operator == ViewCreatorGUI.STRING_EXACTLY){
					operator = "=";
				}else if(operator == ViewCreatorGUI.NUMBER_EQUAL){
					operator = "=";
				}else if(operator == ViewCreatorGUI.NUMBER_NOT){
					operator = "!=";
				}else if(operator == ViewCreatorGUI.NUMBER_GREATER){
					operator = ">";
				}else if(operator == ViewCreatorGUI.NUMBER_LESS){
					operator = "<";
				}else if(operator == ViewCreatorGUI.NUMBER_RANGE){
					//Need to create two rules in this case.
					operator = ViewCreatorGUI.NUMBER_RANGE;
				}
			}
		}
 	}
	private boolean isTrialCol(String column) {
		for(String s: TAUdbTrial.TRIAL_COLUMNS)
			if(s.equals(column)) return true;
		return false;
	}

}
