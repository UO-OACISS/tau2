package edu.uoregon.tau.perfdmf.viewcreator;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JComboBox;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.BadLocationException;

public class RuleListener implements DocumentListener, ActionListener {
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
		return operator;
	}
	public void setOperator(String operator) {
		this.operator = operator;
	}
	public String getValue() {
		fixValue();
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
	public String getCompare() {
		return compare;
	}
	public void setCompare(String compare) {
		this.compare = compare;
	}
	String table_name; //primary or secondary metadata
	String column_name; //Metadata name
	String operator; //= > < 
	String value; //value of field
	String value2;
	private String compare;

	private void change (DocumentEvent e){
		// TODO Auto-generated method stub
		try {
			
			 
			if(e.getDocument().getProperty(ViewCreatorGUI.NUMBER_RANGE) != null){
				String range = (String)e.getDocument().getProperty(ViewCreatorGUI.NUMBER_RANGE) ;
				if(range.equals("begin")){
					value = e.getDocument().getText(0, e.getDocument().getLength());
					System.out.println("insert update: " +value);
				}else{
					value2 = e.getDocument().getText(0, e.getDocument().getLength());
					System.out.println("insert update (value2): " +value2);

				}
			}else{
				value = e.getDocument().getText(0, e.getDocument().getLength());
				System.out.println("insert update: " +value);
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
			JComboBox combo = (JComboBox) arg.getSource();
			if(combo.getName() == ViewCreatorGUI.STRING || combo.getName() == ViewCreatorGUI.NUMBER){
			 compare = combo.getSelectedItem().toString();
//			}else if(combo.getName() == ViewCreatorGUI.STRING_BEGINS){
//				operator = "like";
//			}else if(combo.getName() == ViewCreatorGUI.STRING_CONTAINS){
//				operator = "like";
//			}else if(combo.getName() == ViewCreatorGUI.STRING_ENDS){
//				operator = "like";
//			}else if(combo.getName() == ViewCreatorGUI.STRING_EXACTLY){
//				operator = "=";
//			}else if(combo.getName() == ViewCreatorGUI.NUMBER_EQUAL){
//				operator = "=";
//			}else if(combo.getName() == ViewCreatorGUI.NUMBER_GREATER){
//				operator = ">";
//			}else if(combo.getName() == ViewCreatorGUI.NUMBER_LESS){
//				operator = "<";
//			}else if(combo.getName() == ViewCreatorGUI.NUMBER_NOT){
//				operator = "!=";
//
//			}else if(combo.getName() == ViewCreatorGUI.NUMBER_RANGE){
//				operator = ViewCreatorGUI.NUMBER_RANGE;
System.out.println(compare);
			}	else {
			
				column_name = combo.getSelectedItem().toString();
				System.out.println("colname: "+column_name);
			}
		}
	}
	public void fixValue(){
		if(value.contains(WILDCARD)) return;
		if(compare == ViewCreatorGUI.STRING_BEGINS){
			operator = "like";
			value = value+WILDCARD;
		} else if(compare == ViewCreatorGUI.STRING_ENDS){
			operator = "like";
			value = WILDCARD + value;
		}else if(compare == ViewCreatorGUI.STRING_CONTAINS){
			operator = "like";
			value = WILDCARD + value+WILDCARD;
		}else if(compare == ViewCreatorGUI.STRING_EXACTLY){
			operator = "=";
		}else if(compare == ViewCreatorGUI.NUMBER_EQUAL){
			operator = "=";
		}else if(compare == ViewCreatorGUI.NUMBER_GREATER){
			operator = ">";
		}else if(compare == ViewCreatorGUI.NUMBER_LESS){
			operator = "<";
		}else if(compare == ViewCreatorGUI.NUMBER_RANGE){
			//Need to create two rules in this case.
			operator = ViewCreatorGUI.NUMBER_RANGE;
		}
	}
}
