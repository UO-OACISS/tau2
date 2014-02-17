package edu.uoregon.tau.perfdmf.taudb;

import java.awt.Container;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JComboBox;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.BadLocationException;

import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.View.ViewRule;


public class ViewCreatorRuleListener implements DocumentListener, ActionListener {
	Container container = null;
	ViewRule rule = new ViewRule();
	/**
	 * We need to hang on to the container of the +/- buttons so we can check if
	 * it is disabled. We ignore rules from disabled containers.
	 * 
	 * @param c
	 */
	public void setContainer(Container c) {
		this.container = c;
	}

	public boolean isRuleEnabled() {
		return (container != null && container.isEnabled());
	}

	private void change (DocumentEvent e){
		try {			
			 
			if(e.getDocument().getProperty(View.NUMBER_RANGE) != null){
				String range = (String)e.getDocument().getProperty(View.NUMBER_RANGE) ;
				if(range.equals("begin")){
					rule.setValue(e.getDocument().getText(0, e.getDocument().getLength()));
				}else{
					rule.setValue2(e.getDocument().getText(0, e.getDocument().getLength()));
				}
			}else{
				rule.setValue(e.getDocument().getText(0, e.getDocument().getLength()));
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
			if(combo.getName() == View.NUMBER_RANGE){
				rule.setOperator(View.NUMBER_RANGE);
			}else if(combo.getName() == ViewCreatorGUI.METADATA){
				rule.setColumn_name(combo.getSelectedItem().toString());				
			}else if(combo.getName() == ViewCreatorGUI.READ_TYPE){
				rule.setType(combo.getSelectedItem().toString());
			}else {
				rule.setOperator(combo.getSelectedItem().toString());
			
				if (rule.getOperator() == View.STRING_EXACTLY) {
						rule.setOperator("=");
					} else if (rule.getOperator() == View.NUMBER_EQUAL) {
						rule.setOperator("=");
					} else if (rule.getOperator() == View.NUMBER_NOT) {
						rule.setOperator("!=");
					} else if (rule.getOperator() == View.NUMBER_GREATER) {
						rule.setOperator(">");
					} else if (rule.getOperator() == View.NUMBER_LESS) {
						rule.setOperator("<");
					} else if (rule.getOperator() == View.NUMBER_RANGE) {
					//Need to create two rules in this case.
					rule.setOperator(View.NUMBER_RANGE);
				}
			}
		}
 	}
	
	public ViewRule getViewRule() {
		return rule;
	}

}
