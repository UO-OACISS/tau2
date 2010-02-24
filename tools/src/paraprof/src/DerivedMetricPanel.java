/*  
 DerivedMetricPanel.java

 Title:      ParaProf
 Author:     Robert Bell
 Description:  
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.io.CharArrayReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.ListIterator;

import javax.swing.*;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.perfdmf.DBDataSource;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Trial;

public class DerivedMetricPanel extends JPanel implements ActionListener {

	private ParaProfManagerWindow paraProfManager = null;
	private JTextField arg1Field = new JTextField("", 30);



	public DerivedMetricPanel(ParaProfManagerWindow paraProfManager) {
		this.paraProfManager = paraProfManager;

		int windowWidth = 800;
		int windowHeight = 200;
		setSize(new java.awt.Dimension(windowWidth, windowHeight));

		//Set component properties.
		arg1Field.setEditable(true);

		//Create and add the components.


		//Setting up the layout system for the main window.
		GridBagLayout gbl = new GridBagLayout();
		this.setLayout(gbl);
		GridBagConstraints gbc = new GridBagConstraints();
		gbc.insets = new Insets(5, 5, 5, 5);

		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.WEST;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(new JLabel("Expression:"), gbc, 0, 0, 1, 1);

		gbc.fill = GridBagConstraints.BOTH;
		gbc.anchor = GridBagConstraints.WEST;
		gbc.weightx = 100;
		gbc.weighty = 0;
		addCompItem(arg1Field, gbc, 1, 0, 7, 1);
		
		JButton jClear = new JButton("Clear");
		jClear.addActionListener(this);
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.WEST;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(jClear, gbc, 8,0, 1, 1);



		JButton jButton = new JButton("Apply");
		jButton.addActionListener(this);
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.WEST;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(jButton, gbc, 7, 2, 1, 1);

		JButton leftParen = new JButton("(");
		leftParen.addActionListener(this);
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.WEST;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(leftParen, gbc, 5, 2, 1, 1);

		JButton rightParen = new JButton(")");
		rightParen.addActionListener(this);
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(rightParen, gbc, 6, 2, 1, 1);

		JButton plus = new JButton("+");
		plus.addActionListener(this);
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(plus, gbc, 0, 2, 1, 1);

		JButton minus = new JButton("-");
		minus.addActionListener(this);
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(minus, gbc, 1, 2, 1, 1);

		JButton times = new JButton("*");
		times.addActionListener(this);
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(times, gbc, 2, 2, 1, 1);

		JButton divide = new JButton("/");
		divide.addActionListener(this);
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(divide, gbc, 3, 2, 1, 1);

		JButton equals = new JButton("=");
		equals.addActionListener(this);
		gbc.fill = GridBagConstraints.NONE;
		gbc.anchor = GridBagConstraints.CENTER;
		gbc.weightx = 0;
		gbc.weighty = 0;
		addCompItem(equals, gbc, 4, 2, 1, 1);


		//Copy Paste:
		JMenuItem copy = new JMenuItem("Copy");
		copy.addActionListener(this);
		copy.setAccelerator(
				KeyStroke.getKeyStroke(KeyEvent.VK_C, ActionEvent.CTRL_MASK));
		copy.setMnemonic(KeyEvent.VK_C);

		JMenuItem cut = new JMenuItem("Cut");
		cut.setAccelerator(
				KeyStroke.getKeyStroke(KeyEvent.VK_X, ActionEvent.CTRL_MASK));
		cut.setMnemonic(KeyEvent.VK_T);
		cut.addActionListener(this);


		JMenuItem paste = new JMenuItem("Paste");
		paste.setAccelerator(
				KeyStroke.getKeyStroke(KeyEvent.VK_V, ActionEvent.CTRL_MASK));
		paste.setMnemonic(KeyEvent.VK_P);
		paste.addActionListener(this);


	}

	public void setArg1Field(String arg1) {
		arg1Field.setText(arg1);
	}

	public String getArg1Field() {
		return arg1Field.getText().trim();
	}

	private void copy(){

		String expressions= arg1Field.getText();
		setClipboard(expressions);
	}
	private void paste() throws IOException{
		String clip = getFromClipboard();
		arg1Field.setText(clip);
	}


	private void setClipboard(String in){
		StringSelection stringSelection = new StringSelection( in );
		Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
		clipboard.setContents( stringSelection, stringSelection );
	}
	private String getFromClipboard(){
		String result = "";
		Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
		Transferable contents = clipboard.getContents(null);
		boolean hasTransferableText =
			(contents != null) &&
			contents.isDataFlavorSupported(DataFlavor.stringFlavor);
		if ( hasTransferableText ) {
			try {
				result = (String)contents.getTransferData(DataFlavor.stringFlavor);
			}
			catch (UnsupportedFlavorException ex){
				//highly unlikely since we are using a standard DataFlavor
				System.out.println(ex);
				ex.printStackTrace();
			}
			catch (IOException ex) {
				System.out.println(ex);
				ex.printStackTrace();
			}
		}
		return result;

	}




	private void applyToTrial(ParaProfTrial trial, String expression) throws MetricNotFoundException{
		if(trial !=null){
			while (trial.loading()) {
				sleep(500);
			}
		
			try{
				paraProfManager.expandTrial(trial);
				ParaProfExpression exp1 = new ParaProfExpression();
				ParaProfMetric metric = exp1 .evaluateExpression(trial, expression);					
				if (metric != null) {
					if (metric.getParaProfTrial().dBTrial()) {
						paraProfManager.uploadMetric(metric);
					}
					paraProfManager.populateTrialMetrics(metric.getParaProfTrial());
				}							

			}catch(ParsingException ex){
				ex.printStackTrace();
				//TODO Alert User

			}

		}
	}

	public void applyOperation()   {

		if(!ParaProfExpression.validate(arg1Field.getText())){
			JOptionPane.showMessageDialog(paraProfManager, 
					"The expression entered is not valid.",
					"Expression Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
		DefaultMutableTreeNode sel = paraProfManager.getSelectedObject();

		if(sel==null){
			JOptionPane.showMessageDialog(paraProfManager, 
					"Please select a trial, experiment or application.",
					"Warning", JOptionPane.WARNING_MESSAGE);
			return;
		}

		ArrayList collectTrials = collectTrials(sel); 

		ArrayList errors = new ArrayList();
		for (int i=0;i<collectTrials.size();i++){
			try {
				applyToTrial((ParaProfTrial) collectTrials.get(i), arg1Field.getText());
			} catch (MetricNotFoundException e) {
				errors.add(collectTrials.get(i));
			}
		}
		if(errors.size()>0){
			errors.add(0,"The metric could not be derived for the following trials because they did not contain all of the metrics required.\n");
			JOptionPane.showMessageDialog(paraProfManager, 
					errors.toArray(),
					"Warning", JOptionPane.WARNING_MESSAGE);
		}

	}
	private ArrayList collectTrials(DefaultMutableTreeNode sel) {

		Object selected = paraProfManager.getSelectedObject().getUserObject();
		ArrayList collectTrials = new ArrayList();
		if(selected instanceof ParaProfMetric){
			ParaProfMetric met = (ParaProfMetric) selected;
			collectTrials .add(met.getParaProfTrial());
		}else if(selected instanceof ParaProfTrial){
			collectTrials.add(sel.getUserObject());
		}else if(selected instanceof ParaProfExperiment){
			paraProfManager.expand(sel);
			Enumeration trials = sel.children();
			while(trials.hasMoreElements()){
				DefaultMutableTreeNode next = (DefaultMutableTreeNode)trials.nextElement();
				paraProfManager.expand(next);
				collectTrials.add(next.getUserObject());
			}
		}else if(selected instanceof ParaProfApplication){
			paraProfManager.expand( sel);
			Enumeration exps = sel.children();
			while(exps.hasMoreElements()){
				DefaultMutableTreeNode next = (DefaultMutableTreeNode)exps.nextElement();
				paraProfManager.expand(next);
				Enumeration trls = next.children();
				while(trls.hasMoreElements()){
					DefaultMutableTreeNode node = (DefaultMutableTreeNode) trls.nextElement();
					paraProfManager.expand(node);
					collectTrials.add(node.getUserObject());
				}
			}

		}else{
			JOptionPane.showMessageDialog(paraProfManager, 
					"Please select a trial, experiment or application.",
					"Warning", JOptionPane.WARNING_MESSAGE);
			//Please select a trial, experiment or application
		}		return collectTrials;
	}

	private static void sleep(int msec) {
		try {
			java.lang.Thread.sleep(msec);
		} catch (Exception e) {
			throw new RuntimeException("Exception while sleeping");
		}
	}
	public void actionPerformed(ActionEvent evt) {
		try {
			String arg = evt.getActionCommand();
			if (arg.equals("Apply")) {
				applyOperation();
			}else if(arg.equals("Clear")){
				arg1Field.setText("");
			}else if(arg.equals("+")){
				insertString("+");
			}else if(arg.equals("+")){
				insertString("+");
			}else if(arg.equals("-")){
				insertString("-");
			}else if(arg.equals("*")){
				insertString("*");
			}else if(arg.equals("/")){
				insertString("/");
			}else if(arg.equals("(")){
				insertString(")");
			}else if(arg.equals(")")){
				insertString(")");
			}else if(arg.equals("=")){
				insertString("=");
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
		this.add(c, gbc);
	}

	public void insertMetric(ParaProfMetric metric) {
		arg1Field.replaceSelection("\""+metric.getName()+"\"");//puts string at point of cursor
		arg1Field.requestFocusInWindow();
		//de-select the text
		int pos=arg1Field.getSelectionStart();
		arg1Field.setCaretPosition(pos);
		if(arg1Field.getText().length()==pos){
			arg1Field.setText(arg1Field.getText()+" ");
			arg1Field.setCaretPosition(pos);
		}
	}
	private void insertString(String s){
		arg1Field.replaceSelection(s);//puts string at point of cursor
		arg1Field.requestFocusInWindow();
		//de-select the text
		int pos=arg1Field.getSelectionStart();
		arg1Field.setCaretPosition(pos);
		if(arg1Field.getText().length()==pos){
			arg1Field.setText(arg1Field.getText()+" ");
			arg1Field.setCaretPosition(pos);
		}
	}

	public void applyExpressionFile(LineNumberReader scan) throws IOException {
		String expression = scan.readLine();
		ArrayList errors = new ArrayList();

		DefaultMutableTreeNode sel = paraProfManager.getSelectedObject();

		if(sel==null){
			JOptionPane.showMessageDialog(paraProfManager, 
					"Please select a trial, experiment or application.",
					"Warning", JOptionPane.WARNING_MESSAGE);
			return;
		}else if(!(		(sel.getUserObject() instanceof ParaProfMetric)||(sel.getUserObject() instanceof ParaProfTrial)||(sel.getUserObject() instanceof ParaProfTrial)||(sel.getUserObject() instanceof ParaProfApplication))){
			JOptionPane.showMessageDialog(paraProfManager,
					"Please select a trial, experiment or application.",
					"Warning", JOptionPane.WARNING_MESSAGE);
			return;

		}
		while(expression != null){
			expression = expression.trim();

			if(!expression.equals("")){
				if(!ParaProfExpression.validate(expression)){
					JOptionPane.showMessageDialog(paraProfManager, 
							"The expression entered is not valid.",
							"Expression Error", JOptionPane.ERROR_MESSAGE);
				}else{

					ArrayList collectTrials = collectTrials(sel); 

					for (int i=0;i<collectTrials.size();i++){
						try {
							applyToTrial((ParaProfTrial) collectTrials.get(i), expression);
						} catch (MetricNotFoundException e) {
							errors.add(collectTrials.get(i));
						}
					}
				}

			}
			expression = scan.readLine();
		}
		if(errors.size()>0){
			errors.add(0,"The metric could not be derived for the follow trials because they did not contain all of the metrics required.\n");
			JOptionPane.showMessageDialog(paraProfManager, 
					errors.toArray(),
					"Warning", JOptionPane.WARNING_MESSAGE);
		}
	}

}
