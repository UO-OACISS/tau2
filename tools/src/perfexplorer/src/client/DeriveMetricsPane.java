package edu.uoregon.tau.perfexplorer.client;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import java.util.Enumeration;
import java.util.List;
import java.util.ListIterator;
import java.util.Scanner;

import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import javax.swing.border.TitledBorder;
import javax.swing.plaf.basic.BasicComboPopup;
import javax.swing.plaf.basic.ComboPopup;
import javax.swing.plaf.metal.MetalComboBoxUI;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.common.PythonInterpreterFactory;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.common.ScriptThread;

public class DeriveMetricsPane extends JScrollPane implements ActionListener {

   /**
    * 
    */

   private static final String IDEAL="Ideal";

   private static final long serialVersionUID = -8971827392560223964L;
   private static DeriveMetricsPane thePane = null;
   private PerfExplorerConnection server = null;

   private JPanel mainPanel = null;

   private static Application selectApp = null;
   private static Experiment selectExp = null;
   private static Trial selectTrial = null;

   private JButton derive = new JButton("Apply");
   private JLabel trialsSelected = new JLabel("<html><p>No Trials Selected<br/ <br/ <br/ </p> ");
   private JButton saveScript = new JButton("Save as Script");

   private JFileChooser fc = new JFileChooser(System.getProperty("user.dir"));



   private JTextArea addExpression = new  JTextArea(4,10 );
   private JButton leftParen = new JButton("(");
   private JButton rightParen = new JButton(")");
   private JButton plus = new JButton("+");
   private JButton minus = new JButton("-");
   private JButton times = new JButton("*");
   private JButton divide = new JButton("/");
   private JButton clear = new JButton("Clear");
   private JButton apply  =new JButton("Apply");
   private JButton equals = new JButton("=");
   private JButton insert = new JButton("Add to List");

   private DefaultListModel expressionList = new DefaultListModel();
   private JButton edit = new JButton("Edit");
   private JButton remove = new JButton("Remove");
   private JButton selectAll = new JButton("Select All");
   private JButton deselect = new JButton("Deselect All");
   private JButton validate = new JButton("Validate");
   JList expression = new JList(expressionList)
   {//This allows the tool tip to display the expression
      //the mouse is over.
      public String getToolTipText(MouseEvent e) {
         int index = locationToIndex(e.getPoint());
         if (-1 < index) {
            String line = expressionList.get(index).toString();
            String wrap = "<html>",end="<html>";
            char[] array = line.toCharArray();
            for(int i=0;i<array.length;i++){   
               wrap +=array[i];
               end =wrap;
               if(i%80==0&&i!=0) wrap +="<br>";
            }         
            return end.trim();
         } else {
            return null;
         }
      }
   };



   public static DeriveMetricsPane getPane () {

      if (thePane == null) {
         JPanel mainPanel = new JPanel(new GridLayout(1,3,10,5));
         thePane = new DeriveMetricsPane(mainPanel);
      }
      thePane.repaint();
      return thePane;
   }

   private DeriveMetricsPane (JPanel mainPanel) {
      super(mainPanel);

      this.server = PerfExplorerConnection.getConnection();
      this.mainPanel = mainPanel;

      JScrollBar jScrollBar = this.getVerticalScrollBar();
      jScrollBar.setUnitIncrement(35);

      JPanel panel = new JPanel();
      panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
      panel.setAlignmentY(TOP_ALIGNMENT);
      JPanel create =createCreateExpressionMenu();
      create.setAlignmentX(LEFT_ALIGNMENT);
      panel.add(create);
   
      JPanel list = createExpressionListMenu();
      list.setAlignmentX(LEFT_ALIGNMENT);
      panel.add(list);
      
      JPanel apply = createApplyExpression();
      apply.setAlignmentX(LEFT_ALIGNMENT);
      panel.add(apply);   
         
      this.mainPanel.add(panel, BorderLayout.EAST);

      resetChartSettings();
   }

   private JPanel createApplyExpression() {
      JPanel panel = new JPanel();
      TitledBorder tb = BorderFactory.createTitledBorder("Apply Derived Metric Expressions");

      panel.setBorder(tb);
      panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
      trialsSelected.setAlignmentX(LEFT_ALIGNMENT);
      trialsSelected.setMaximumSize(new Dimension(1000,100));
      derive.setAlignmentX(LEFT_ALIGNMENT);

      panel.add(trialsSelected);
      panel.add(derive);
      derive.addActionListener(this);

      return (panel);
   }

   private JPanel createExpressionListMenu() {


      expression = new JList(expressionList)
      {//This allows the tool tip to display the expression
         //the mouse is over.
         public String getToolTipText(MouseEvent e) {
            int index = locationToIndex(e.getPoint());
            if (-1 < index) {
               String line = expressionList.get(index).toString();
               String wrap = "<html>",end="<html>";
               char[] array = line.toCharArray();
               for(int i=0;i<array.length;i++){   
                  wrap +=array[i];
                  end =wrap;
                  if(i%80==0&&i!=0) wrap +="<br>";
               }         
               return end.trim();
            } else {
               return null;
            }
         }
      };


      JScrollPane expressionScroll = new JScrollPane(expression);
      expressionScroll.setPreferredSize(new Dimension(400, 350));

      // create a new panel, 
      JPanel panel = new JPanel();
      TitledBorder tb = BorderFactory.createTitledBorder("Select Derived Metric  Expressions");
      panel.setBorder(tb);
      panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
      
      JPanel opt = createSelectionOptionsPanel();
      
      expressionScroll.setAlignmentY(TOP_ALIGNMENT);
      panel.add(expressionScroll);
      opt.setAlignmentY(TOP_ALIGNMENT);
      panel.add(opt);

      return (panel);
   }
   private JPanel createSelectionOptionsPanel() {
      JPanel panel = new JPanel();
      panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
      
      Dimension maxDim = new Dimension(150, 30);
      selectAll.setMaximumSize(maxDim);
      deselect.setMaximumSize(maxDim);
      edit.setMaximumSize(maxDim);
      validate.setMaximumSize(maxDim);
      remove.setMaximumSize(maxDim);

      selectAll.setAlignmentX(LEFT_ALIGNMENT);
      deselect.setAlignmentX(LEFT_ALIGNMENT);
      edit.setAlignmentX(LEFT_ALIGNMENT);
      validate.setAlignmentX(LEFT_ALIGNMENT);
      remove.setAlignmentX(LEFT_ALIGNMENT);

      selectAll.setAlignmentY(TOP_ALIGNMENT);
      deselect.setAlignmentY(TOP_ALIGNMENT);
      edit.setAlignmentY(TOP_ALIGNMENT);
      validate.setAlignmentY(TOP_ALIGNMENT);
      remove.setAlignmentY(TOP_ALIGNMENT);

      selectAll.addActionListener(this);
      deselect.addActionListener(this);
      edit.addActionListener(this);
      validate.addActionListener(this);
      remove.addActionListener(this);


      panel.add(selectAll);
      panel.add(deselect);
      panel.add(edit);
      panel.add(validate);
      panel.add(remove);

      return panel;
   }


   private JPanel createCreateExpressionMenu() {
      JPanel panel = new JPanel();

      TitledBorder tb = BorderFactory.createTitledBorder("Create Derived Metric Expression");
      panel.setBorder(tb);
      panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
   
      JPanel add = createAddOptions() ;
      add.setAlignmentX(LEFT_ALIGNMENT);
      panel.add(add);
        
      JPanel oper = createOperPanel();
      oper.setAlignmentX(LEFT_ALIGNMENT);
      panel.add(oper);
      return (panel);
   }


   private JPanel createAddOptions() {
      JPanel panel = new JPanel();

      panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));

      addExpression.setAlignmentY(TOP_ALIGNMENT);
      JScrollPane expressionScroll = new JScrollPane(addExpression);
      expressionScroll.setAlignmentY(TOP_ALIGNMENT);
      
      addExpression.setLineWrap(true);

      panel.add(expressionScroll);
        
      clear.setAlignmentY(TOP_ALIGNMENT);
      panel.add(clear);
      clear.addActionListener(this);
      
      return panel;
   }

   private JPanel createOperPanel() {
      JPanel opers = new JPanel();
      opers.setLayout(new BoxLayout(opers, BoxLayout.X_AXIS));

      Dimension dim = new Dimension(60,5);
      plus.setPreferredSize(dim);
      minus.setPreferredSize(dim);
      divide.setPreferredSize(dim);
      times.setPreferredSize(dim);
      rightParen.setPreferredSize(dim);
      leftParen.setPreferredSize(dim);
      equals.setPreferredSize(dim);
      
      plus.setAlignmentY(TOP_ALIGNMENT);
      minus.setAlignmentY(TOP_ALIGNMENT);
      divide.setAlignmentY(TOP_ALIGNMENT);
      times.setAlignmentY(TOP_ALIGNMENT);
      rightParen.setAlignmentY(TOP_ALIGNMENT);
      leftParen.setAlignmentY(TOP_ALIGNMENT);
      equals.setAlignmentY(TOP_ALIGNMENT);
      insert.setAlignmentY(TOP_ALIGNMENT);
      apply.setAlignmentY(TOP_ALIGNMENT);

      plus.addActionListener(this);
      minus.addActionListener(this);
      divide.addActionListener(this);
      times.addActionListener(this);
      rightParen.addActionListener(this);
      leftParen.addActionListener(this);
      equals.addActionListener(this);
      insert.addActionListener(this);
      apply.addActionListener(this);

      opers.add(plus);
      opers.add(minus);
      opers.add(divide);
      opers.add(times);
      opers.add(equals);
      opers.add(leftParen);
      opers.add(rightParen);
      opers.add(insert);
      opers.add(apply);

      return opers;
   }

   private void resetChartSettings() {
      // top toggle buttons
      refreshDynamicControls(true, true, false);
   }

   public void metricClick(Metric metric) {
      insertText(metric.getName());
   }

   private void insertText(String text) {
      addExpression.replaceSelection(text);//puts string at point of cursor
      requestFocusInWindow();
      //de-select the text
      int pos=addExpression.getSelectionStart();
      addExpression.setCaretPosition(pos);
      if(addExpression.getText().length()==pos){
         addExpression.setText(addExpression.getText()+" ");
         addExpression.setCaretPosition(pos);
      }

   }

   public void refreshDynamicControls(boolean getMetrics, boolean getEvents, boolean getXML) {
      PerfExplorerModel theModel = PerfExplorerModel.getModel();
      Object selection = theModel.getCurrentSelection();
      String oldMetric = "";
      String oldXML = "";
      String oldSXML="";
      Object obj = null;

      PerfExplorerConnection server = PerfExplorerConnection.getConnection();


      if(selection instanceof Application){
         selectApp = (Application) selection;
         trialsSelected.setText("<html><p>The selected expressions will be applied to \n "

               +"<br/ all the trials in the all of the experiment <br/ in the \"" +selectApp.getName()+"\" application.</p>");
         this.selectExp = null;
         this.selectTrial = null;
      }else if(selection instanceof Experiment){
         selectExp = (Experiment) selection;
         ListIterator<Application> apps = server.getApplicationList();
         while(apps.hasNext()){
            selectApp = apps.next();
            if(selectApp.getID()==selectExp.getApplicationID()){
               break;
            }
         }
         this.selectTrial = null;
         trialsSelected.setText("<html> <p> The selected expressions will be applied to \n "+
               "<br/all the trials in the \""+selectExp.getName()
               +"\" experiment \n<br/in the \""+selectApp.getName()+"\" application. \n </p> ");
      }else if(selection instanceof Trial){
         selectTrial = (Trial)selection;
         ListIterator<Application> apps = server.getApplicationList();
         while(apps.hasNext()){
            selectApp = apps.next();
            if(selectApp.getID()==selectTrial.getApplicationID()){
               break;
            }
         }
         ListIterator<Experiment> experiments = server.getExperimentList(selectTrial.getApplicationID());
         while(experiments.hasNext()){
            selectExp = experiments.next();
            if(selectExp.getID()==selectTrial.getExperimentID()){
               break;
            }
         }
         trialsSelected.setText("<html> <p> The selected expressions will be applied to "+
               "the \""+selectTrial.getName() +"\" trial \n <br/from the \""+selectExp.getName()
               +"\" experiment\n <br/ from the \""+selectApp.getName()+"\" application.\n</p>");
      }

      if(selectTrial !=null){

      }else if(selectExp !=null){

      }else if(selectApp !=null){

      }
      if (selection instanceof Metric){


      }
   }



   public void actionPerformed(ActionEvent e) {
      //   try {
      Object arg = e.getSource();

      if(arg.equals(selectAll)){
         int[] ind =new int[expressionList.getSize()];
         for (int i=0;i<expressionList.getSize();i++){
            ind[i]=i;
         }
         expression.setSelectedIndices(ind);
      }else if(arg.equals(this.insert)){

         String text = addExpression.getText();

         if(text == null){         JOptionPane.showMessageDialog(this,
               "You cannot add a blank expression.",
               "Warning",JOptionPane.WARNING_MESSAGE);
         }
         else if(text.trim().equals("")){
            JOptionPane.showMessageDialog(this,
                  "You cannot add a blank expression.",
                  "Warning",JOptionPane.WARNING_MESSAGE);
         }else{
            boolean isvalid = validate(text);
            if(isvalid){
               expressionList.addElement(text.trim());
            }
         }

      }else if(arg.equals(deselect)){
         expression.getSelectionModel().clearSelection();
      }else if(arg.equals(apply)){
         if(addExpression.getText().trim().equals("")){
            JOptionPane.showMessageDialog(this,
                  "Please select an expression.",
                  "Warning",
                  JOptionPane.WARNING_MESSAGE);
         }else
            deriveMetric(addExpression.getText().trim());
      }else if(arg.equals(derive)){
         if(expression.isSelectionEmpty()){
            JOptionPane.showMessageDialog(this,
                  "Please select an expression.",
                  "Warning",
                  JOptionPane.WARNING_MESSAGE);
         }else
            deriveMetric();
      }else if(arg.equals(edit)){
         if(expression.isSelectionEmpty()){
            JOptionPane.showMessageDialog(this,
                  "Please select an expression.",
                  "Warning",
                  JOptionPane.WARNING_MESSAGE);
         }else{
            edit(expression.getSelectedValue().toString());
         }
      }else if(arg.equals(remove)){
         remove();
      }else if (arg.equals(validate)){

         String text = expression.getSelectedValue().toString();


         boolean isvalid = validate(text);
         if(isvalid){
            JOptionPane.showMessageDialog(this,
                  "This expression is correct.",
                  "Correct",
                  JOptionPane.INFORMATION_MESSAGE);
         }

      }else if(arg.equals(plus)){
         insertText("+");
      }else if(arg.equals(minus)){
         insertText("-");
      }else if(arg.equals(times)){
         insertText("*");
      }else if(arg.equals(divide)){
         insertText("/");
      }else if(arg.equals(leftParen)){
         insertText("(");
      }else if(arg.equals(rightParen)){
         insertText(")");
      }else if(arg.equals(equals)){
         insertText("=");
      }else if(arg.equals(clear)){
         addExpression.setText("");
         addExpression.requestFocusInWindow();
      }else{
      }         
      //      } catch (FileNotFoundException ex) {
      //         // TODO Auto-generated catch block
      //         ex.printStackTrace();
      //      } catch (IOException ex) {
      //         // TODO Auto-generated catch block
      //         ex.printStackTrace();
      //      } finally {}
   }
   private void remove() {
      if(expression.isSelectionEmpty()){
         JOptionPane.showMessageDialog(this,
               "Please select an expression.",
               "Warning",
               JOptionPane.WARNING_MESSAGE);
      }else{
         while(!expression.isSelectionEmpty())
            expressionList.removeElementAt(expression.getSelectedIndex());
      }
   }
   private void edit(String expr) {
      String s = (String)JOptionPane.showInputDialog(
            this, "Edit Expression:\n","Edit",
            JOptionPane.PLAIN_MESSAGE,null,null,expr);
      if(s!=null){
         boolean isValid = validate(s);
         if(isValid){
            int index = expression.getSelectedIndex();
            expressionList.removeElementAt(index);
            expressionList.add(index, s);
         }else{
            edit(s);
         }
      }
   }

   private boolean validate(String expression){
      boolean result = PerfExplorerExpression.validate(expression);
      if(!result){
         JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(),
               "The expression you entered is not valid.",
               "Invalid Expression",
               JOptionPane.ERROR_MESSAGE);

      }
      return result;
   }
   private void cut (){
      copy();
      remove();
   }

   private void addExpression(){
      addExpression("");
   }
   private void addExpression(String expression) {
      String text = addExpression.getText();

      if(text == null){         JOptionPane.showMessageDialog(this,
            "You cannot add a blank expression.",
            "Warning",JOptionPane.WARNING_MESSAGE);
      }
      else if(text.trim().equals("")){
         JOptionPane.showMessageDialog(this,
               "You cannot add a blank expression.",
               "Warning",JOptionPane.WARNING_MESSAGE);
      }else{
         boolean isvalid = validate(text);
         if(isvalid){
            expressionList.addElement(text.trim());
         }
      }
   }
   private void copy(){
      Object[] selectedExp = expression.getSelectedValues();
      String expressions="";
      for(Object express:selectedExp)   {
         expressions+=express+"\n";
      }
      setClipboard(expressions);
   }
   private void paste(){
      String clip = getFromClipboard();
      addExpressions(new Scanner(clip));
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
   private void openFile() throws FileNotFoundException {
      int returnVal = fc.showOpenDialog(this);
      if (returnVal == JFileChooser.APPROVE_OPTION) {
         addExpressions (new Scanner(fc.getSelectedFile()));
      }

   }
   private void addExpressions(Scanner scan){
      while(scan.hasNextLine()){
         String line = scan.nextLine().trim();
         if(!line.equals("")){
            if(Expression.validate(line))
               expressionList.addElement(line);
            else
               addExpression(line);
         }
      }
   }
   private void saveFile() throws IOException {
      if(getFiletoSave()){
         File savefile =  fc.getSelectedFile();
         FileOutputStream write =new FileOutputStream(savefile);
         String out = "";

         for(Object i: expressionList.toArray()){
            out += i.toString() + "\n";
         }
         write.write(out.getBytes());
      }

   }
   private boolean getFiletoSave() {
      int returnVal = fc.showSaveDialog(this);
      if (returnVal == JFileChooser.APPROVE_OPTION) {
         File savefile =  fc.getSelectedFile();
         if(savefile.exists()){
            int result =    JOptionPane.showOptionDialog(this, 
                  "\""+savefile.getName()+"\" already exists. Do you want to replace it? ", "Replace File"
                  ,JOptionPane.YES_NO_OPTION,JOptionPane.WARNING_MESSAGE, null, null, null);
            if(result ==JOptionPane.NO_OPTION) 
               return getFiletoSave();
         }
         return true;
      }else 
         return false;//user hit cancel 
   }

   private void saveSelected() throws IOException {
      if(expression.isSelectionEmpty()){
         JOptionPane.showMessageDialog(this,
               "Please select an expression.",
               "Warning",
               JOptionPane.WARNING_MESSAGE);
         return;
      }

      if(getFiletoSave()){
         File savefile =  fc.getSelectedFile();
         FileOutputStream write =new FileOutputStream(savefile);
         String out = "";
         int[] indexs = expression.getSelectedIndices();
         for(int i: indexs){
            out += expressionList.get(i) + "\n";
         }
         write.write(out.getBytes());
      }
   }
   private void deriveMetric() {
      Object[] selectedExp = expression.getSelectedValues();
      String expressions="";
      for(Object express:selectedExp)   {
         expressions+=express+"\n";
      }
      deriveMetric(expressions);
  }
   private void deriveMetric(String expressions){

      String message = "Are you sure you want to apply the selected expressions to \n";
      if(selectTrial !=null){
         message += "the \""+selectTrial.getName() +"\" trial from the \""+selectExp.getName()
         +"\" experiment from the \""+selectApp.getName()+"\" application?";
      }else if(selectExp !=null){
         message += "all the trials in the \""+selectExp.getName()
         +"\" experiment in the \""+selectApp.getName()+"\" application?";
      }else if(selectApp !=null){
         message += "all the trials in the all of the exepriments in the \"" +selectApp.getName()+"\" application?";
      }else{
         JOptionPane.showMessageDialog(this,
               "Please select a trial, exerpriment, or application.",
               "Warning",JOptionPane.WARNING_MESSAGE);
         return;
      }

      int result =    JOptionPane.showOptionDialog(this, 
            message, "Confirm Trials"
            ,JOptionPane.YES_NO_OPTION,JOptionPane.WARNING_MESSAGE, null, null, null);
      if(result ==JOptionPane.NO_OPTION) 
         return;

      try{
         PerfExplorerExpression exp = new PerfExplorerExpression();
         String script;
         String app=null,ex=null,trial=null;
         if(selectApp != null)  app = selectApp.getName();
         if(selectExp != null)  ex = selectExp.getName();
         if(selectTrial != null)  trial = selectTrial.getName();
         script = exp.getScriptFromExpressions(app,ex,trial,expressions);   
         PythonInterpreterFactory.defaultfactory.getPythonInterpreter().exec(script);
         
	 PerfExplorerJTree tree = PerfExplorerJTree.getTree();
	 TreePath path = tree.getSelectionPath();
	 TreeNode node = (TreeNode)path.getLastPathComponent();
	 if(((PerfExplorerTreeNode)node.getParent()).getUserObject() instanceof Experiment){
	     tree.collapsePath(path);
	     tree.collapsePath(path.getParentPath());
	     //tree.expandToMetricsAll(path.getParentPath());
	     expandExp(path.getParentPath(), ((Trial)((PerfExplorerTreeNode)node).getUserObject()).getName());
 
	 }else{
	 tree.expandToMetricsAll(path);
	 }
      }catch(ParsingException ex){

         JOptionPane.showMessageDialog(this, 
               "The expression did not parse correctly.\n"+ex.getMessage(),
               "Parse Error", JOptionPane.ERROR_MESSAGE);
      }
   }

private void expandExp(TreePath parentPath, String name) {
    PerfExplorerJTree tree = PerfExplorerJTree.getTree();
    TreeNode node = (TreeNode)parentPath.getLastPathComponent();
	if(((PerfExplorerTreeNode)node).getUserObject() instanceof Metric){
	    return;
	}else if(((PerfExplorerTreeNode)node).getUserObject() instanceof Experiment){
	   // collapsePath(parent);
	}
	    tree.expandPath(parentPath);


	if (node.getChildCount() >= 0) {
	    
	    for (Enumeration e=node.children(); e.hasMoreElements(); ) {
		TreeNode n = (TreeNode)e.nextElement();
		if(((PerfExplorerTreeNode)n).getUserObject() instanceof Trial){
		   Trial t = (Trial) ((PerfExplorerTreeNode)n).getUserObject() ; 
		   if(t.getName().equals(name)){
		       TreePath path = parentPath.pathByAddingChild(n);
		       tree.expandToMetricsAll(path);
		   }
		}
	    }
	}
}
}
