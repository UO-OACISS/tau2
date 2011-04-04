/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.List;
import java.util.Stack;
import java.util.regex.Pattern;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerExpression;

/**
 * @author smillst
 * 
 */
public class DeriveMetricEquation extends AbstractPerformanceOperation {
   /**
	 * 
	 */
	private static final long serialVersionUID = 2874115882488110455L;
private ArrayList<String> equation = null;
   private PerformanceResult input = null;
   private String newName = null;
   private boolean correctEquation = true;

   /**
    * @param input
    */
   public DeriveMetricEquation(PerformanceResult input) {
      super(input);
      // TODO Auto-generated constructor stub
   }

   /**
    * @param trial
    */
   public DeriveMetricEquation(Trial trial) {
      super(trial);
      // TODO Auto-generated constructor stub
   }

   /**
    * @param inputs
    */
   public DeriveMetricEquation(List<PerformanceResult> inputs) {
      super(inputs);
      // TODO Auto-generated constructor stub
   }

   /**
    * @param input
    */
   public DeriveMetricEquation(PerformanceResult input, String infixEquation) {
      super(input);
      this.input = input;
      int index = infixEquation.indexOf("=");
      if(index != -1){
          this.newName = infixEquation.substring(0,index);
          infixEquation= infixEquation.substring(index+1);

      }else{
         this.newName= PerfExplorerExpression.getNewName(infixEquation);

      }
      if(PerfExplorerExpression.validate(infixEquation)){
          
      
      try {
          this.equation = infixToPostfix(infixEquation);
          checkMetrics(equation);
         
      } catch (Exception e) {
          // TODO Auto-generated catch block
          e.printStackTrace();
      }
      }else{
          System.err.println("\n\n *** ERROR: Equation is not correct: " + infixEquation + " ***\n\n");
          correctEquation = false;
      }
   }
//
//   if(!(firstMetric.equals("CALLS")||firstMetric.equals("SUBROUTINES")||secondMetric.equals("CALLS")||secondMetric.equals("SUBROUTINES"))){
//      if (!(input.getMetrics().contains(firstMetric)))
//         System.err.println("\n\n *** ERROR: Trial does not have a metric named: " + firstMetric + " ***\n\n");
//      if (!(input.getMetrics().contains(secondMetric)))
//         System.err.println("\n\n *** ERROR: Trial does not have a metric named: " + secondMetric + " ***\n\n");
//      }

   private void checkMetrics(ArrayList<String> equation2) {
       for(String metric:equation2){
      if(!isOperation(metric)&&!isValue(metric)){
          if(!(metric.equals("CALLS")||metric.equals("SUBROUTINES"))){
         if (!(input.getMetrics().contains(metric))){
             System.err.println("\n\n *** ERROR: Trial does not have a metric named: " + metric + " ***\n\n");
             correctEquation = false;
         }
          }
      }
       }
   }

   public DeriveMetricEquation(PerformanceResult input, String equation,
         String newName) {
      this(input, equation);
      this.newName = newName;
   }

   public DeriveMetricEquation(PerformanceResult input, String[] infixEquation) {
      super(input);
      this.input = input;
      this.equation = infixToPostfix(infixEquation);
   }
public boolean noErrors(){
    return correctEquation;
}

   /**
    * Convert the infix equation to postfix, using Dijkstra`s Shunting
    * Algorithm, so the equation can be evaluated from left to right following
    * the order of operation including parenthesis.
    * 
    * @param input
    * @return
    * @throws ParsingException 
    */
   private static ArrayList<String> infixToPostfix(String input) throws Exception {
      ArrayList<String> out = new ArrayList<String> ();
      String name = "";
      Stack<Character>  stack = new Stack<Character>();
      char[] in = input.toCharArray();
      for (int i=0;i<in.length;i++) {
         char current = in[i];
         switch (current) {
         case'\"':
            //Skip over anything in quotes
            i++;
            while(in[i]!='\"'){ 
               name +=in[i];
               i++;
            }
            break;
         case '+':
            if (!name.equals(""))
               out.add(name + "");   
            name = "";
            try {
               while (stack.peek().charValue() != '(')
                  out.add(stack.pop() + "");
            } catch (EmptyStackException ex) {}
            stack.push(new Character('+'));
            break;
         case '-':
            if (!name.equals(""))
               out.add(name + "");   
            name = "";
            try {
               while (stack.peek().charValue()!= '(')
                  out.add(stack.pop() + "");
            } catch (EmptyStackException ex) {}
            stack.push(new Character('-'));
            break;
         case '/':
            if (!name.equals(""))
               out.add(name + "");   
            name = "";
            try {
               while (stack.peek().charValue() != '(' && stack.peek().charValue() != '-' && stack.peek().charValue() != '+')
                  out.add(stack.pop() + "");
            } catch (EmptyStackException ex) {}
            stack.push(new Character('/'));
            break;
         case '*':
            if (!name.equals(""))
               out.add(name + "");
            name = "";
            try {
               while (stack.peek().charValue() != '(' && stack.peek().charValue() != '-' && stack.peek().charValue() != '+')
                  out.add(stack.pop() + "");
            } catch (EmptyStackException ex) {}
            stack.push(new Character('*'));
            break;
         case '(':
            stack.push(new Character('('));
            break;
         case ')':
            if (!name.equals(""))
               out.add(name + "");
            name = "";
            try {
               while (stack.peek().charValue() != '(')
                  out.add(stack.pop() + "");
               stack.pop();
            } catch (EmptyStackException ex) {
               throw new Exception ("Unmatched )");
            }
            break;
         case ' ':
            break;
         default:
            name += current;
            break;

         }

      }
      if (!name.equals(""))
         out.add(name + "");
      name = "";
      while (!stack.isEmpty())
         out.add(stack.pop() + "");

      return out;
   }
   
   private ArrayList<String> infixToPostfix(String[] input) {
      ArrayList<String> out = new ArrayList<String>();

      Stack<Character> stack = new Stack<Character>();

      for (String current : input) {
         char oper = current.charAt(0);
         switch (oper) {
         case '+':
            try {
               while (stack.peek() != '(')
                  out.add(stack.pop() + "");
            } catch (EmptyStackException ex) {

            }
            stack.push('+');
            break;
         case '-':
            try {
               while (stack.peek() != '(')
                  out.add(stack.pop() + "");
            } catch (EmptyStackException ex) {

            }
            stack.push('-');
            break;
         case '/':
            try {
               while (stack.peek() != '(' && stack.peek() != '-'
                     && stack.peek() != '+')
                  out.add(stack.pop() + "");
            } catch (EmptyStackException ex) {

            }
            stack.push('/');
            break;
         case '*':
            try {
               while (stack.peek() != '(' && stack.peek() != '-'
                     && stack.peek() != '+')
                  out.add(stack.pop() + "");
            } catch (EmptyStackException ex) {

            }
            stack.push('*');
            break;
         case '(':
            stack.push('(');
            break;
         case ')':
            try {
               while (stack.peek() != '(')
                  out.add(stack.pop() + "");
               stack.pop();
            } catch (EmptyStackException ex) {
               System.err.println("Unmatched )");
            }
            break;
         default:
            out.add(current);
            break;

         }

      }
      while (!stack.isEmpty())
         out.add(stack.pop() + "");

      return out;
   }

   private boolean isValue(String myString) {
      final String Digits = "(\\p{Digit}+)";
      final String HexDigits = "(\\p{XDigit}+)";

      final String Exp = "[eE][+-]?" + Digits;
      final String fpRegex = ("[\\x00-\\x20]*" + // Optional leading
            // "whitespace"
            "[+-]?(" + // Optional sign character
            "NaN|" + // "NaN" string
            "Infinity|" + // "Infinity" string

            // A decimal floating-point string representing a finite
            // positive
            // number without a leading sign has at most five basic pieces:
            // Digits . Digits ExponentPart FloatTypeSuffix
            // 
            // Since this method allows integer-only strings as input
            // in addition to strings of floating-point literals, the
            // two sub-patterns below are simplifications of the grammar
            // productions from the Java Language Specification, 2nd
            // edition, section 3.10.2.

            // Digits ._opt Digits_opt ExponentPart_opt FloatTypeSuffix_opt
            "(((" + Digits + "(\\.)?(" + Digits + "?)(" + Exp + ")?)|" +

      // . Digits ExponentPart_opt FloatTypeSuffix_opt
            "(\\.(" + Digits + ")(" + Exp + ")?)|" +

            // Hexadecimal strings
            "((" +
            // 0[xX] HexDigits ._opt BinaryExponent FloatTypeSuffix_opt
            "(0[xX]" + HexDigits + "(\\.)?)|" +

            // 0[xX] HexDigits_opt . HexDigits BinaryExponent
            // FloatTypeSuffix_opt
            "(0[xX]" + HexDigits + "?(\\.)" + HexDigits + ")" +

            ")[pP][+-]?" + Digits + "))" + "[fFdD]?))" + "[\\x00-\\x20]*");// Optional
      // trailing
      // "whitespace"

      return Pattern.matches(fpRegex, myString);
   }
   private static boolean isOperation(Object op) {
      if(op instanceof String){
         String oper = (String)op;
         char o = oper.charAt(0);
         return o == '+' || o == '-' || o == '*' || o == '/';
      }else{
         return false;
      }

   }
   private static double apply(char op, double arg1, double arg2) {
      double d = 0.0;
      switch (op) {
      case ('+'):
         d = arg1 + arg2;
      break;
      case ('-'):
         if (arg1 > arg2) {
            d = arg1 - arg2;
         }
      break;
      case ('*'):
         d = arg1 * arg2;
      break;
      case ('/'):
         if (arg2 != 0) {
            return arg1 / arg2;
         }
      break;
      default:
         //throw new PerfExplorerException("Unexpected operation type: " + op);
      }
      return d;
   }
   private double[] eval(ArrayList<Object> equation) {
      int i = 0;
      
      while (equation.size() > 1 && equation.size() > i) {

         if (isOperation(equation.get(i))) {
            try{
               char oper = ((String)equation.remove(i)).trim().charAt(0);
                double[] second = (double[])equation.remove(i - 1);
                double[] first = (double[]) equation.remove(i - 2);
               i = i - 2;
               
               double x = apply(oper,first[0],second[0]);
               double y = apply(oper,first[1],second[1]);
               double[] current = {x, y};
               equation.add(i, current);
            }catch(java.lang.ArrayIndexOutOfBoundsException ex){
               throw ex;
            }
         }
         i++;
      }
      return (double[])equation.get(0);
      
   }

   public List<PerformanceResult> processData() {
      if(newName != null) newName = newName.trim();
      for(int x=0;x<equation.size();x++){
           String current = (String)equation.get(x);
           if(isOperation(current)){
              
           }else if (isValue(current)){
        
           }else{
              input.getMetrics().contains(current);
              
           }
        }
      
      for (PerformanceResult input : inputs) {
         PerformanceResult output = new DefaultResult(input, false);
         
         for (String event : input.getEvents()) {
            for (Integer thread : input.getThreads()) {
               
               ArrayList<Object> newEquation = new ArrayList<Object>();
               for(int x=0;x<equation.size();x++){
                    String current = (String)equation.get(x);
                    if(isOperation(current)){
                       newEquation.add(current);
                    }else if (isValue(current)){
                       double[] array = {Double.valueOf(current),Double.valueOf(current)};
                       newEquation.add(array);
                    }else{
                       String metric = current;
                     double[] values = getValue(metric,thread,event);
                       newEquation.add(values);
                    }
                 }
               double[] result = eval(newEquation);
      
               output.putInclusive(thread, event, newName, result[0]);
               output.putExclusive(thread, event, newName, result[1]);
               output.putCalls(thread, event, input.getCalls(thread, event));
               output.putSubroutines(thread, event, input.getSubroutines(thread, event));
            }
         }
         outputs.add(output);
      }
      return outputs;
   }
   private double[] getValue(String metric, Integer thread, String event) {
      double[] a = new double[2];
      if(metric.equals("CALLS")){
         a[0] = input.getCalls(thread, event);       
         a[1]  = input.getCalls(thread, event);
      }else if(metric.equals("SUBROUTINES")){
         a[0]  = input.getSubroutines(thread, event);       
         a[1]  = input.getSubroutines(thread, event);   
      }else{
         a[0]  = input.getInclusive(thread, event, metric);          
         a[1]  = input.getExclusive(thread, event, metric);
      }
      return a;
   }

//   /*
//    * (non-Javadoc)
//    * 
//    * @see glue.PerformanceAnalysisOperation#processData()
//    */
//   public List<PerformanceResult> processDataOLD() {
//      int i = 0;
//      PerformanceResult merged = null;
//
//      while (equation.size() > 3 && equation.size() > i) {
//
//         if (isOperation(equation.get(i))) {
//
//            String oper = getOp(equation.remove(i));
//            String second = equation.remove(i - 1);
//            String first = equation.remove(i - 2);
//
//            i = i - 2;
//            PerformanceResult derived;
//            if (isValue(first)) {
//               double value = Double.valueOf(first);
//               ScaleMetricOperation scaler = new ScaleMetricOperation(
//                     input, value, second, oper);
//               derived = scaler.processData().get(0);
//            } else if (isValue(second)) {
//               double value = Double.valueOf(second);
//               ScaleMetricOperation scaler = new ScaleMetricOperation(
//                     input, first, value, oper);
//               derived = scaler.processData().get(0);
//            } else {
//               DeriveMetricOperation derivor = new DeriveMetricOperation(
//                     input, first, second, oper);
//               derived = derivor.processData().get(0);
//            }
//
//            String currentName = (String) derived.getMetrics().toArray()[0];
//            equation.add(i, currentName);
//            MergeTrialsOperation merger = new MergeTrialsOperation(input);
//            merger.addInput(derived);
//            merged = merger.processData().get(0);
//            input = merged;
//
//         }
//         i++;
//      }
//
//      String oper = getOp(equation.remove(2));
//      String second = equation.remove(1);
//      String first = equation.remove(0);
//
//      PerformanceResult derived;
//      if (isValue(first)) {
//         double value = Double.valueOf(first);
//         ScaleMetricOperation scaler = new ScaleMetricOperation(input,
//               value, second, oper);
//         if (newName != null)
//            scaler.setNewName(newName);
//         derived = scaler.processData().get(0);
//      } else if (isValue(second)) {
//         double value = Double.valueOf(second);
//         ScaleMetricOperation scaler = new ScaleMetricOperation(input,
//               first, value, oper);
//         if (newName != null)
//            scaler.setNewName(newName);
//         derived = scaler.processData().get(0);
//      } else {
//         DeriveMetricOperation derivor = new DeriveMetricOperation(input,
//               first, second, oper);
//         if (newName != null)
//            derivor.setNewName(newName);
//         derived = derivor.processData().get(0);
//      }
//
//      MergeTrialsOperation merger = new MergeTrialsOperation(input);
//      merger.addInput(derived);
//      merged = merger.processData().get(0);
//      input = merged;
//
//      if (merged == null) {
//         System.err.println("\n\n *** ERROR: Invaild Equation  ***\n\n");
//      } else {
//         outputs.add(merged);
//      }
//      return outputs;
//   }

//   private String getOp(String op) {
//      char o = op.charAt(0);
//
//      switch (o) {
//      case '+':
//         return DeriveMetricOperation.ADD;
//      case '-':
//         return DeriveMetricOperation.SUBTRACT;
//      case '*':
//         return DeriveMetricOperation.MULTIPLY;
//      case '/':
//         return DeriveMetricOperation.DIVIDE;
//      }
//      return null;
//   }

   private boolean isOperation(String op) {
      char o = op.charAt(0);
      return o == '+' || o == '-' || o == '*' || o == '/';
   }

   /**
    * @return the newName
    */
   public String getNewName() {
      return newName;
   }

   /**
    * @param newName
    *            the newName to set
    */
   public void setNewName(String newName) {
      this.newName = newName;
   }

   /**
    * Check if the derived metric already exists
    * 
    */
   public boolean exists() {
      return (inputs.get(0).getMetrics().contains(newName));
   }
}
