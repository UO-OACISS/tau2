package edu.uoregon.tau.paraprof;
import java.io.CharArrayReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.Iterator;
import java.util.List;
import java.util.Stack;

import javax.swing.JOptionPane;


import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Thread;

/**
 * 
 * @author smillst
 *
 */


public class ParaProfExpression {

	/**
	 * Parses a single expression.  
	 * Note: if any metrics used in the equation contain operators (+,-,/,*),
	 * then they should be surrounded in quotes so they are processed correctly.
	 * @param expression
	 * @return
	 * @throws ParsingException
	 * @throws MetricNotFoundException 
	 */
	public  void evaluateExpression(String expression, List trials) throws ParsingException, MetricNotFoundException{
		char[] array = expression.toCharArray();
		String newName = null;
		//check for =
		for(int i=0; i<array.length;i++){
			//skip sections in quotes
			if(array[i]=='\"'){
				i++;
				while(array[i]!='\"') i++;
			}else if(array[i]=='='){
				//This expression is of the form: newName = expression
				newName =expression.substring(0, i);
				expression = expression.substring(i+1);
				break;
			}
		}
		if(newName ==null){
			newName =expression;
		}
		for(int i=0;i<trials.size();i++){
		ParaProfTrial trial= (ParaProfTrial)trials.get(i);
			ArrayList 	expressArray =  infixToPostfix(expression);
			evaluate(newName,trial,expressArray);
		}

	}
	
	public ParaProfMetric evaluateExpression(ParaProfTrial trial, String text) throws ParsingException, MetricNotFoundException {
		char[] array = text.toCharArray();
		String newName = null;
		//check for =
		for(int i=0; i<array.length;i++){
			//skip sections in quotes
			if(array[i]=='\"'){
				i++;
				while(array[i]!='\"') i++;
			}else if(array[i]=='='){
				//This expression is of the form: newName = expression
				newName =text.substring(0, i);
				text = text.substring(i+1);
				break;
			}
		}

		ArrayList 	expressArray =  infixToPostfix(text);
		if(newName ==null){
			text.trim();
			text = text.replace('\"', ' ');
			newName ="("+text+")";
		}

		return evaluate(newName,trial,expressArray);



	}


	public  String evaluateExpressions(String expressions, List trials) throws ParsingException, IOException, MetricNotFoundException{
		return evaluateMany(new LineNumberReader(new CharArrayReader(expressions.toCharArray())),trials);
	}
	public  String evaluateFile(String file, List trials) throws ParsingException, IOException, MetricNotFoundException{
		LineNumberReader scan = new LineNumberReader(new FileReader(new File(file)));
		return evaluateMany(scan,trials);
	}
	private  String evaluateMany(LineNumberReader scan, List trials) throws ParsingException, IOException, MetricNotFoundException{
		String line = scan.readLine();

		while(line !=null){
			
		   
			line.trim();
			if(!line.equals("")){	
				evaluateExpression(line.trim(),trials);
			}
			 line = scan.readLine();
		}
		return "";

	}
	/*public static boolean validate(String expression){
		try {
			new TestExpression().parse(expression);
			return true;
		}catch(java.lang.ArrayIndexOutOfBoundsException ex){
			return false;
		}catch (ParsingException e) {
			return false;
		}
	}*/
public static boolean validate(String expression){
	try {
		ArrayList equation = infixToPostfix(expression);
		if (equation.size()<2){
			return false;
			
		}
		 
		 int i = 0;
			
			while (equation.size() > 1 && equation.size() > i) {

				if (isOperation(equation.get(i))) {
					try{
						Object oper = equation.remove(i);
						 Object second = equation.remove(i - 1);
						 Object first = equation.remove(i - 2);
						i = i - 2;
						
						equation.add(i, "Intermedate");
					}catch(java.lang.ArrayIndexOutOfBoundsException ex){
						throw new ParsingException();
					}
				}
				i++;
			}
		return true;
	}catch(java.lang.ArrayIndexOutOfBoundsException ex){
		return false;
	}catch (ParsingException e) {
		return false;
	}
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
	private static ArrayList infixToPostfix(String input) throws ParsingException {
		ArrayList out = new ArrayList();
		String name = "";
		Stack stack = new Stack();
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
					while (((Character)stack.peek()).charValue() != '(')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {}
				stack.push(new Character('+'));
				break;
			case '-':
				if (!name.equals(""))
					out.add(name + "");	
				name = "";
				try {
					while (((Character)stack.peek()).charValue()!= '(')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {}
				stack.push(new Character('-'));
				break;
			case '/':
				if (!name.equals(""))
					out.add(name + "");	
				name = "";
				try {
					while (((Character)stack.peek()).charValue() != '(' && ((Character)stack.peek()).charValue() != '-' && ((Character)stack.peek()).charValue() != '+')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {}
				stack.push(new Character('/'));
				break;
			case '*':
				if (!name.equals(""))
					out.add(name + "");
				name = "";
				try {
					while (((Character)stack.peek()).charValue() != '(' && ((Character)stack.peek()).charValue() != '-' && ((Character)stack.peek()).charValue() != '+')
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
					while (((Character)stack.peek()).charValue() != '(')
						out.add(stack.pop() + "");
					stack.pop();
				} catch (EmptyStackException ex) {
					throw new ParsingException ("Unmatched )");
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

	private static boolean isValue(String myString) {
		try{
			Double.valueOf(myString);
			return true;
		}catch (NumberFormatException ex){
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
			throw new ParaProfException("Unexpected operation type: " + op);
		}
		return d;
	}

	private ParaProfMetric findMetric(Object first, ParaProfTrial trial) throws  MetricNotFoundException {
		if(first instanceof ParaProfMetric) return (ParaProfMetric) first;
		

		int id =trial.getMetricID((String)first);
		if(id<0) throw new MetricNotFoundException("Metric \""+first+"\" was not found");
		return trial.getMetric(id);
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
	
	protected ParaProfMetric rename(String newName, ParaProfMetric metric) {
		metric.setName(newName);
		return metric;
	}
	private static void sleep(int msec) {
		try {
			java.lang.Thread.sleep(msec);
		} catch (Exception e) {
			throw new RuntimeException("Exception while sleeping");
		}
	}
	private ParaProfMetric evaluate(String newName,ParaProfTrial trial,ArrayList equation) throws ParsingException, MetricNotFoundException{
		if(trial==null) throw new ParsingException("Null trial");
		if(newName != null) newName = newName.trim();
		if(equation.size()==1){
			return rename(newName, findMetric(equation.get(0),trial));
		}
		while (trial.loading()) {
			sleep(500);
		}
		
  		for(int x=0;x<equation.size();x++){
  			String current = (String)equation.get(x);
  			if(isOperation(current)){
  				//Do nothing
  			}else if (isValue(current)){
  				Double[] array = {Double.valueOf(current),Double.valueOf(current)};
  				equation.remove(x);
  				equation.add(x, array);
  			}else{
  				
  					ParaProfMetric xMetric = findMetric(current,trial);
  					equation.remove(x);
  					equation.add(x,xMetric);
  				
  			}
  		}
        ParaProfMetric newMetric = trial.addMetric();
        newMetric.setPpTrial(trial);
        newMetric.setName(newName);
        newMetric.setDerivedMetric(true);
        int metric = newMetric.getID();
        //            trialOpA.setSelectedMetricID(metric);

        Iterator l = trial.getDataSource().getFunctions();

        for (Iterator it = trial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            Thread thread = (Thread) it.next();
            thread.addMetric();
            l = thread.getFunctionProfileIterator();
            while (l.hasNext()) {
                FunctionProfile functionProfile = (FunctionProfile) l.next();
                
                if (functionProfile != null) {
                    ArrayList newEquation = new ArrayList();
             		for(int x=0;x<equation.size();x++){
             			Object current = equation.get(x);
             			if(current instanceof ParaProfMetric){
             					ParaProfMetric xMetric = (ParaProfMetric )current;
             					double ex =functionProfile.getExclusive(xMetric.getID());
             					double in =functionProfile.getInclusive(xMetric.getID());
             					Double[] array = {new Double(in),new Double(ex)};
             					newEquation.add(array);
             			}else{
             				newEquation.add(current);
             			}
             		}
            		Double[] result = eval(newEquation);
                    functionProfile.setInclusive(metric, result[0].doubleValue());
                    functionProfile.setExclusive(metric, result[1].doubleValue());
                }
            }
            thread.setThreadData(metric);
        }

        //Done with this metric, compute the mean values.
        trial.setMeanData(metric);

        trial.getDataSource().getMeanData().setThreadData(metric);
        trial.getDataSource().getTotalData().setThreadData(metric);
        trial.getDataSource().getStdDevData().setThreadData(metric);
        

        return newMetric;
	}
	private Double[] eval(ArrayList equation) throws ParsingException{
		int i = 0;
		
		while (equation.size() > 1 && equation.size() > i) {

			if (isOperation(equation.get(i))) {
				try{
					char oper = ((String)equation.remove(i)).trim().charAt(0);
					 Double[] second = (Double[])equation.remove(i - 1);
					 Double[] first = (Double[]) equation.remove(i - 2);
					i = i - 2;
					
					double x = apply(oper,first[0].doubleValue(),second[0].doubleValue());
					double y = apply(oper,first[1].doubleValue(),second[1].doubleValue());
					Double[] current = {new Double(x), new Double(y)};
					equation.add(i, current);
				}catch(java.lang.ArrayIndexOutOfBoundsException ex){
					throw new ParsingException();
				}
			}
			i++;
		}
		return (Double[])equation.get(0);
	}


	


}
class ParsingException extends Exception{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public ParsingException() {
		super();
		// TODO Auto-generated constructor stub
	}

	public ParsingException(String message, Throwable cause) {
		super(message, cause);
		// TODO Auto-generated constructor stub
	}

	public ParsingException(String message) {
		super(message);
		// TODO Auto-generated constructor stub
	}

	public ParsingException(Throwable cause) {
		super(cause);
		// TODO Auto-generated constructor stub
	}
}
class MetricNotFoundException extends Exception{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * 
	 */
	public MetricNotFoundException(){
		super();
	}

	public MetricNotFoundException(String string) {
		// TODO Auto-generated constructor stub
	}
	
}






