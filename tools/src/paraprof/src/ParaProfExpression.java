package edu.uoregon.tau.paraprof;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
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
	 */
	public  void evaluateExpression(String expression, List<ParaProfTrial> trials) throws ParsingException{
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
		
		for(ParaProfTrial trial : trials){
			ArrayList<Object> 	expressArray =  infixToPostfix(expression);
			evaluateExpression(newName,expressArray,trial);
		}

	}

	public  String evaluateExpressions(String expressions, List<ParaProfTrial> trials) throws ParsingException{
		return evaluateMany(new Scanner(expressions),trials);
	}
	public  String evaluateFile(String file, List<ParaProfTrial> trials) throws ParsingException, FileNotFoundException{
		Scanner scan = new Scanner(new File(file));
		return evaluateMany(scan,trials);
	}
	private  String evaluateMany(Scanner scan, List<ParaProfTrial> trials) throws ParsingException{
		String last = "";
		while(scan.hasNextLine()){
			String line = scan.nextLine();
			line.trim();
			if(!line.equals("")){	
				evaluateExpression(line.trim(),trials);
			}
		}
		return last;

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


	/**
	 * Convert the infix equation to postfix, using Dijkstra`s Shunting
	 * Algorithm, so the equation can be evaluated from left to right following
	 * the order of operation including parenthesis.
	 * 
	 * @param input
	 * @return
	 * @throws ParsingException 
	 */
	private static ArrayList<Object> infixToPostfix(String input) throws ParsingException {
		ArrayList<Object> out = new ArrayList<Object>();
		String name = "";
		Stack<Character> stack = new Stack<Character>();
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
					while (stack.peek() != '(')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {}
				stack.push('+');
				break;
			case '-':
				if (!name.equals(""))
					out.add(name + "");	
				name = "";
				try {
					while (stack.peek() != '(')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {}
				stack.push('-');
				break;
			case '/':
				if (!name.equals(""))
					out.add(name + "");	
				name = "";
				try {
					while (stack.peek() != '(' && stack.peek() != '-' && stack.peek() != '+')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {}
				stack.push('/');
				break;
			case '*':
				if (!name.equals(""))
					out.add(name + "");
				name = "";
				try {
					while (stack.peek() != '(' && stack.peek() != '-' && stack.peek() != '+')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {}
				stack.push('*');
				break;
			case '(':
				stack.push('(');
				break;
			case ')':
				if (!name.equals(""))
					out.add(name + "");
				name = "";
				try {
					while (stack.peek() != '(')
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

	private  ParaProfMetric evaluateExpression(String newName, ArrayList<Object> equation,ParaProfTrial trial) throws ParsingException {
		if(newName != null) newName = newName.trim();
		int i = 0;
		if(equation.size()==1){
			return rename(newName, findMetric(equation.get(0),trial));
		}

		while (equation.size() > 1 && equation.size() > i) {

			if (isOperation(equation.get(i))) {
				try{
					char oper = ((String)equation.remove(i)).trim().charAt(0);
					Object second = equation.remove(i - 1);
					Object first = equation.remove(i - 2);
					i = i - 2;

					ParaProfMetric current;
					if(first instanceof String){
						String firstS = (String) first;
						if (isValue(firstS)) {
							first = Double.valueOf(firstS);
						} else {
							first = findMetric(first,trial);
						}
					}
					if(second instanceof String){
						String secondS = (String) second;
						if (isValue(secondS)) {
							second = Double.valueOf(secondS);
						} else {
							second = findMetric(second,trial);
						}

					}


					if(!(equation.size()<2 && newName != null)){
						current = applyOperation(first,second,oper);
					}else{		
						current = applyOperation(first, second, oper,newName);
					}
					equation.add(i, current);
				}catch(java.lang.ArrayIndexOutOfBoundsException ex){
					throw new ParsingException();

				}

			}
			i++;
		}
		return (ParaProfMetric)equation.get(0);
	}

	public static double apply(char op, double arg1, double arg2) {
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

	private ParaProfMetric findMetric(Object first, ParaProfTrial trial) {
		if(first instanceof ParaProfMetric) return (ParaProfMetric) first;
		int id =trial.getMetricID((String)first);
		ParaProfTrial p =  trial.getMetric(id).getParaProfTrial();
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
	protected ParaProfMetric applyOperation(Object operand1, Object operand2, char operation) {
		if(operand1 instanceof ParaProfMetric){
			ParaProfMetric left = (ParaProfMetric)operand1;
			if(operand2 instanceof ParaProfMetric){
				ParaProfMetric right = (ParaProfMetric)operand2;
				return	applyOperation(left,right,operation);
			}else{
				Double right = (Double)operand2;
				return	applyOperation(left,right,operation);
			}
		}else{
			Double left = (Double)operand1;
			if(operand2 instanceof ParaProfMetric){
				ParaProfMetric right = (ParaProfMetric)operand2;
				return	applyOperation(left,right,operation);
			}else{
				Double right = (Double)operand2;
				return	applyOperation(left,right,operation);
			}
		}

	}
	protected ParaProfMetric applyOperation(Object operand1, Object operand2, char operation, String newName) {
		if(operand1 instanceof ParaProfMetric){
			ParaProfMetric left = (ParaProfMetric)operand1;
			if(operand2 instanceof ParaProfMetric){
				ParaProfMetric right = (ParaProfMetric)operand2;
				return	applyOperation(left,right,operation, newName);
			}else{
				Double right = (Double)operand2;
				return	applyOperation(left,right,operation, newName);
			}
		}else{
			Double left = (Double)operand1;
			if(operand2 instanceof ParaProfMetric){
				ParaProfMetric right = (ParaProfMetric)operand2;
				return	applyOperation(left,right,operation, newName);
			}else{
				Double right = (Double)operand2;
				return	applyOperation(left,right,operation, newName);
			}
		}

	}
	protected ParaProfMetric rename(String newName, ParaProfMetric metric) {
		return applyOperation(metric,0,'+',newName);
	}
	protected ParaProfMetric applyOperation(ParaProfMetric operand1, ParaProfMetric operand2, char operation, String newName) {

		try {


			ParaProfTrial trialOpA = operand1.getParaProfTrial();
			int opA = operand1.getID();
			ParaProfTrial trialOpB = operand2.getParaProfTrial();
			int  opB =  operand2.getID();

			//We do not support metric from different trials yet. Check for this.
			if  (trialOpA != trialOpB) {
				JOptionPane.showMessageDialog(ParaProf.paraProfManagerWindow,
						"Sorry, please select metrics from the same trial!", "ParaProf Error", JOptionPane.ERROR_MESSAGE);
				return null;
			}
			if(newName ==null)
				newName = ((ParaProfMetric) trialOpA.getMetrics().get(opA)).getName() + " "+operation+" "
				+ ((ParaProfMetric) trialOpA.getMetrics().get(opB)).getName();
			ParaProfMetric newMetric = trialOpA.addMetric();
			newMetric.setPpTrial(trialOpA);
			newMetric.setName(newName);
			newMetric.setDerivedMetric(true);

			int metric = newMetric.getID();
			//            trialOpA.setSelectedMetricID(metric);

			Iterator l = trialOpA.getDataSource().getFunctions();

			//######
			//Calculate the raw values.
			//We only need establish exclusive and inclusive time.
			//The rest of the data can either be computed from these,
			//or is already in the system (number of calls as an example
			//of the latter.
			//######

			for (Iterator it = trialOpA.getDataSource().getAllThreads().iterator(); it.hasNext();) {
				Thread thread = (Thread) it.next();
				thread.addMetric();
				l = thread.getFunctionProfileIterator();
				while (l.hasNext()) {
					FunctionProfile functionProfile = (FunctionProfile) l.next();
					if (functionProfile != null) {
						Function function = functionProfile.getFunction();

						double d1 = 0.0;
						double d2 = 0.0;
						double result = 0.0;

						d1 = functionProfile.getExclusive(opA);

						d2 = functionProfile.getExclusive(opB);
						result = apply(operation, d1, d2);

						functionProfile.setExclusive(metric, result);

						d1 = functionProfile.getInclusive(opA);

						d2 = functionProfile.getInclusive(opB);
						result =  apply(operation, d1, d2);

						functionProfile.setInclusive(metric, result);

					}
				}
				thread.setThreadData(metric);
			}

			//Done with this metric, compute the mean values.
			trialOpA.setMeanData(metric);

			trialOpA.getDataSource().getMeanData().setThreadData(metric);
			trialOpA.getDataSource().getTotalData().setThreadData(metric);
			trialOpA.getDataSource().getStdDevData().setThreadData(metric);

			return newMetric;
		} catch (NumberFormatException e) {
			//Display an error
			JOptionPane.showMessageDialog(ParaProf.paraProfManagerWindow, "Did not recognize arguments! ", "Argument Error!",
					JOptionPane.ERROR_MESSAGE);
			return null;
		}
	}
	protected ParaProfMetric applyOperation(ParaProfMetric operand1, Double operand2, char operation, String newName) {

		try {

			double constantValue = operand2;
			ParaProfTrial trialOpA = operand1.getParaProfTrial();
			int opA = operand1.getID();

			if(newName == null)
				newName = ((ParaProfMetric) trialOpA.getMetrics().get(opA)).getName() + " "+operation+" " + constantValue;

			ParaProfMetric newMetric = trialOpA.addMetric();
			newMetric.setPpTrial(trialOpA);
			newMetric.setName(newName);
			newMetric.setDerivedMetric(true);

			int metric = newMetric.getID();
			//            trialOpA.setSelectedMetricID(metric);

			Iterator l = trialOpA.getDataSource().getFunctions();

			//######
			//Calculate the raw values.
			//We only need establish exclusive and inclusive time.
			//The rest of the data can either be computed from these,
			//or is already in the system (number of calls as an example
			//of the latter.
			//######

			for (Iterator it = trialOpA.getDataSource().getAllThreads().iterator(); it.hasNext();) {
				Thread thread = (Thread) it.next();
				thread.addMetric();
				l = thread.getFunctionProfileIterator();
				while (l.hasNext()) {
					FunctionProfile functionProfile = (FunctionProfile) l.next();
					if (functionProfile != null) {
						Function function = functionProfile.getFunction();

						double d1 = 0.0;
						double d2 = 0.0;
						double result = 0.0;

						d1 = functionProfile.getExclusive(opA);
						result =  apply(operation, d1, constantValue);
						functionProfile.setExclusive(metric, result);

						d1 = functionProfile.getInclusive(opA);
						result =  apply(operation, d1, constantValue);
						functionProfile.setInclusive(metric, result);

					}
				}
				thread.setThreadData(metric);
			}

			//Done with this metric, compute the mean values.
			trialOpA.setMeanData(metric);

			trialOpA.getDataSource().getMeanData().setThreadData(metric);
			trialOpA.getDataSource().getTotalData().setThreadData(metric);
			trialOpA.getDataSource().getStdDevData().setThreadData(metric);

			return newMetric;

		} catch (NumberFormatException e) {
			//Display an error
			JOptionPane.showMessageDialog(ParaProf.paraProfManagerWindow, "Did not recognize arguments! ", "Argument Error!",
					JOptionPane.ERROR_MESSAGE);
			return null;
		}
	}
	protected ParaProfMetric applyOperation(Double operand1, ParaProfMetric operand2, char operation, String newName) {

		try {

			double constantValue = operand1;
			ParaProfTrial trialOpB = operand2.getParaProfTrial();
			int opB = operand2.getID();

			if(newName == null)
				newName = ((ParaProfMetric) trialOpB.getMetrics().get(opB)).getName() + " "+operation+" " + constantValue;

			ParaProfMetric newMetric = trialOpB.addMetric();
			newMetric.setPpTrial(trialOpB);
			newMetric.setName(newName);
			newMetric.setDerivedMetric(true);

			int metric = newMetric.getID();
			//            trialOpA.setSelectedMetricID(metric);

			Iterator l = trialOpB.getDataSource().getFunctions();

			//######
			//Calculate the raw values.
			//We only need establish exclusive and inclusive time.
			//The rest of the data can either be computed from these,
			//or is already in the system (number of calls as an example
			//of the latter.
			//######

			for (Iterator it = trialOpB.getDataSource().getAllThreads().iterator(); it.hasNext();) {
				Thread thread = (Thread) it.next();
				thread.addMetric();
				l = thread.getFunctionProfileIterator();
				while (l.hasNext()) {
					FunctionProfile functionProfile = (FunctionProfile) l.next();
					if (functionProfile != null) {
						Function function = functionProfile.getFunction();

						double d1 = 0.0;
						double result = 0.0;

						d1 = functionProfile.getExclusive(opB);
						result =  apply(operation, constantValue, d1);
						functionProfile.setExclusive(metric, result);

						d1 = functionProfile.getInclusive(opB);
						result =  apply(operation,constantValue, d1);
						functionProfile.setInclusive(metric, result);

					}
				}
				thread.setThreadData(metric);
			}

			//Done with this metric, compute the mean values.
			trialOpB.setMeanData(metric);

			trialOpB.getDataSource().getMeanData().setThreadData(metric);
			trialOpB.getDataSource().getTotalData().setThreadData(metric);
			trialOpB.getDataSource().getStdDevData().setThreadData(metric);

			return newMetric;

		} catch (NumberFormatException e) {
			//Display an error
			JOptionPane.showMessageDialog(ParaProf.paraProfManagerWindow, "Did not recognize arguments! ", "Argument Error!",
					JOptionPane.ERROR_MESSAGE);
			return null;
		}
	}
	protected ParaProfMetric applyOperation(ParaProfMetric operand1, ParaProfMetric operand2, char operation) {
		String oper = "";
		switch(operation){
		case '+':
			oper = "Add";
			break;
		case '-':
			oper = "Subtract";
			break;
		case '*':
			oper = "Multiply";
			break;
		case '/':
			oper = "Divide";
			break;
		}
		return DerivedMetrics.applyOperation(operand1, operand2, oper);
	}
	protected ParaProfMetric applyOperation(ParaProfMetric operand1, Double operand2, char operation) {
		return applyOperation(operand1,operand2,operation,null);

	}
	protected ParaProfMetric applyOperation(Double operand1, ParaProfMetric operand2, char operation) {
		return applyOperation(operand1,operand2,operation,null);

	}


}
class ParsingException extends Exception{

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

/*class TestExpression extends Expression{

	@Override
	protected String applyOperation(String operand1, String operand2,
			char operation, String newName) {
		return newName + "="+operand1+operation+operand2;
	}

	@Override
	protected String applyOperation(String operand1, double operand2,
			char operation, String newName) {
		return newName + "="+operand1+operation+operand2;
	}

	@Override
	protected String applyOperation(double operand1, String operand2,
			char operation, String newName) {
		return newName + "="+operand1+operation+operand2;
	}

	@Override
	protected String applyOperation(String operand1, String operand2,
			char operation) {
		return operand1+operation+operand2;
	}

	@Override
	protected String applyOperation(String operand1, double operand2,
			char operation) {
		return operand1+operation+operand2;
	}

	@Override
	protected String applyOperation(double operand1, String operand2,
			char operation) {
		return ""+operand1+operation+operand2;
	}

	@Override
	protected String rename(String newName, String metric) {
		return newName +"="+metric;
	}

}*/




