package edu.uoregon.tau.perfexplorer.client;




import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.Scanner;
import java.util.Stack;

/**
 * 
 * @author smillst
 *
 */

public abstract class Expression {

	public static void main(String args[]) throws ParsingException{
		//System.out.println("\"a=b\"=a+\"b+c\"- 12 / c");
		//System.out.println("\"a++100\"+\"b+_110\"");
		System.out.println(Expression.validate("\"a++100\"+\"b+_110\""));
		System.out.println(Expression.validate("(a+b)*c"));
		System.out.println(Expression.validate("(a+b+b)*(a+b)/(34+\"b-a:\")"));
		System.out.println(Expression.validate("(a-b+c.d*f)-ab*(ab*c)/2-(3-4)*(34+a"));
		System.out.println(Expression.validate("(a+b+b)*(a+b)/(34+\"b-a:\")"));
		System.out.println(Expression.validate("(a+b+b)*(a+b)/(34+\"b-a:\")"));

	}
/**
 * Parses a single expression.  
 * Note: if any metrics used in the equation contain operators (+,-,/,*),
 * then they should be surrounded in quotes so they are processed correctly.
 * @param expression
 * @return
 * @throws ParsingException
 */
	public   String parse(String expression) throws ParsingException{
		char[] array = expression.toCharArray();
		//check for =
		for(int i=0; i<array.length;i++){
			//skip sections in quotes
			if(array[i]=='\"'){
				i++;
				while(array[i]!='\"') i++;
			}else if(array[i]=='='){
				//This expression is of the form: newName = expression
				return evaluateExpression(expression.substring(0, i),infixToPostfix(expression.substring(i+1)));
			}
		}
		//This expression is just an expression
		return evaluateExpression(null,infixToPostfix(expression));
	}

	public  String parseExpressions(String expressions) throws ParsingException{
		return parseMany(new Scanner(expressions));
	}
	public  String parseFile(String file) throws ParsingException, FileNotFoundException{
		Scanner scan = new Scanner(new File(file));
		return parseMany(scan);
	}
	private  String parseMany(Scanner scan) throws ParsingException{
		String last = "";
		while(scan.hasNextLine()){
			String line = scan.nextLine();
			line.trim();
			if(!line.equals("")){	
				last = parse(line.trim());
			}
		}
		return last;

	}
	public static boolean validate(String expression){
		try {
			new TestExpression().parse(expression);
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
	private static ArrayList<String> infixToPostfix(String input) throws ParsingException {
		ArrayList<String> out = new ArrayList<String>();
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

	private  String evaluateExpression(String newName, ArrayList<String> equation) throws ParsingException {
if(newName != null) newName = newName.trim();
		int i = 0;
		if(equation.size()==1){
			return rename(newName, equation.get(0));
		}

		while (equation.size() > 1 && equation.size() > i) {

			if (isOperation(equation.get(i))) {
				try{
					char oper = equation.remove(i).trim().charAt(0);
					String second = equation.remove(i - 1);
					String first = equation.remove(i - 2);
					i = i - 2;

					String current;

					if(!(equation.size()<2 && newName != null)){
						if (isValue(first)) {
							double value = Double.valueOf(first);
							current = applyOperation(value, second, oper);
						} else if (isValue(second)) {
							double value = Double.valueOf(second);
							current = applyOperation(first, value, oper);				
						} else {
							current = applyOperation(first, second, oper);
						}
					}else{		

						if (isValue(first)) {
							double value = Double.valueOf(first);
							current = applyOperation(value, second, oper,newName);
						} else if (isValue(second)) {
							double value = Double.valueOf(second);
							current = applyOperation(first, value, oper, newName);				
						} else {
							current = applyOperation(first, second, oper, newName);
						}	
					}
					equation.add(i, current);
				}catch(java.lang.ArrayIndexOutOfBoundsException ex){
					throw new ParsingException();

				}

			}
			i++;
		}
		return equation.get(0);
	}



	private static boolean isOperation(String op) {
		char o = op.charAt(0);
		return o == '+' || o == '-' || o == '*' || o == '/';
	}
	protected abstract String rename(String newName, String metric);
	protected abstract String applyOperation(String operand1, String operand2, char operation, String newName);
	protected abstract String applyOperation(String operand1, double operand2, char operation, String newName);
	protected abstract String applyOperation(double operand1, String operand2, char operation, String newName);
	protected abstract String applyOperation(String operand1, String operand2, char operation);
	protected abstract String applyOperation(String operand1, double operand2, char operation);
	protected abstract String applyOperation(double operand1, String operand2, char operation);



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

class TestExpression extends Expression{

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

}


