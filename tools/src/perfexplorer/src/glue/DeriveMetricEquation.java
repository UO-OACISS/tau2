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
import edu.uoregon.tau.perfexplorer.glue.AbstractPerformanceOperation;
import edu.uoregon.tau.perfexplorer.glue.DeriveMetricOperation;
import edu.uoregon.tau.perfexplorer.glue.MergeTrialsOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;

/**
 * @author smillst
 * 
 */
public class DeriveMetricEquation extends AbstractPerformanceOperation {
	private ArrayList<String> equation = null;
	private PerformanceResult input = null;
	private String newName = null;

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
		this.equation = infixToPostfix(infixEquation);
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

	/**
	 * Convert the infix equation to postfix, using Dijkstra`s Shunting
	 * Algorithm, so the equation can be evaluated from left to right.
	 * 
	 * @param input
	 * @return
	 */
	private ArrayList<String> infixToPostfix(String input) {
		ArrayList<String> out = new ArrayList<String>();
		String name = "";
		Stack<Character> stack = new Stack<Character>();
		char[] in = input.toCharArray();
		for (char current : in) {
			switch (current) {
			case '+':
				if (!name.equals(""))
					out.add(name + "");
				name = "";
				try {
					while (stack.peek() != '(')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {

				}
				stack.push('+');
				break;
			case '-':
				if (!name.equals(""))
					out.add(name + "");
				name = "";

				try {
					while (stack.peek() != '(')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {

				}
				stack.push('-');
				break;
			case '/':
				if (!name.equals(""))
					out.add(name + "");
				name = "";
				try {
					while (stack.peek() != '(' && stack.peek() != '-'
							&& stack.peek() != '+')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {

				}
				stack.push('/');
				break;
			case '*':
				if (!name.equals(""))
					out.add(name + "");
				name = "";
				try {
					while (stack.peek() != '(' && stack.peek() != '-'
							&& stack.peek() != '+')
						out.add(stack.pop() + "");
				} catch (EmptyStackException ex) {

				}
				stack.push('*');
				break;
			case '(':
				if (!name.equals(""))
					out.add(name + "");
				name = "";
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
					System.err.println("Unmatched )");
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

	/*
	 * (non-Javadoc)
	 * 
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		int i = 0;
		PerformanceResult merged = null;

		while (equation.size() > 3 && equation.size() > i) {

			if (isOperation(equation.get(i))) {

				String oper = getOp(equation.remove(i));
				String second = equation.remove(i - 1);
				String first = equation.remove(i - 2);

				i = i - 2;
				PerformanceResult derived;
				if (isValue(first)) {
					double value = Double.valueOf(first);
					ScaleMetricOperation scaler = new ScaleMetricOperation(
							input, value, second, oper);
					derived = scaler.processData().get(0);
				} else if (isValue(second)) {
					double value = Double.valueOf(second);
					ScaleMetricOperation scaler = new ScaleMetricOperation(
							input, first, value, oper);
					derived = scaler.processData().get(0);
				} else {
					DeriveMetricOperation derivor = new DeriveMetricOperation(
							input, first, second, oper);
					derived = derivor.processData().get(0);
				}

				String currentName = (String) derived.getMetrics().toArray()[0];
				equation.add(i, currentName);
				MergeTrialsOperation merger = new MergeTrialsOperation(input);
				merger.addInput(derived);
				merged = merger.processData().get(0);
				input = merged;

			}
			i++;
		}

		String oper = getOp(equation.remove(2));
		String second = equation.remove(1);
		String first = equation.remove(0);

		PerformanceResult derived;
		if (isValue(first)) {
			double value = Double.valueOf(first);
			ScaleMetricOperation scaler = new ScaleMetricOperation(input,
					value, second, oper);
			if (newName != null)
				scaler.setNewName(newName);
			derived = scaler.processData().get(0);
		} else if (isValue(second)) {
			double value = Double.valueOf(second);
			ScaleMetricOperation scaler = new ScaleMetricOperation(input,
					first, value, oper);
			if (newName != null)
				scaler.setNewName(newName);
			derived = scaler.processData().get(0);
		} else {
			DeriveMetricOperation derivor = new DeriveMetricOperation(input,
					first, second, oper);
			if (newName != null)
				derivor.setNewName(newName);
			derived = derivor.processData().get(0);
		}

		MergeTrialsOperation merger = new MergeTrialsOperation(input);
		merger.addInput(derived);
		merged = merger.processData().get(0);
		input = merged;

		if (merged == null) {
			System.err.println("\n\n *** ERROR: Invaild Equation  ***\n\n");
		} else {
			outputs.add(merged);
		}
		return outputs;
	}

	private String getOp(String op) {
		char o = op.charAt(0);

		switch (o) {
		case '+':
			return DeriveMetricOperation.ADD;
		case '-':
			return DeriveMetricOperation.SUBTRACT;
		case '*':
			return DeriveMetricOperation.MULTIPLY;
		case '/':
			return DeriveMetricOperation.DIVIDE;
		}
		return null;
	}

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
