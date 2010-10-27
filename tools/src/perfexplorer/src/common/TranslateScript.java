package edu.uoregon.tau.perfexplorer.common;

import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

/**
 * 
 * @author smillst
 *
 */
public class TranslateScript {
	private static String spaces = "";


	public static String translate(String filename) throws EquationParseException {
		try {
			FileReader read = new FileReader(filename);

			String write = "";

			int in = read.read();
			while (in != -1) {
				char c = (char) in;
				if (c == '%') {

					if (write.charAt(write.length() - 1) == '#') {
						spaces = findSpaces(write);
						write = write.substring(0, write.length() - 1);
						write += translateEquation(read);
					} else {
						write += (c);
					}
				} else {
					write += (c);
				}

				in = read.read();
			}
			return write;
		} catch (IOException ex) {
			System.err.println("\n\n *** ERROR: Unknown file  ***\n\n");
			return "";
		}
	}

	private static String translateEquation(InputStreamReader read)
	throws IOException, EquationParseException {
		Equation equ = parseEquation(read);
		return writeCode(equ);

	}


	/**
	 * Pulls out the equation data from the file so it can be translated.
	 * Equations start with #% and stop at the end of a line.
	 * 
	 * @param read
	 * @return
	 * @throws IOException
	 * @throws IOException
	 * @throws EquationParseException 
	 */
	private static Equation parseEquation(InputStreamReader read)
			throws IOException, EquationParseException {
		char in = (char) read.read();
		String statement = "";
		String input = "";
		String var = "";
		String newName = "";
		
			// Read in the input var
			while (in != ',') {
				if (in == (char) -1 || EOL(in) || in =='=')
					throw new EquationParseException("\n\n *** ERROR: Invaild Equation  ***\n\n");
				input += in;
				in = (char) read.read();
			}
			in = (char) read.read();
	
			// Read in the newMetric var
			while (in != '=') {
				if (in == (char) -1 || EOL(in))
					throw new EquationParseException("\n\n *** ERROR: Invaild Equation  ***\n\n");
				var += in;
				in = (char) read.read();
			}
			
			in = (char) read.read();
			// Read in the equation
			while (!EOL(in)) {
				
				if (in == (char) -1)
					throw new EquationParseException("\n\n *** ERROR: Invaild Equation  ***\n\n");
				statement += in;
				in = (char) read.read();
			}
			var = var.trim();
			if(var.indexOf("\"") !=-1){
				newName = var;
			}
			return new Equation(newName.trim(), input.trim(), statement.trim(), var.trim());

	}

	private static boolean EOL(char in) {
		return in == '\n' || in == '\r' || in == '\u0085' || in == '\u2028'
				|| in == '\u2029';
	}

	private static String addName(ArrayList<String> out, String name) {
		try {
			Double.valueOf(name);
			out.add("\"" + name + "\"");
		} catch (NumberFormatException ex) {
			if (!name.trim().equals(""))
				out.add(name);
		}
		return "";
	}

	/**
	 * Converts the equation into the following: temp =
	 * DeriveMetricEquation(input, <equation>) input = tmep.processData() var =
	 * temp.getNewName
	 * 
	 * @param equ
	 * @return
	 */
	private static String writeCode(Equation equ) {
		String out = "TAU_DONTUSE" + " = DeriveMetricEquation(" + equ.getPerfResult() + ",";
		if(!equ.hasNewName()){
			out += formatEquation(equ.getStatement()) + ")\n";
			out += spaces + equ.getPerfResult() + " = TAU_DONTUSE.processData().get(0)\n";
			out += spaces + equ.getNameVar() + " = TAU_DONTUSE.getNewName()\n";
			return out;
		}else{
			out += formatEquation(equ.getStatement())+ ","+equ.getNewName()+")\n";
			out += spaces + equ.getPerfResult() + " = TAU_DONTUSE.processData().get(0)\n";
			return out;
		}
	}
	private static String formatEquation(String input) {
		ArrayList<String> out = new ArrayList<String>();
		String name = "";

		char[] in = input.toCharArray();
		for (char current : in) {
			if (isOperation(current + "") || current == ')' || current == '(') {
				name = addName(out, name);

				// ex: "+"
				out.add("\"" + current + "\"");
			} else {
				name += current;
			}
		}
		name = addName(out, name);
		String output = "";
		for (int i = 0; i < out.size(); i++) {
			output += out.get(i) + "+";
		}
		output = output.substring(0, output.length() - 1);
		return output;
	}

	private static boolean isOperation(String op) {
		char o = op.charAt(0);
		return o == '+' || o == '-' || o == '*' || o == '/';
	}

	private static String findSpaces(String in) {
		String spaces = "";
		int end = in.lastIndexOf('#');
		char current = in.charAt(end - 1);
		for (int i = end - 2; !EOL(current); i--) {
			spaces = current + spaces;
			current = in.charAt(i);
		}
		return spaces;
	}

}

class EquationParseException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5359545769052001459L;

	public EquationParseException() {
		super();
		// TODO Auto-generated constructor stub
	}

	public EquationParseException(String message, Throwable cause) {
		super(message, cause);
		// TODO Auto-generated constructor stub
	}

	public EquationParseException(String message) {
		super(message);
		// TODO Auto-generated constructor stub
	}

	public EquationParseException(Throwable cause) {
		super(cause);
		// TODO Auto-generated constructor stub
	}

}

class Equation{
	private String newName;
	private String perfResult;
	private String statement;
	private String nameVar;
	public Equation(String newName, String perfResult, String statement,
			String nameVar) {
		super();
		this.newName = newName;
		this.perfResult = perfResult;
		this.statement = statement;
		this.nameVar = nameVar;
	}
	public boolean hasNewName(){
		return !newName.equals("");
	}
	public String getNewName() {
		return newName;
	}
	public void setNewName(String newName) {
		this.newName = newName;
	}
	public String getPerfResult() {
		return perfResult;
	}
	public void setPerfResult(String perfResult) {
		this.perfResult = perfResult;
	}
	public String getStatement() {
		return statement;
	}
	public void setStatement(String statement) {
		this.statement = statement;
	}
	public String getNameVar() {
		return nameVar;
	}
	public void setNameVar(String nameVar) {
		this.nameVar = nameVar;
	}
	
}
