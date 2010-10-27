package edu.uoregon.tau.perfexplorer.client;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.List;
import java.util.Scanner;
import java.util.Stack;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;




/**
 * 
 * @author smillst
 *
 */

public class PerfExplorerExpression /*extends Expression*/{

    private static String spaces = "  ";
    //private String script="";

    public static void main(String args[]){
	PerfExplorerExpression exp = new PerfExplorerExpression();
	//System.out.println(exp.parse("Apples","a+b+x/d*34.4"));
	try {
	    System.out.println(exp.getScriptFromExpressions("Matrix","200",null,"\"TIME+TIME\"+300/TIME*20" +
	    		"\n TIME++TIME \n 20*TIME+32/(10+TIME)"));
	} catch (Exception e) {
	    // TODO Auto-generated catch block
	    e.printStackTrace();
	}
    }
    public static String getNewName(String expression){
	try {
	    ArrayList<String> equation = infixToPostfix(expression);
	    if (equation.size()<2){
		return null;

	    }

	    int i = 0;

	    while (equation.size() > 1 && equation.size() > i) {

		if (isOperation(equation.get(i))) {
		    try{
			String oper = equation.remove(i);
			String second = equation.remove(i - 1);
			String first = equation.remove(i - 2);
			i = i - 2;

			equation.add(i, "("+first+oper+second+")");
		    }catch(java.lang.ArrayIndexOutOfBoundsException ex){
			return null;
		    }
		}
		i++;
	    }
	    
	    return equation.get(0);
	}catch(java.lang.ArrayIndexOutOfBoundsException ex){
		return null;
	}catch (Exception e) {
		return null;
	}
    }
    public static boolean validate(String expression){
	String newname= getNewName(expression);
	
	
	return newname != null;
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
    private static boolean isOperation(Object op) {
	if(op instanceof String){
	    String oper = (String)op;
	    char o = oper.charAt(0);
	    return o == '+' || o == '-' || o == '*' || o == '/';
	}else{
	    return false;
	}

    }
    public String getScriptFromExpressions(String app,String experiment, String trial, String expressions) 
    throws ParsingException{
	
	//this.script = "";
	String script = initScript();
	script +=parseExpressions(expressions);
	//script +=this.script;
	script +=finalizeScript(app,experiment,trial);
	return script;
    }
    public String getScriptFromFile(String app,String experiment, String trial, String fileName) 
    throws ParsingException, FileNotFoundException{
	//this.script = "";
	String script = initScript();
	script +=parseFile(fileName);
	//script +=this.script;
	script +=finalizeScript(app,experiment,trial);
	return script;
    }

    public  String parseExpressions(String expressions) throws ParsingException{
	return parseMany(new Scanner(expressions));
    }
    public  String parseFile(String file) throws ParsingException, FileNotFoundException{
	Scanner scan = new Scanner(new File(file));
	return parseMany(scan);
    }
    private  String parseMany(Scanner scan) throws ParsingException{
	String rS = "";
	while(scan.hasNextLine()){
	    String line = scan.nextLine();
	    line =line.trim();
	    validate(line);
	    if(!line.equals("")){	
		rS +=spaces+"result, newName = addDerivedOperation( result,";
		rS +="\""+line+"\")\n";
	    }
	}
	return rS;
    }


    private String finalizeScript(String app, String experiment,String trial){
	if(app == null){
	    return	endOfScript();
	}
	String script = "";
	script += spaces+"saver = SaveResultOperation(result)\n";
	script += spaces +"saver.processData()\n";
	script += spaces +"return\n";
	script += "print \"--------------- JPython test script start ------------\"\n";
	if(experiment == null){
	    script += "inExp = Utilities.getExperimentsForApplication(\""+app+"\")\n";
	    script += "for exp in inExp:\n";
	    script += "  trials = Utilities.getTrialsForExperiment(\""+app+"\", exp.getName())\n";
	    script += "  for trial in trials:\n";
	    script += "     result = load(\""+app+"\",exp.getName(),trial.getName())\n";
	    script += "     computeDerived(result)\n";
	    script += "\n";
	    script += "print \"---------------- JPython test script end -------------\"\n";
	    return script;
	}
	if(trial == null){
	    script += "trials = Utilities.getTrialsForExperiment(\""+app+"\", \""+experiment+"\")\n";
	    script += "for trial in trials:\n";
	    script += "   result = load(\""+app+"\",\""+experiment+"\", trial.getName())\n";
	    script += "   computeDerived(result)\n";
	    script += "\n";
	    script += "print \"---------------- JPython test script end -------------\"\n";
	    return script;
	}
	script += "result = load(\""+app+"\",\""+experiment+"\",\""+trial+"\")\n";
	script += "computeDerived(result)\n";
	script += "print \"---------------- JPython test script end -------------\"\n";

	return script;
    }


    private static String endOfScript() {
	String script = spaces+"saver = SaveResultOperation(result)\n";
	script += spaces +"saver.setForceOverwrite(True)\n";
	script += spaces +"saver.processData()\n";
	script += spaces +"return\n";
	script += "print \"--------------- JPython test script start ------------\"\n";
	script += addApplicaitons();
	script += "for app in apps:\n";
	script += " inExp = Utilities.getExperimentsForApplication(app)\n";
	script += " for exp in inExp:\n";
	script += "   trials = Utilities.getTrialsForExperiment(app, exp.getName())\n";
	script += "   for trial in trials:\n";
	script += "      result = load(app,exp.getName(),trial.getName())\n";
	script += "      computeDerived(result)\n";
	script += "\n";
	script += "print \"---------------- JPython test script end -------------\"\n";
	return script;
    }

    private static String initScript() {
	String out = "from edu.uoregon.tau.perfexplorer.glue import *\n";
	out+= "from java.util import HashSet\n"; 
out+= "from java.util import ArrayList\n";
out+= "from edu.uoregon.tau.perfdmf import Trial\n";
out+= "from edu.uoregon.tau.perfdmf import Metric\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import PerformanceResult\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import PerformanceAnalysisOperation\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import ExtractEventOperation\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import Utilities\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import BasicStatisticsOperation\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import DeriveMetricOperation\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import ScaleMetricOperation\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import DeriveMetricEquation\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import DeriveMetricsFileOperation\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import MergeTrialsOperation\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import TrialResult\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import AbstractResult\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import DrawGraph\n";
out+= "from edu.uoregon.tau.perfexplorer.glue import SaveResultOperation\n";
	out += loadMethod();
	out += addDerivedOperation();
	out += computeDerived();
	return out;
    }

    private static String loadMethod(){
	String script = "\n";
	script += "def load(inApp, inExp, inTrial):\n";
	//	script += spaces+"Utilities.setSession(\"perfdmf\")\n";
	script += spaces+"trial1 = Utilities.getTrial(inApp, inExp, inTrial)\n";
	script += spaces+"result1 = TrialResult(trial1)\n";
	script += spaces+"return result1\n";
	script += "\n";
	return script;
    }
    private final static String addDerivedOperation(){
	String script = "";
	script += "def addDerivedOperation(result, equation, newName=\"\"):\n";
	script += spaces+"# derive the metric\n";
	script += spaces+"derivor = DeriveMetricEquation(result,equation )\n";
	script += spaces+"if newName:\n";
	script += spaces+"      derivor.setNewName(newName)\n";
	script += spaces+"\n";
	script += spaces+"derived = derivor.processData().get(0)\n";
	script += spaces+"newName = derived.getMetrics().toArray()[0]\n";
	script += spaces+"merger = MergeTrialsOperation(result)\n";
	script += spaces+"merger.addInput(derived)\n";
	script += spaces+"derived = merger.processData().get(0)\n";
	script += spaces+"return derived, newName\n";
	script += "\n";
	return script;
    }

    private final static String computeDerived(){
	return "def computeDerived(result):\n";
    }
    private static String addApplicaitons(){
	String output = "apps = [";
	PerfExplorerServer server = PerfExplorerServer.getServer();
	List<Application> apps = server.getApplicationList();

	for (Application app : apps ) {
	    output += "\'"+app.getName()+"\'";
	}

	if(output.charAt(output.length()-1)==',')
	    output = output.substring(0, output.length()-1);
	return output + "]\n";
    }

}
