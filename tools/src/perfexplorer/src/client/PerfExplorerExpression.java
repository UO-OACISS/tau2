package edu.uoregon.tau.perfexplorer.client;

import java.io.FileNotFoundException;
import java.util.List;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;




/**
 * 
 * @author smillst
 *
 */

public class PerfExplorerExpression extends Expression{

	private static String spaces = "  ";
	private String script="";

	public static void main(String args[]){
		PerfExplorerExpression exp = new PerfExplorerExpression();
		//System.out.println(exp.parse("Apples","a+b+x/d*34.4"));
		try {
			System.out.println(exp.getScriptFromExpressions("a","b-c",null,"a+B"));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String getScriptFromExpressions(String app,String experiment, String trial, String expressions) 
		throws ParsingException{
		this.script = "";
		String script = initScript();
		super.parseExpressions(expressions);
		script +=this.script;
		script +=finalizeScript(app,experiment,trial);
		return script;
	}
	public String getScriptFromFile(String app,String experiment, String trial, String fileName) 
		throws ParsingException, FileNotFoundException{
		this.script = "";
		String script = initScript();
		super.parseFile(fileName);
		script +=this.script;
		script +=finalizeScript(app,experiment,trial);
		return script;
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

	private String getDeriveOper(char o) {
		switch (o) {
		case '+':
			return "DeriveMetricOperation.ADD";
		case '-':
			return "DeriveMetricOperation.SUBTRACT";
		case '*':
			return "DeriveMetricOperation.MULTIPLY";
		case '/':
			return "DeriveMetricOperation.DIVIDE";
		}
		return null;
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
		out += loadMethod();
		out += addDerivedOperation();
		out +=addScaledOperation();
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
		script += "def addDerivedOperation(result, metric1, metric2, op, newName=\"\"):\n";
		script += spaces+"# derive the metric\n";
		script += spaces+"derivor = DeriveMetricOperation(result, metric1, metric2, op)\n";
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
	private final static String addScaledOperation(){
		String script = "def addScaledOperation(result, metric1, scale, op, newName=\"\"):\n";
		script += spaces+"# derive the metric\n";
		script += spaces+"scaled = ScaleMetricOperation(result, metric1, scale, op)\n";
		script += spaces+"if newName:\n";
		script+=spaces+"        scaled.setNewName(newName)\n";
		script += spaces+"scaled = scaled.processData().get(0)\n";

		script += spaces+"newName = scaled.getMetrics().toArray()[0]\n";
		script += "\n";
		script += spaces+"merger = MergeTrialsOperation(result)\n";
		script += spaces+"merger.addInput(scaled)\n";
		script += spaces+"scaled = merger.processData().get(0)\n";
		script += spaces+"return scaled, newName\n";
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

	protected String applyOperation(String operand1, String operand2, char operation) {
		script +=spaces+"result, newName = addDerivedOperation( result,";
		script +="\""+operand1 +"\""+","+"\""+operand2 +"\""+", "+getDeriveOper(operation)+")\n";
		return "("+operand1+operation+operand2+")";
	}

	protected String applyOperation(String operand1, double operand2, char operation) {
		script +=spaces+"result, newName = addScaledOperation( result,";
		script +="\""+operand1 +"\""+","+operand2 +", "+getDeriveOper(operation)+")\n";
		return "("+operand1+operation+operand2+")";
	}

	protected String applyOperation(double operand1, String operand2, char operation) {
		script +=spaces+"result, newName = addScaledOperation( result,";
		script +="\""+operand1 +"\""+","+"\""+operand2 +"\""+", "+getDeriveOper(operation)+")\n";
		return "("+operand1+operation+operand2+")";
	}

	protected String applyOperation(String operand1, String operand2,char operation, String newName) {
		script +=spaces+"result, newName = addDerivedOperation( result,";
		script +="\""+operand1 +"\""+","+"\""+operand2 +"\""+", "+getDeriveOper(operation)+",\""+newName.trim()+"\""+")\n";
		return "("+operand1+operation+operand2+")";		
	}

	protected String applyOperation(String operand1, double operand2,char operation, String newName) {
		script +=spaces+"result, newName = addScaledOperation( result,";
		script +="\""+operand1 +"\""+","+operand2 +", "+getDeriveOper(operation)+",\""+newName+"\""+")\n";
		return "("+operand1+operation+operand2+")";
	}

	protected String applyOperation(double operand1, String operand2,char operation, String newName) {
		script +=spaces+"result, newName = addScaledOperation( result,";
		script +=operand1 +","+"\""+operand2 +"\""+", "+getDeriveOper(operation)+",\""+newName+"\""+")\n";
		return "("+operand1+operation+operand2+")";
	}

	protected String rename(String newName, String metric) {
		return applyOperation(0,metric, '+',newName);
	}
}
