package edu.uoregon.tau.perfexplorer.glue.test;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import junit.framework.TestCase;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.glue.CQoSClassifierOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.SplitTrialPhasesOperation;
import edu.uoregon.tau.perfexplorer.glue.TrialMeanResult;
import edu.uoregon.tau.perfexplorer.glue.Utilities;

public class CQoSClassifierOperationTest extends TestCase {

	public void testProcessData() {
		Utilities.setSession("local");
		Trial trial = Utilities.getTrial("simple", "test", "method2");
		TrialMeanResult result = new TrialMeanResult(trial);
		PerformanceAnalysisOperation operator = new SplitTrialPhasesOperation(result, "Iteration");
		operator.addInput(new TrialMeanResult(trial));
		Trial trial2 = Utilities.getTrial("simple", "test", "method1");
		operator.addInput(new TrialMeanResult(trial2));
		Trial trial3 = Utilities.getTrial("simple", "test", "method 3");
		operator.addInput(new TrialMeanResult(trial3));
		List<PerformanceResult> outputs = operator.processData();
		
		Set<String> metadataFields = new HashSet<String>();
		//metadataFields.add("method name");
		metadataFields.add("sleep value");
		String methodField = "method name";
		CQoSClassifierOperation classifier = new CQoSClassifierOperation(outputs, result.getTimeMetric(), metadataFields, methodField);
		classifier.processData();
		
        System.out.println("");
        for (int i = 0 ; i < 5 ; i++) {
    		Map<String,String> inputFields = new HashMap<String,String>();
        	inputFields.put("sleep value", Integer.toString(i));
	        System.out.println(inputFields + ", " + classifier.getClass(inputFields) + ", confidence: " + classifier.getConfidence());
        }
        
        String fileName = "/tmp/pleasework.classifier";
        classifier.writeClassifier(fileName);
	}

}
