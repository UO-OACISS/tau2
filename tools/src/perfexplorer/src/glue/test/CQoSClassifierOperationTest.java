package glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import glue.CQoSClassifierOperation;
import glue.PerformanceAnalysisOperation;
import glue.PerformanceResult;
import glue.SplitTrialPhasesOperation;
import glue.TrialMeanResult;
import glue.Utilities;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cqos.WekaClassifierWrapper;


import junit.framework.TestCase;

public class CQoSClassifierOperationTest extends TestCase {

	public void testProcessData() {
		Utilities.setSession("cqos");
		Trial trial = Utilities.getTrial("simple", "test", "method 2");
		TrialMeanResult result = new TrialMeanResult(trial);
		PerformanceAnalysisOperation operator = new SplitTrialPhasesOperation(result, "Iteration");
		operator.addInput(new TrialMeanResult(trial));
		trial = Utilities.getTrial("simple", "test", "method 1");
		operator.addInput(new TrialMeanResult(trial));
		operator.addInput(new TrialMeanResult(trial));
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
