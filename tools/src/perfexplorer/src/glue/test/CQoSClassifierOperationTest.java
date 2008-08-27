package glue.test;

import edu.uoregon.tau.perfdmf.Trial;
import glue.CQoSClassifierOperation;
import glue.PerformanceAnalysisOperation;
import glue.PerformanceResult;
import glue.SplitTrialPhasesOperation;
import glue.TrialMeanResult;
import glue.TrialMetadata;
import glue.Utilities;

import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

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
		Classifier cls = classifier.getClassifier();
		FastVector classes = null;
		FastVector atts = new FastVector();
		Attribute c = new Attribute(methodField, classes);
		Attribute s = new Attribute("sleep value");
		atts.addElement(c);
		atts.addElement(s);
        Instances test = new Instances("test", atts, 3);
        for (int i = 0 ; i < 5 ; i++) {
        	Instance inst = new Instance(2);
        	inst.setValue(s, i);
        	test.add(inst);
        }
        
        try {
	        for(Enumeration e = test.enumerateInstances(); e.hasMoreElements();) {
	            Instance inst = (Instance)e.nextElement();
		        System.out.print(inst + " = [");
		        double[] dist = cls.distributionForInstance(inst);
		        for (int i = 0 ; i < dist.length ; i++) {
		            System.out.print (dist[i] + ",");
		        }
		        System.out.print("] ");
		        System.out.println(classifier.getClass(dist));
	        }
        } catch (Exception e) {
        	System.err.println(e.getMessage());
        	e.printStackTrace();
        }
	}

}
