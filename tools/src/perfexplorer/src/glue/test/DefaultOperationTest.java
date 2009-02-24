package edu.uoregon.tau.perfexplorer.glue.test;

import edu.uoregon.tau.perfexplorer.glue.AbstractPerformanceOperation;
import edu.uoregon.tau.perfexplorer.glue.DefaultOperation;
import edu.uoregon.tau.perfexplorer.glue.DefaultResult;
import edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation;
import edu.uoregon.tau.perfexplorer.glue.PerformanceResult;
import edu.uoregon.tau.perfexplorer.glue.Provenance;

import java.util.List;

import junit.framework.TestCase;

/**
 * This class is a JUnit test for the default operation class.
 * It also tests the Provenance class.
 * 
 * <P>CVS $Id: DefaultOperationTest.java,v 1.2 2009/02/24 00:53:43 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0
 */
public class DefaultOperationTest extends TestCase {
	public DefaultOperationTest() {
		super();
	}
	
	public void testProcessData() {
		System.out.print("Testing DefaultOperation.processData()... ");
		PerformanceResult input = new DefaultResult();
		AbstractPerformanceOperation dummy = new DefaultOperation(input);
		List<PerformanceResult> outputs = dummy.processData();
		
		// TestCase.assert we have a list of output
		TestCase.assertNotNull(outputs);

		// TestCase.assert we get the same output if we get the output
		TestCase.assertSame(outputs, dummy.getOutputs());
		
		// TestCase.assert the actual output is valid
		TestCase.assertNotNull(outputs.get(0));
		TestCase.assertSame(outputs.get(0), dummy.getOutputAtIndex(0));
		
		// TestCase.assert that input matches output
		TestCase.assertSame(input, outputs.get(0));
		
		// TestCase.assert that the provenance is not null
		Provenance provenance = Provenance.getCurrent();
		TestCase.assertNotNull(provenance);
		
		// Assert the data in the provenance is valid
		List<PerformanceAnalysisOperation> operations = provenance.getOperations();
		TestCase.assertNotNull(operations);
		
		// Assert there is one operation
		//TestCase.assertEquals(operations.size(), 1);
		
		// Assert it is our operation
		PerformanceAnalysisOperation operation = operations.get(operations.size()-1);
		TestCase.assertNotNull(operation);
		TestCase.assertEquals(operation.getClass(), dummy.getClass());
		TestCase.assertSame(operation, dummy);
		
		// Assert the operation input and output are our input and output
		TestCase.assertSame(operation.getInputs().get(0), input);
		TestCase.assertSame(operation.getOutputs(), outputs);
		
		System.out.println("done.");
	}
	
	public static void main (String[] args) {
		junit.textui.TestRunner.run(DefaultOperationTest.class);
	}
}
