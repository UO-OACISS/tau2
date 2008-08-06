/**
 * 
 */
package glue;

import java.util.List;

import junit.framework.TestCase;

/**
 * @author khuck
 *
 */
public class SupportVectorOperationTest extends TestCase {

	/**
	 * Test method for {@link glue.NaiveBayesOperation#processData()}.
	 */
	public final void testProcessData() {
		
		// get some things initialized, but don't use these variables.
		Utilities.setSession("spaceghost");

		// each input is a different class / category for the supervised learning.
		PerformanceResult trainingInput1 = new DefaultResult();
		trainingInput1.setName("first");
		PerformanceResult trainingInput2 = new DefaultResult();
		trainingInput2.setName("second");
		for (int i = 0 ; i < 50 ; i++) {
			int factor1 = 3;
			int factor2 = 5;
			trainingInput1.putExclusive(i, "attr1", "dummy", Math.random() + factor1);
			trainingInput1.putExclusive(i, "attr2", "dummy", Math.random() + factor2);
			trainingInput2.putExclusive(i, "attr1", "dummy", Math.random());
			trainingInput2.putExclusive(i, "attr2", "dummy", Math.random());
		}
		
		NaiveBayesOperation oper = new SupportVectorOperation(trainingInput1, "dummy", AbstractResult.EXCLUSIVE);
		oper.addInput(trainingInput2);
		oper.processData();
		
		PerformanceResult testInput = new DefaultResult();
		for (int i = 0 ; i < 10 ; i++) {
			int factor1 = (i % 2 == 1) ? 0 : 3;
			int factor2 = (i % 2 == 1) ? 0 : 5;
			testInput.putExclusive(i, "attr1", "dummy", Math.random() + factor1);
			testInput.putExclusive(i, "attr2", "dummy", Math.random() + factor2);
		}
		
		List<String> classes = oper.classifyInstances(testInput);
		List<double[]> distributions = oper.getDistributions();
		for (int i = 0 ; i < 10 ; i++) {
			System.out.print(testInput.getExclusive(i, "attr1", "dummy") + ", " +
					testInput.getExclusive(i, "attr2", "dummy") + " = " +
					classes.get(i).toString() + ", distributions: ");
			double[] tmp = distributions.get(i);
			for (int j = 0 ; j < tmp.length ; j++) {
				if (j > 0)
					System.out.print(",");
				System.out.print(tmp[j]);
			}
			System.out.println("");
		}
	}

}
