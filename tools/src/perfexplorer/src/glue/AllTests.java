package glue;

import junit.framework.Test;
import junit.framework.TestSuite;

public class AllTests {

	public static Test suite() {
		TestSuite suite = new TestSuite("Test for glue");
		//$JUnit-BEGIN$
		suite.addTestSuite(DifferenceMetadataOperationTest.class);
		suite.addTestSuite(PCAOperationTest.class);
		suite.addTestSuite(TrialResultTest.class);
		suite.addTestSuite(DifferenceOperationTest.class);
		suite.addTestSuite(ExtractEventOperationTest.class);
		suite.addTestSuite(DrawGraphTest.class);
		suite.addTestSuite(KMeansOperationTest.class);
		suite.addTestSuite(ScalabilityOperationTest.class);
		suite.addTestSuite(BasicStatisticsOperationTest.class);
		suite.addTestSuite(DrawBoxChartGraphTest.class);
		suite.addTestSuite(CorrelationOperationTest.class);
		suite.addTestSuite(DefaultOperationTest.class);
		suite.addTestSuite(DeriveMetricOperationTest.class);
		suite.addTestSuite(TopXPercentEventsTest.class);
		suite.addTestSuite(ProvenanceTest.class);
		suite.addTestSuite(TrialMetadataTest.class);
		suite.addTestSuite(DrawMMMGraphTest.class);
		suite.addTestSuite(MergeTrialsOperationTest.class);
		suite.addTestSuite(ExtractPhasesOperationTest.class);
		suite.addTestSuite(DataSourceResultTest.class);
		suite.addTestSuite(CopyOperationTest.class);
		suite.addTestSuite(ExtractMetricOperationTest.class);
		suite.addTestSuite(TopXEventsTest.class);
		suite.addTestSuite(ExtractRankOperationTest.class);
		//$JUnit-END$
		return suite;
	}

}
