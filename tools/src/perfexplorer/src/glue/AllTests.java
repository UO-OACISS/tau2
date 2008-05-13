package glue;

import junit.framework.Test;
import junit.framework.TestSuite;

public class AllTests {

	public static Test suite() {
		TestSuite suite = new TestSuite("Test for glue");
		//$JUnit-BEGIN$
		suite.addTestSuite(DataSourceResultTest.class);
		suite.addTestSuite(TrialMetadataTest.class);
		suite.addTestSuite(DrawGraphTest.class);
		suite.addTestSuite(DefaultOperationTest.class);
		suite.addTestSuite(TrialResultTest.class);
		suite.addTestSuite(PCAOperationTest.class);
		suite.addTestSuite(TrialThreadMetadataTest.class);
		suite.addTestSuite(DeriveMetricOperationTest.class);
		suite.addTestSuite(TopXPercentEventsTest.class);
		suite.addTestSuite(CopyOperationTest.class);
		suite.addTestSuite(ProvenanceTest.class);
		suite.addTestSuite(CorrelationOperationTest.class);
		suite.addTestSuite(DrawBoxChartGraphTest.class);
		suite.addTestSuite(ExtractMetricOperationTest.class);
		suite.addTestSuite(MergeTrialsOperationTest.class);
		suite.addTestSuite(CorrelateEventsWithMetadataTest.class);
		suite.addTestSuite(LinearRegressionOperationTest.class);
		suite.addTestSuite(DeriveAllMetricsOperationTest.class);
		suite.addTestSuite(DifferenceMetadataOperationTest.class);
		suite.addTestSuite(ExtractPhasesOperationTest.class);
		suite.addTestSuite(DrawMMMGraphTest.class);
		suite.addTestSuite(ExtractRankOperationTest.class);
		suite.addTestSuite(DifferenceOperationTest.class);
		suite.addTestSuite(ExtractEventOperationTest.class);
		suite.addTestSuite(RatioOperationTest.class);
		suite.addTestSuite(LogarithmicOperationTest.class);
		suite.addTestSuite(KMeansOperationTest.class);
		suite.addTestSuite(ScaleMetricOperationTest.class);
		suite.addTestSuite(ScalabilityOperationTest.class);
		suite.addTestSuite(TopXEventsTest.class);
		suite.addTestSuite(BasicStatisticsOperationTest.class);
		//$JUnit-END$
		return suite;
	}

}
