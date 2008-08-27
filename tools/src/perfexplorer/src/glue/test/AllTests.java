package glue.test;

import junit.framework.Test;
import junit.framework.TestSuite;

public class AllTests {

	public static Test suite() {
		TestSuite suite = new TestSuite("Test for glue");
		//$JUnit-BEGIN$
		suite.addTestSuite(DrawMMMGraphTest.class);
		suite.addTestSuite(LinearRegressionOperationTest.class);
		suite.addTestSuite(DeriveAllMetricsOperationTest.class);
		suite.addTestSuite(CorrelateEventsWithMetadataTest.class);
		suite.addTestSuite(LogarithmicOperationTest.class);
		suite.addTestSuite(DifferenceOperationTest.class);
		suite.addTestSuite(RatioOperationTest.class);
		suite.addTestSuite(TopXEventsTest.class);
		suite.addTestSuite(DefaultOperationTest.class);
		suite.addTestSuite(ProvenanceTest.class);
		suite.addTestSuite(BasicStatisticsOperationTest.class);
		suite.addTestSuite(ExtractRankOperationTest.class);
		suite.addTestSuite(SupportVectorOperationTest.class);
		suite.addTestSuite(MergeTrialsOperationTest.class);
		suite.addTestSuite(SplitTrialPhasesOperationTest.class);
		suite.addTestSuite(ExtractMetricOperationTest.class);
		suite.addTestSuite(DeriveMetricOperationTest.class);
		suite.addTestSuite(CorrelationOperationTest.class);
		suite.addTestSuite(DrawGraphTest.class);
		suite.addTestSuite(ExtractEventOperationTest.class);
		suite.addTestSuite(SaveResultOperationTest.class);
		suite.addTestSuite(KMeansOperationTest.class);
		suite.addTestSuite(CopyOperationTest.class);
		suite.addTestSuite(ExtractPhasesOperationTest.class);
		suite.addTestSuite(DataSourceResultTest.class);
		suite.addTestSuite(DrawBoxChartGraphTest.class);
		suite.addTestSuite(NaiveBayesOperationTest.class);
		suite.addTestSuite(TrialThreadMetadataTest.class);
		suite.addTestSuite(TrialResultTest.class);
		suite.addTestSuite(TopXPercentEventsTest.class);
		suite.addTestSuite(PCAOperationTest.class);
		suite.addTestSuite(ScalabilityOperationTest.class);
		suite.addTestSuite(DifferenceMetadataOperationTest.class);
		suite.addTestSuite(ScaleMetricOperationTest.class);
		//$JUnit-END$
		return suite;
	}

}
