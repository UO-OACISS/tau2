package edu.uoregon.tau.perfexplorer.client;

public final class TestType {
	private final transient String _name;

	public static final TestType CHART = new TestType("chart");
	public static final TestType CLUSTER = new TestType("cluster");
	public static final TestType VIZ = new TestType("viz");
	public static final TestType CORRELATION = new TestType("correlation");
	public static final TestType SCRIPT = new TestType("script");
	public static final TestType VIEWS = new TestType("views");
	public static final TestType RULES = new TestType("rules");
	public static final TestType ALL = new TestType("all");

	private TestType(String name) {
		this._name = name;
	}

	public String toString() {
		return this._name;
	}

	public static TestType getType(String name) throws Exception {
		String lowerName = name.toLowerCase();
		if (lowerName.equals(CHART.toString()))
			return CHART;
		if (lowerName.equals(CLUSTER.toString()))
			return CLUSTER;
		if (lowerName.equals(VIZ.toString()))
			return VIZ;
		if (lowerName.equals(CORRELATION.toString()))
			return CORRELATION;
		if (lowerName.equals(SCRIPT.toString()))
			return SCRIPT;
		if (lowerName.equals(VIEWS.toString()))
			return VIEWS;
		if (lowerName.equals(RULES.toString()))
			return RULES;
		if (lowerName.equals(ALL.toString()))
			return ALL;
		Exception e = new Exception ("Unknown test type.");
		throw (e);
	}
}


