package dms.dss;

public class Experiment {
	private int experimentID;
	private int applicationID;
	private String systemInfo;
	private String configurationInfo;
	private String instrumentationInfo;
	private String compilerInfo;
	private String trialTableName;

	public int getID () {
		return experimentID;
	}

	public int getApplicationID() {
		return applicationID;
	}
	
	public String getSystemInfo() {
		return systemInfo;
	}
	
	public String getConfigInfo() {
		return configurationInfo;
	}
	
	public String getInstrumentationInfo() {
		return instrumentationInfo;
	}
	
	public String getCompilerInfo() {
		return compilerInfo;
	}
	
	public String getTrialTableName() {
		return trialTableName;
	}
	
	public void setID (int id) {
		this.experimentID = id;
	}

	public void setApplicationID (int appid) {
		this.applicationID = appid;
	}

	public void setSystemInfo (String systemInfo) {
		this.systemInfo = systemInfo;
	}

	public void setConfigurationInfo (String configurationInfo) {
		this.configurationInfo = configurationInfo;
	}

	public void setInstrumentationInfo (String instrumentationInfo) {
		this.instrumentationInfo = instrumentationInfo;
	}

	public void setCompilerInfo (String compilerInfo) {
		this.compilerInfo = compilerInfo;
	}

	public void setTrialTableName (String trialTableName) {
		this.trialTableName = trialTableName;
	}

}


