package dms.dss;

// get these from Robert
public class Application {
	private int applicationID;
	private String name;
	private String version;
	private String description;
	private String language;
	private String paraDiag;
	private String usage;
	private String executableOptions;
	private String experimentTableName;

	public int getID() {
		return applicationID;
	}

	public String getName() {
		return name;
	}

	public String getVersion() {
		return version;
	}

	public String getDescription() {
		return description;
	}

	public String getLanguage() {
		return language;
	}

	public String getParaDiag() {
		return paraDiag;
	}

	public String getUsage() {
		return usage;
	}

	public String getExecutableOptions() {
		return executableOptions;
	}

	public String getExperimentTableName() {
		return experimentTableName;
	}

	public void setID(int id) {
		this.applicationID = id;
	}

	public void setName(String name) {
		this.name = name;
	}

	public void setVersion(String version) {
		this.version = version;
	}

	public void setDescription(String description) {
		this.description = description;
	}

	public void setLanguage(String language) {
		this.language = language;
	}

	public void setParaDiag(String paraDiag) {
		this.paraDiag = paraDiag;
	}

	public void setUsage(String usage) {
		this.usage = usage;
	}

	public void setExecutableOptions(String executableOptions) {
		this.executableOptions = executableOptions;
	}

	public void setExperimentTableName(String experimentTableName) {
		this.experimentTableName = experimentTableName;
	}
}

