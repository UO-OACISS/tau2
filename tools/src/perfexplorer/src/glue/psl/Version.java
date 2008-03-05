/**
 * 
 */
package glue.psl;

import java.util.HashSet;
import java.util.Set;

/**
 * @author khuck
 *
 */
public class Version {

	private Application application = null;
	private String versionID = null;
	private Set<SourceFile> sourceFiles = null;
	private Set<Experiment> experiments = null;
	
	/**
	 * 
	 */
	public Version(Application application, String versionID) {
		this.application = application;
		application.addVersion(this);
		this.versionID = versionID;
		this.sourceFiles = new HashSet<SourceFile>();
	}

	/**
	 * @return the application
	 */
	public Application getApplication() {
		return application;
	}

	/**
	 * @param application the application to set
	 */
	public void setApplication(Application application) {
		this.application = application;
	}

	/**
	 * @return the versionID
	 */
	public String getVersionID() {
		return versionID;
	}

	/**
	 * @param versionID the versionID to set
	 */
	public void setVersionID(String versionID) {
		this.versionID = versionID;
	}
	
	public SourceFile addSourceFile(String fileName) {
		for (SourceFile file : sourceFiles) {
			if (file.getName().equals(fileName)) {
				return file;
			}
		}
		SourceFile sourceFile = new SourceFile(this, fileName);
		return sourceFile;
	}

	public SourceFile addSourceFile(SourceFile sourceFile) {
		this.sourceFiles.add(sourceFile);
		return sourceFile;
	}

	/**
	 * @return the sourceFiles
	 */
	public Set<SourceFile> getSourceFiles() {
		return sourceFiles;
	}

	/**
	 * @param sourceFiles the sourceFiles to set
	 */
	public void setSourceFiles(Set<SourceFile> sourceFiles) {
		this.sourceFiles = sourceFiles;
	}

	/**
	 * @return the experiments
	 */
	public Set<Experiment> getExperiments() {
		return experiments;
	}

	/**
	 * @param experiments the experiments to set
	 */
	public void setExperiments(Set<Experiment> experiments) {
		this.experiments = experiments;
	}

}
