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
public class Application {

	private String name = null;
	private Set<Version> versions = null;
	
	/**
	 * 
	 */
	public Application(String name) {
		this.name = name;
		this.versions = new HashSet<Version>();
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}

	public void addVersion(Version version) {
		this.versions.add(version);
	}
	
	/**
	 * @return the versions
	 */
	public Set<Version> getVersions() {
		return versions;
	}

	/**
	 * @param versions the versions to set
	 */
	public void setVersions(Set<Version> versions) {
		this.versions = versions;
	}

}
