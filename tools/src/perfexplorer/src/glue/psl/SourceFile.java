/**
 * 
 */
package glue.psl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author khuck
 *
 */
public class SourceFile {

	private Version version = null;
	private String name = null;
	private List<String> contents = null;
	private Set<CodeRegion> codeRegions = null;
	
	/**
	 * 
	 */
	public SourceFile(Version version, String name) {
		this.version = version;
		version.addSourceFile(this);
		this.name = name;
		this.contents = new ArrayList<String>();
		this.codeRegions = new HashSet<CodeRegion>();
	}

	/**
	 * @return the contents
	 */
	public List<String> getContents() {
		return contents;
	}

	/**
	 * @param contents the contents to set
	 */
	public void setContents(List<String> contents) {
		this.contents = contents;
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

	/**
	 * @return the version
	 */
	public Version getVersion() {
		return version;
	}

	/**
	 * @param version the version to set
	 */
	public void setVersion(Version version) {
		this.version = version;
	}
	
	public void addCodeRegion(CodeRegion codeRegion) {
		this.codeRegions.add(codeRegion);
	}

	/**
	 * @return the codeRegions
	 */
	public Set<CodeRegion> getCodeRegions() {
		return codeRegions;
	}

	/**
	 * @param codeRegions the codeRegions to set
	 */
	public void setCodeRegions(Set<CodeRegion> codeRegions) {
		this.codeRegions = codeRegions;
	}
}
