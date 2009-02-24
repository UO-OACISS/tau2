/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author khuck
 *
 */
public class CodeRegion {
	
	public enum GroupType {MPI, GENERAL};

	private SourceFile sourceFile = null;
	private int startPositionColumn = 0;
	private int startPositionLine = 0;
	private int endPositionColumn = 0;
	private int endPositionLine = 0;
	private String shortName = null;
	private Set<PerformanceProperty> performanceProperties = null;
	private List<CodeRegion> codeRegions = null;
	private String longName = null;
	private GroupType groupType = null;

	/**
	 * 
	 */
	public CodeRegion(SourceFile sourceFile) {
		this.sourceFile = sourceFile;
		this.performanceProperties = new HashSet<PerformanceProperty>();
	}

	/**
	 * @return the endPositionColumn
	 */
	public int getEndPositionColumn() {
		return endPositionColumn;
	}

	/**
	 * @param endPositionColumn the endPositionColumn to set
	 */
	public void setEndPositionColumn(int endPositionColumn) {
		this.endPositionColumn = endPositionColumn;
	}

	/**
	 * @return the endPositionLine
	 */
	public int getEndPositionLine() {
		return endPositionLine;
	}

	/**
	 * @param endPositionLine the endPositionLine to set
	 */
	public void setEndPositionLine(int endPositionRow) {
		this.endPositionLine = endPositionRow;
	}

	/**
	 * @return the sourceFile
	 */
	public SourceFile getSourceFile() {
		return sourceFile;
	}

	/**
	 * @param sourceFile the sourceFile to set
	 */
	public void setSourceFile(SourceFile sourceFile) {
		this.sourceFile = sourceFile;
	}

	/**
	 * @return the startPositionColumn
	 */
	public int getStartPositionColumn() {
		return startPositionColumn;
	}

	/**
	 * @param startPositionColumn the startPositionColumn to set
	 */
	public void setStartPositionColumn(int startPositionColumn) {
		this.startPositionColumn = startPositionColumn;
	}

	/**
	 * @return the startPositionLine
	 */
	public int getStartPositionLine() {
		return startPositionLine;
	}

	/**
	 * @param startPositionLine the startPositionLine to set
	 */
	public void setStartPositionLine(int startPositionRow) {
		this.startPositionLine = startPositionRow;
	}
	
	public void addPerformanceProperty(PerformanceProperty performanceProperty) {
		this.performanceProperties.add(performanceProperty);
	}

	/**
	 * @return the performanceProperties
	 */
	public Set<PerformanceProperty> getPerformanceProperties() {
		return performanceProperties;
	}

	/**
	 * @param performanceProperties the performanceProperties to set
	 */
	public void setPerformanceProperties(
			Set<PerformanceProperty> performanceProperties) {
		this.performanceProperties = performanceProperties;
	}
	
	public void addCodeRegion(CodeRegion codeRegion) {
		if (this.codeRegions == null) {
			this.codeRegions = new ArrayList<CodeRegion>();
		}
		this.codeRegions.add(codeRegion);
	}

	/**
	 * @return the codeRegions
	 */
	public List<CodeRegion> getCodeRegions() {
		return codeRegions;
	}

	/**
	 * @param codeRegions the codeRegions to set
	 */
	public void setCodeRegions(List<CodeRegion> codeRegions) {
		this.codeRegions = codeRegions;
	}

	/**
	 * @return the shortName
	 */
	public String getShortName() {
		return shortName;
	}

	/**
	 * @param shortName the shortName to set
	 */
	public void setShortName(String shortName) {
		this.shortName = shortName;
	}

	/**
	 * @return the longName
	 */
	public String getLongName() {
		return longName;
	}

	/**
	 * @param longName the longName to set
	 */
	public void setLongName(String longName) {
		this.longName = longName;
	}

	/**
	 * @return the groupName
	 */
	public GroupType getGroupType() {
		return groupType;
	}

	/**
	 * @param groupName the groupName to set
	 */
	public void setGroupType(GroupType groupType) {
		this.groupType = groupType;
	}

}
