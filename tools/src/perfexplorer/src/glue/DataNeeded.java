/**
 * 
 */
package glue;

/**
 * @author khuck
 *
 */
public class DataNeeded {

	public static enum DataType {CLASSPATH, FLOATING_POINT, TOTAL_INSTRUCTIONS, TIME, CACHE_ACCESS_L1, CACHE_MISS_L1, CACHE_ACCESS_L2, CACHE_MISS_L2};
	
	private boolean classpath = false;
	private boolean floatingPoint = false;
	private boolean totalInstruction = false;
	private boolean time = false;
	private boolean cacheAccessL1 = false;
	private boolean cacheMissL1 = false;
	private boolean cacheAccessL2 = false;
	private boolean cacheMissL2 = false;
	private DataType dataType = null;
	
	/**
	 * 
	 */
	public DataNeeded() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * 
	 */
	public DataNeeded(DataType type) {
		this.dataType = type;
	}

	/**
	 * @return the cacheAccessL1
	 */
	public boolean isCacheAccessL1() {
		return cacheAccessL1 || this.dataType == DataType.CACHE_ACCESS_L1;
	}

	/**
	 * @param cacheAccessL1 the cacheAccessL1 to set
	 */
	public void setCacheAccessL1(boolean cacheAccessL1) {
		this.cacheAccessL1 = cacheAccessL1;
	}

	/**
	 * @return the cacheAccessL2
	 */
	public boolean isCacheAccessL2() {
		return cacheAccessL2 || this.dataType == DataType.CACHE_ACCESS_L2;
	}

	/**
	 * @param cacheAccessL2 the cacheAccessL2 to set
	 */
	public void setCacheAccessL2(boolean cacheAccessL2) {
		this.cacheAccessL2 = cacheAccessL2;
	}

	/**
	 * @return the cacheMissL1
	 */
	public boolean isCacheMissL1() {
		return cacheMissL1 || this.dataType == DataType.CACHE_MISS_L1;
	}

	/**
	 * @param cacheMissL1 the cacheMissL1 to set
	 */
	public void setCacheMissL1(boolean cacheMissL1) {
		this.cacheMissL1 = cacheMissL1;
	}

	/**
	 * @return the cacheMissL2
	 */
	public boolean isCacheMissL2() {
		return cacheMissL2 || this.dataType == DataType.CACHE_MISS_L2;
	}

	/**
	 * @param cacheMissL2 the cacheMissL2 to set
	 */
	public void setCacheMissL2(boolean cacheMissL2) {
		this.cacheMissL2 = cacheMissL2;
	}

	/**
	 * @return the classpath
	 */
	public boolean isClasspath() {
		return classpath || this.dataType == DataType.CLASSPATH;
	}

	/**
	 * @param classpath the classpath to set
	 */
	public void setClasspath(boolean classpath) {
		this.classpath = classpath;
	}

	/**
	 * @return the floatingPoint
	 */
	public boolean isFloatingPoint() {
		return floatingPoint || this.dataType == DataType.FLOATING_POINT;
	}

	/**
	 * @param floatingPoint the floatingPoint to set
	 */
	public void setFloatingPoint(boolean floatingPoint) {
		this.floatingPoint = floatingPoint;
	}

	/**
	 * @return the time
	 */
	public boolean isTime() {
		return time || this.dataType == DataType.TIME;
	}

	/**
	 * @param time the time to set
	 */
	public void setTime(boolean time) {
		this.time = time;
	}

	/**
	 * @return the totalInstruction
	 */
	public boolean isTotalInstruction() {
		return totalInstruction || this.dataType == DataType.TOTAL_INSTRUCTIONS;
	}

	/**
	 * @param totalInstruction the totalInstruction to set
	 */
	public void setTotalInstruction(boolean totalInstruction) {
		this.totalInstruction = totalInstruction;
	}

	/**
	 * @return the dataType
	 */
	public DataType getDataType() {
		return dataType;
	}

	/**
	 * @param dataType the dataType to set
	 */
	public void setDataType(DataType dataType) {
		this.dataType = dataType;
	}

}
