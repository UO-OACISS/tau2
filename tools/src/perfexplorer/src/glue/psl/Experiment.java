/**
 * 
 */
package glue.psl;

import java.util.Date;
import java.util.HashSet;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.SourceRegion;
import edu.uoregon.tau.perfdmf.Trial;
import glue.TrialResult;
import java.util.Iterator;

/**
 * @author khuck
 *
 */
public class Experiment {

	private Version version = null;
	private Date startTime = null;
	private Date endTime = null;
	private String commandLine = null;
	private String compiler = null;
	private String compilerOptions = null;
	private Set<RegionSummary> regionSummaries = null;
	private Set<VirtualMachine> virtualMachines = null;
	private TrialResult trialResult = null;
	private CodeRegion topCodeRegion = null;
	private int numberOfProcessingUnits = 1;
	
	/**
	 * 
	 */
	public Experiment(Version version) {
		this.version = version;
		this.regionSummaries = new HashSet<RegionSummary>();
		this.virtualMachines = new HashSet<VirtualMachine>();
	}

	public Experiment (Version version, TrialResult trialResult) {
		this.version = version;
		this.trialResult = trialResult;
		this.regionSummaries = new HashSet<RegionSummary>();
		this.virtualMachines = new HashSet<VirtualMachine>();
		loadEverything();
	}
	
	/**
	 * 
	 */
	public int getNumberOfProcessingUnits() {
		return numberOfProcessingUnits;
	}
	
	/**
	 * @return the commandLine
	 */
	public String getCommandLine() {
		return commandLine;
	}

	/**
	 * @param commandLine the commandLine to set
	 */
	public void setCommandLine(String commandLine) {
		this.commandLine = commandLine;
	}

	/**
	 * @return the compiler
	 */
	public String getCompiler() {
		return compiler;
	}

	/**
	 * @param compiler the compiler to set
	 */
	public void setCompiler(String compiler) {
		this.compiler = compiler;
	}

	/**
	 * @return the compilerOptions
	 */
	public String getCompilerOptions() {
		return compilerOptions;
	}

	/**
	 * @param compilerOptions the compilerOptions to set
	 */
	public void setCompilerOptions(String compilerOptions) {
		this.compilerOptions = compilerOptions;
	}

	/**
	 * @return the endTime
	 */
	public Date getEndTime() {
		return endTime;
	}

	/**
	 * @param endTime the endTime to set
	 */
	public void setEndTime(Date endTime) {
		this.endTime = endTime;
	}

	/**
	 * @return the startTime
	 */
	public Date getStartTime() {
		return startTime;
	}

	/**
	 * @param startTime the startTime to set
	 */
	public void setStartTime(Date startTime) {
		this.startTime = startTime;
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

	public void addRegionSummary(RegionSummary regionSummary) {
		this.regionSummaries.add(regionSummary);
	}
	
	/**
	 * @return the regionSummaries
	 */
	public Set<RegionSummary> getRegionSummaries() {
		return regionSummaries;
	}

	/**
	 * @param regionSummaries the regionSummaries to set
	 */
	public void setRegionSummaries(Set<RegionSummary> regionSummaries) {
		this.regionSummaries = regionSummaries;
	}

	public void addVirtualMachine(VirtualMachine virtualMachine) {
		this.virtualMachines.add(virtualMachine);
	}
	
	/**
	 * @return the virtualMachines
	 */
	public Set<VirtualMachine> getVirtualMachines() {
		return virtualMachines;
	}

	/**
	 * @param virtualMachines the virtualMachines to set
	 */
	public void setVirtualMachines(Set<VirtualMachine> virtualMachines) {
		this.virtualMachines = virtualMachines;
	}

	public RegionSummaryIterator summaryIterator(CodeRegionFilter filter) {
		RegionSummaryIterator iterator = new RegionSummaryIterator(filter, this); 
		return iterator;
	}

	public RegionSummaryIterator summaryIterator() {
		RegionSummaryIterator iterator = new RegionSummaryIterator(this); 
		return iterator;
	}

	private void loadEverything() {
		this.numberOfProcessingUnits = trialResult.getThreads().size();
		// get the list of events - check if we have callpath data
		boolean callpath = false;
		for (String event : trialResult.getEvents()) {
			if (Function.isCallPathFunction(event))
				callpath = true;
		}
		if (!callpath) {
			this.topCodeRegion = buildNonCallpathCodeRegion();
		} else {
			// find main
			this.topCodeRegion = buildCallpathCodeRegion(trialResult.getMainEvent());
		}
	}

	private CodeRegion buildNonCallpathCodeRegion() {
		CodeRegion topCodeRegion = null;
		// no callpath data?  Everything is top level.
		for (String event : trialResult.getEvents()) {
			CodeRegion codeRegion = createCodeRegion(event);
			if (trialResult.getMainEvent().equals(event))
				topCodeRegion = codeRegion;
		
			// create the region summaries
			for (Integer thread : trialResult.getThreads()) {
				RegionSummary regionSummary = new RegionSummary(this, codeRegion, thread.toString());
				regionSummary.setInclusiveTime(trialResult.getInclusive(thread, event, trialResult.getTimeMetric()));
				regionSummary.setExclusiveTime(trialResult.getExclusive(thread, event, trialResult.getTimeMetric()));
				regionSummary.setTotalInstructions(trialResult.getExclusive(thread, event, trialResult.getTotalInstructionMetric()));
				regionSummary.setFloatingPointInstructions(trialResult.getExclusive(thread, event, trialResult.getFPMetric()));
				double[] cacheAccesses = {0.0, 0.0, 0.0};
				double[] cacheMisses = {0.0, 0.0, 0.0};
				cacheMisses[0] = trialResult.getExclusive(thread, event, trialResult.getL1MissMetric());
				cacheMisses[1] = trialResult.getExclusive(thread, event, trialResult.getL2MissMetric());
				cacheMisses[2] = trialResult.getExclusive(thread, event, trialResult.getL3MissMetric());
				regionSummary.setCacheMisses(cacheMisses);
				cacheAccesses[0] = trialResult.getExclusive(thread, event, trialResult.getL1AccessMetric());
				cacheAccesses[1] = trialResult.getExclusive(thread, event, trialResult.getL2AccessMetric());
				cacheAccesses[2] = trialResult.getExclusive(thread, event, trialResult.getL3AccessMetric());
				regionSummary.setCacheAccesses(cacheAccesses);
				regionSummary.setProcessingUnit(thread.toString());
			}
		}
		
		return topCodeRegion;
	}

	private CodeRegion buildCallpathCodeRegion(String parentEvent) {
		// create the code region
		CodeRegion codeRegion = createCodeRegion(parentEvent);
		
		// create the region summaries
		for (Integer thread : trialResult.getThreads()) {
			RegionSummary regionSummary = new RegionSummary(this, codeRegion, thread.toString());
			regionSummary.setInclusiveTime(trialResult.getInclusive(thread, parentEvent, trialResult.getTimeMetric()));
			regionSummary.setExclusiveTime(trialResult.getExclusive(thread, parentEvent, trialResult.getTimeMetric()));
			regionSummary.setTotalInstructions(trialResult.getExclusive(thread, parentEvent, trialResult.getTotalInstructionMetric()));
			regionSummary.setFloatingPointInstructions(trialResult.getExclusive(thread, parentEvent, trialResult.getFPMetric()));
			double[] cacheMisses = {0.0, 0.0, 0.0};
			double[] cacheHits = {0.0, 0.0, 0.0};
			double[] cacheAccesses = {0.0, 0.0, 0.0};
			cacheMisses[0] = trialResult.getExclusive(thread, parentEvent, trialResult.getL1MissMetric());
			cacheMisses[1] = trialResult.getExclusive(thread, parentEvent, trialResult.getL2MissMetric());
			cacheMisses[2] = trialResult.getExclusive(thread, parentEvent, trialResult.getL3MissMetric());
			regionSummary.setCacheMisses(cacheMisses);
			cacheAccesses[0] = trialResult.getExclusive(thread, parentEvent, trialResult.getL1AccessMetric());
			cacheAccesses[1] = trialResult.getExclusive(thread, parentEvent, trialResult.getL2AccessMetric());
			cacheAccesses[2] = trialResult.getExclusive(thread, parentEvent, trialResult.getL3AccessMetric());
			regionSummary.setCacheAccesses(cacheAccesses);
			regionSummary.setProcessingUnit(thread.toString());
		}
		
		// for all the children of this event, create code regions
		for (String event : trialResult.getEvents()) {
			if (event.startsWith(parentEvent) && Function.isCallPathFunction(event)) {
				int linker = event.indexOf(" => " );
				String childEvent = event.substring(linker + 4);
				buildCallpathCodeRegion(childEvent);
			}
		}
		
		return codeRegion;
	}
	
	private CodeRegion createCodeRegion(String event) {
		// extract the source file name and line number from the event name
		SourceRegion sourceLink = Function.getSourceLink(event);
		String fileName = sourceLink.getFilename();
		CodeRegion codeRegion = null;
		if (fileName == null) {
			if (event.startsWith("MPI_")) {
				SourceFile sourceFile = this.version.addSourceFile("MPI");
				codeRegion = new CodeRegion(sourceFile);
				codeRegion.setShortName(event);
				codeRegion.setGroupType(CodeRegion.GroupType.MPI);
			} else {
				SourceFile sourceFile = this.version.addSourceFile("UNKNOWN");
				codeRegion = new CodeRegion(sourceFile);
				codeRegion.setShortName(event);
				codeRegion.setGroupType(CodeRegion.GroupType.GENERAL);
			}
		} else {
			SourceFile sourceFile = this.version.addSourceFile(fileName.trim());
			codeRegion = new CodeRegion(sourceFile);
			codeRegion.setShortName(removeSource(event).trim());
			codeRegion.setStartPositionLine(sourceLink.getStartLine());
			codeRegion.setStartPositionColumn(sourceLink.getStartColumn());
			codeRegion.setEndPositionLine(sourceLink.getEndLine());
			codeRegion.setEndPositionColumn(sourceLink.getEndColumn());
		}
		codeRegion.setLongName(event);
		return codeRegion;
	}
		
	private static String removeSource(String str) {
	    if (str.startsWith("Loop:")) {
	        return str;
	    }
	    while (str.indexOf("[{") != -1) {
	        int a = str.indexOf("[{");
	        int b = str.indexOf("}]");
            str = str.substring(0, a) + str.substring(b + 2);
        }
	    return str;
	}

	/**
	 * @return the topCodeRegion
	 */
	public CodeRegion getTopCodeRegion() {
		return topCodeRegion;
	}

	/**
	 * @param topCodeRegion the topCodeRegion to set
	 */
	public void setTopCodeRegion(CodeRegion topCodeRegion) {
		this.topCodeRegion = topCodeRegion;
	}
		
}
