/**
 * 
 */
package rules;

import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.Reader;

import org.drools.FactHandle;
import org.drools.RuleBase;
import org.drools.RuleBaseFactory;
import org.drools.WorkingMemory;
import org.drools.compiler.PackageBuilder;
import org.drools.rule.Package;

/**
 * @author khuck
 *
 */
public class RuleHarness {
	
    private WorkingMemory workingMemory = null;
    private RuleBase ruleBase = null;
    private WorkingMemoryStringLogger logger = null;
	private PackageBuilder builder = null;
	
	private static RuleHarness instance = null;
	
	public static RuleHarness useGlobalRules(String ruleFile) {
//		if (instance == null) {
			instance = new RuleHarness(ruleFile);
//		} else {
//			instance.addRules(ruleFile);
//		}
		return instance;
	}

    public RuleHarness(String ruleFile) {
        try {
        	//load up the rulebase
            readRule(ruleFile);
            this.workingMemory = this.ruleBase.newWorkingMemory();
            this.logger = new WorkingMemoryStringLogger(this.workingMemory);
        } catch (Throwable t) {
            t.printStackTrace();
        }
    }
    
    public void addRules(String ruleFile) {
        try {
        	//load up the rulebase
            readRule(ruleFile);
        } catch (Throwable t) {
            t.printStackTrace();
        }
    }
       

    public void processRules() {
        try {
        	System.out.println("Firing rules...");
            this.workingMemory.fireAllRules();   
        	System.out.println("...done with rules.");
        } catch (Throwable t) {
            t.printStackTrace();
        }
    }
    
    public static FactHandle assertObject(Object object) {
    	FactHandle handle = null;
    	RuleHarness harness = RuleHarness.getInstance();
    	if (harness != null) {
	    	handle = harness.workingMemory.assertObject(object);
	    	if (object instanceof SelfAsserting) {
	    		SelfAsserting selfAsserting = (SelfAsserting)object;
	    		selfAsserting.assertFacts(harness);
	    	}
    	}
    	return handle;
    }
    
    public static void retractObject(FactHandle handle) {
    	RuleHarness harness = RuleHarness.getInstance();
    	if (harness != null) {
    		harness.workingMemory.retractObject(handle);
    	}
    	return; 
    }
    
    /**
     * Please note that this is the "low level" rule assembly API.
     */
	private void readRule(String ruleFile) throws Exception {
		if (ruleFile == null) {
			ruleFile =  "/rules/PerfExplorer.drl" ;
		}
		
		Reader source = null;
		try {
			//read in the source
			source = new InputStreamReader( RuleHarness.class.getResourceAsStream(ruleFile) );
		} catch (Exception e) {
			// it may not be a resource - now try the file system
			try {
				source = new FileReader( ruleFile );
			} catch (Exception e2) {
				System.err.println("Error reading rules!");
				System.err.println(e.getMessage());
				System.err.println(e2.getMessage());
				e.printStackTrace(System.err);
				return;
			}
		}

		if (source == null) {
			System.err.println("Unable to read rules...");
			return;
		}
		
		if (this.builder == null) {
			this.builder = new PackageBuilder();
		}

		//this wil parse and compile in one step
		//NOTE: There are 2 methods here, the one argument one is for normal DRL.
		System.out.print("Reading rules: " + ruleFile + "...");
		builder.addPackageFromDrl( source );
		System.out.println(" done.");

		//get the compiled package (which is serializable)
		Package pkg = builder.getPackage();

		if (this.ruleBase == null) {
			//add the package to a rulebase (deploy the rule package).
			this.ruleBase = RuleBaseFactory.newRuleBase();
		}
		this.ruleBase.addPackage( pkg );
	}

	public String getLog() {
		return this.logger.toString();
	}

	/**
	 * @return the instance
	 */
	public static RuleHarness getInstance() {
		return instance;
	}

}
