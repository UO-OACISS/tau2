package edu.uoregon.tau.perfexplorer.glue;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;

public class DeriveMetricsFileOperation extends AbstractPerformanceOperation {

    /**
	 * 
	 */
	private static final long serialVersionUID = 1530523733671075551L;

	private PerformanceResult input = null;

    private String filename;

    public DeriveMetricsFileOperation() {
	super();
	// TODO Auto-generated constructor stub
    }

    public DeriveMetricsFileOperation(List<PerformanceResult> inputs) {
	super(inputs);
	// TODO Auto-generated constructor stub
    }

    public DeriveMetricsFileOperation(PerformanceResult input) {
	super(input);
	// TODO Auto-generated constructor stub
    }
    public DeriveMetricsFileOperation(PerformanceResult input, String filename) {
	super(input);
	this.filename = filename;
	this.input = input;
    }

    public DeriveMetricsFileOperation(Trial trial) {
	super(trial);
	// TODO Auto-generated constructor stub
    }

    public List<PerformanceResult> processData() {
	try {
	    LineNumberReader scan = new LineNumberReader(new FileReader(filename));
	    String line = scan.readLine();
	    while (line != null){
		DeriveMetricEquation derive = new DeriveMetricEquation(input,line);
		if(derive.noErrors()){
		    MergeTrialsOperation merger = new MergeTrialsOperation(input);
		    PerformanceResult derived = derive.processData().get(0);
		    merger.addInput(derived);
		    input = merger.processData().get(0);
		}else{
		    System.err.println("\n\n *** ERROR: This equation was not derived: " + line + " ***\n\n");
		}
		    
		line = scan.readLine();	
	    }
	    outputs.add(input);

	} catch (FileNotFoundException e) {
	    // TODO Auto-generated catch block
	    e.printStackTrace();
	} catch (IOException e) {
	    // TODO Auto-generated catch block
	    e.printStackTrace();
	}

	// TODO Auto-generated method stub
	return outputs;
    }

}
