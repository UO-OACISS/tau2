package TauIL.interpreter;

/* 

import paraprof.ParaProfTrial;
import paraprof.GlobalMapping;
import paraprof.GlobalMappingElement;

*/

// import paraprof.TauOutputSession;

import paraprof.GlobalMapping;
import paraprof.GlobalMappingElement;
import paraprof.ParaProfDataSession;
import paraprof.ParaProfObserver;
import paraprof.TauPprofOutputSession;

// import dms.dss.DataSession;

import java.io.File;
import java.util.ListIterator;
import java.util.Vector;

class ProfileDataSource extends DataSource implements ParaProfObserver {
    //    private ParaProfTrial trial;

    private GlobalMapping event_mapping;
    private GlobalMappingElement event;

    private ParaProfDataSession data;

    private boolean time_metric = true;

    private Vector files = new Vector();
    private File [] source_file = new File[1];
    private ListIterator iterator;

    private boolean loading = false;

    protected ProfileDataSource() {
	this("pprof.dat");
    }

    protected ProfileDataSource(String fname) {
	setFile(fname);
    }

    protected void setFile(String fname) {
	if (fname == null || fname.equals(""))
	    source_file[0] = new File("pprof.dat");
	else
	    source_file[0] = new File(fname);
    }

    protected void load() {
	//	trial = new ParaProfTrial();
	data = new TauPprofOutputSession();

	//	trial.initialize(source_file);
	System.out.println(source_file[0]);
	files.add(source_file);
	
	data.addObserver(this);
	loading = true;
	data.initialize(files);
	
	while(loading) { /* Got Nothing to do so spin our wheels */ }
    }

    public void update(Object obj){
	this.update();
    }

    public void update(){
	data.terminate();

	/*	
	//Set the metrics.
	int numberOfMetrics = dataSession.getNumberOfMetrics();
	for(int i=0;i<numberOfMetrics;i++){
	    Metric metric = this.addMetric();
	    metric.setName(dataSession.getMetricName(i));
	    metric.setTrial(this);
	}
	*/
	
	//	time_metric = data.isTimeMetric();
	System.out.println("Number of mappings : " + data.getNumberOfMappings());
	event_mapping = data.getGlobalMapping();
	iterator = event_mapping.getMappingIterator(0);
	loading = false;
    }

    protected boolean isTimeMetric() {
	return time_metric;
    }

    protected boolean hasNext() {
	return iterator.hasNext();
    }

    protected void next() {
	event = (GlobalMappingElement) iterator.next();
    }

    protected void reset() {
	iterator = event_mapping.getMappingIterator(0);
    }

    protected String getEventName() {
	return event.getMappingName();
    }

    protected double getNumCalls() {
	return event.getMaxNumberOfCalls();
    }

    protected double getNumSubRS() {
	return event.getMaxNumberOfSubRoutines();
    }

    protected double getPercent() {
	return event.getMeanInclusivePercentValue(0);
    }

    protected double getUsec() {
	return getExclusiveValue();
    }

    protected double getCount() {
	return getExclusiveValue();
    }

    protected double getExclusiveValue() {
	return event.getTotalExclusiveValue(0);
    }

    protected double getCumUsec() {
	return getInclusiveValue();
    }

    protected double getTotCount() {
	return getInclusiveValue();
    }

    protected double getInclusiveValue() {
	return event.getTotalInclusiveValue(0);
    }

    protected double getStdDev() {
	return 0.0;
    }

    protected double getUsecsPerCall() {
	return getPerCall();
    }

    protected double getCountsPerCall() {
	return getPerCall();
    }

    protected double getPerCall() {
	return event.getMeanUserSecPerCall(0);
    }
}
