/************************************************************
 *
 *           File : ProfileDataSource.java
 *         Author : Tyrel Datwyler
 *
 *    Description : Gateway between TauIL and ParaProf for
 *                  retrivial of profile data.
 *
 ************************************************************/

package TauIL.interpreter;

import paraprof.GlobalMapping;
import paraprof.GlobalMappingElement;
import paraprof.ParaProfDataSession;
import paraprof.ParaProfObserver;
import paraprof.TauPprofOutputSession;

import java.io.File;
import java.util.ListIterator;
import java.util.Vector;

class ProfileDataSource extends DataSource implements ParaProfObserver {

    private GlobalMapping event_mapping;
    private GlobalMappingElement event;
    private ParaProfDataSession data;

    private Vector files = new Vector();
    private File [] source_file = new File[1];

    private ListIterator iterator;

    private boolean time_metric = true;
    private boolean loading = false;

    /* Assume by default that profile data is coming from pprof.dat. */
    protected ProfileDataSource() {
	this("pprof.dat");
    }

    /* Supply file name as source of profile data. */
    protected ProfileDataSource(String fname) {
	setFile(fname);
    }

    /* Set the filename for profile data source. */
    protected void setFile(String fname) {
	if (fname == null || fname.equals(""))
	    source_file[0] = new File("pprof.dat");
	else
	    source_file[0] = new File(fname);
    }

    /* Load profile data into memory. */
    protected void load() {
	data = new TauPprofOutputSession();

	files.add(source_file);	
	data.addObserver(this);

	loading = true;
	data.initialize(files);
	
	while(loading) { /* Got Nothing to do so spin our wheels */ }
    }

    /* Called by profile reader on completion. Specified by ParaProfObserver interface. */
    public void update(Object obj){
	this.update();
    }

    /* Defined in ParaProfObserver interface. Called on completion of data load. */
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

	event_mapping = data.getGlobalMapping();
	iterator = event_mapping.getMappingIterator(0);
	loading = false;
    }

    /* The following accessor methods should be self-explanatory. */
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
