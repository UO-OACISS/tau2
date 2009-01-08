package edu.uoregon.tau.paraprof;

import java.util.HashMap;
import java.util.Map;

import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.perfdmf.Function;

/*
 * This class is a total kludge to allow for trials in a comparison window to map metric ids to each other.
 * The problem is that the comparison window only has one DataSorter and hence only one "selectedMetric".
 * This metric is a number, and in the other trials PAPI_FP_INS may be a completely different number.
 * So this class wraps it and performs the mapping
 *
 * I'm really abusing Java inheritence here, this thing is more a compositer, but I need a change in the original to 
 * appear as a change to all.  It's the quickest fix I could think of.
 */

public class DataSorterWrapper extends DataSorter {

    private DataSorter parentDataSorter;
    private Map metricMap;

    public DataSorterWrapper(DataSorter dataSorter, ParaProfTrial ppTrial) {
        super(ppTrial);
        this.parentDataSorter = dataSorter;

        ParaProfTrial parentTrial = dataSorter.getPpTrial();

        metricMap = new HashMap();
        for (int i = 0; i < parentTrial.getNumberOfMetrics(); i++) {
            for (int j = 0; j < ppTrial.getNumberOfMetrics(); j++) {
                if (parentTrial.getMetricName(i).compareTo(ppTrial.getMetricName(j)) == 0) {
                    // Where's java 1.5 for autoboxing?!
                    metricMap.put(new Integer(i), new Integer(j));
                }
            }

        }
    }

    public boolean getDescendingOrder() {
        return parentDataSorter.getDescendingOrder();
    }

    public Function getPhase() {
        return parentDataSorter.getPhase();
    }

    public SortType getSortType() {
        return parentDataSorter.getSortType();
    }

    public ValueType getSortValueType() {
        return parentDataSorter.getSortValueType();
    }

    public boolean isDerivedMetric() {
        return parentDataSorter.isDerivedMetric();
    }

    public boolean isTimeMetric() {
        return parentDataSorter.isTimeMetric();
    }

    public ValueType getValueType() {
        return parentDataSorter.getValueType();
    }

    public int getSelectedMetricID() {
        // map to parent
        int parentMetricID = parentDataSorter.getSelectedMetricID();
        Integer metricID = (Integer) metricMap.get(new Integer(parentMetricID));
        if (metricID == null) {
            return 0;
        }
        return metricID.intValue();
    }

}
