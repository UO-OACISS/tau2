package edu.uoregon.tau.paraprof;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Metric;

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
    private ParaProfTrial ppTrial;

    public DataSorterWrapper(DataSorter dataSorter, ParaProfTrial ppTrial) {
        super(ppTrial);
        this.parentDataSorter = dataSorter;
        this.ppTrial = ppTrial;

        ParaProfTrial parentTrial = dataSorter.getPpTrial();

        metricMap = new HashMap();

        for (Iterator it = parentTrial.getMetrics().iterator(); it.hasNext();) {
            Metric parentMetric = (Metric)it.next();
            for (Iterator it2 = ppTrial.getMetrics().iterator(); it2.hasNext();) {
                Metric childMetric = (Metric) it2.next();
                if (parentMetric.getName().compareTo(childMetric.getName()) == 0) {
                    metricMap.put(parentMetric, childMetric);
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

    public Metric getSelectedMetric() {
        // map to parent
        Metric parentMetric = parentDataSorter.getSelectedMetric();
        Metric metric = (Metric) metricMap.get(parentMetric);
        if (metric == null) {
            return (Metric)ppTrial.getMetrics().get(0);
        }
        return metric;
    }
}
