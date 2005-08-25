package edu.uoregon.tau.paraprof.barchart;

import java.util.EventListener;

public interface BarChartModelListener extends EventListener {

    void barChartChanged();
}
