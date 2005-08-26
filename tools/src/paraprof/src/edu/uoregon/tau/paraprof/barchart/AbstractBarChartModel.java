package edu.uoregon.tau.paraprof.barchart;

import javax.swing.event.EventListenerList;

/**
 * Implementation of BarChartModel that implements listener handling.
 * Bar Chart models should extend from this rather than implementing
 * BarChartModel directly.
 * 
 * @author Alan Morris
 *
 */
public abstract class AbstractBarChartModel implements BarChartModel {

    /** List of listeners */
    protected EventListenerList listenerList = new EventListenerList();

    /**
     * Default implementation, only one item per row
     * 
     * @see edu.uoregon.tau.paraprof.barchart.BarChartModel#getSubSize()
     */
    public int getSubSize() {
        return 1;
    }

    /**
     * Adds a listener to the list that's notified each time a change
     * to the data model occurs.
     *
     * @param   l       the BarChartModelListener
     */
    public void addBarChartModelListener(BarChartModelListener l) {
        listenerList.add(BarChartModelListener.class, l);
    }

    /**
     * Removes a listener from the list that's notified each time a
     * change to the data model occurs.
     *
     * @param   l       the BarChartModelListener
     */
    public void removeBarChartModelListener(BarChartModelListener l) {
        listenerList.remove(BarChartModelListener.class, l);
    }

    public BarChartModelListener[] getBarChartModelListeners() {
        return (BarChartModelListener[]) listenerList.getListeners(BarChartModelListener.class);
    }

    /**
     * Notifies all listeners that the model has changed.
     */
    public void fireModelChanged() {
        // Guaranteed to return a non-null array
        Object[] listeners = listenerList.getListenerList();
        // Process the listeners last to first, notifying
        // those that are interested in this event
        for (int i = listeners.length - 2; i >= 0; i -= 2) {
            if (listeners[i] == BarChartModelListener.class) {
                ((BarChartModelListener) listeners[i + 1]).barChartChanged();
            }
        }
    }

}
