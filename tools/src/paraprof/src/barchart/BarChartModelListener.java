package edu.uoregon.tau.paraprof.barchart;

import java.util.EventListener;

/**
 * Simple interface for listeners of BarChartModels.
 * 
 * <P>CVS $Id: BarChartModelListener.java,v 1.1 2005/09/26 21:12:12 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public interface BarChartModelListener extends EventListener {

    void barChartChanged();
}
