package edu.uoregon.tau.paraprof.barchart;

import java.util.EventListener;

/**
 * Simple interface for listeners of BarChartModels.
 * 
 * <P>CVS $Id: BarChartModelListener.java,v 1.2 2005/08/26 01:49:03 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public interface BarChartModelListener extends EventListener {

    void barChartChanged();
}
