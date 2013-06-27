package edu.uoregon.tau.perfexplorer.server;

import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfexplorer.common.ChartDataType;
import edu.uoregon.tau.perfexplorer.common.RMIGeneralChartData;
import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorerModel;

/**
 * The GeneralChartData class is used to select data from the database which 
 * represents the performance profile of the selected trials, and return them
 * in a format for JFreeChart to display them.
 *
 * <P>CVS $Id: GeneralChartData.java,v 1.38 2009/08/04 22:19:01 wspear Exp $</P>
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public class GeneralChartData extends RMIGeneralChartData {


	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected GeneralChartData ( ChartDataType dataType) {
		super (dataType);
	}

	/**
	 * Main method.  The dataType represents the type of chart data being
	 * requested.  The model represents the selected trials of interest.
	 * 
	 * @param model
	 * @param dataType
	 * @return
	 */
	public static GeneralChartData getChartData(RMIPerfExplorerModel model,  ChartDataType dataType) {
		//PerfExplorerOutput.println("getChartData(" + model.toString() + ")...");
		DB db = PerfExplorerServer.getServer().getDB();
		
		//System.out.println("Do just the perfdmf approach");
		//return new PerfDMFGeneralChartData(model, dataType);
		
		if (db.getSchemaVersion() == 0){
			return new PerfDMFGeneralChartData(model, dataType);
		}else {
			return new TAUdbGeneralChartData(model, dataType);
		}
	}


}

