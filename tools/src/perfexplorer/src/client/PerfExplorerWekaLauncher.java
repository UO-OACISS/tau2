package edu.uoregon.tau.perfexplorer.client;

import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.File;
import java.io.InputStream;
import java.util.List;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.common.RMICubeData;
import edu.uoregon.tau.common.ExternalTool;

public class PerfExplorerWekaLauncher {
	
	private static final int numFunctions = 10;

	public static void launch () {
		// get the server
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMICubeData data = server.requestCubeData(PerfExplorerModel.getModel(), numFunctions); 

        if(data==null)
        {
        	return;
        }
    	
		// get the data values
        float values[][] = data.getValues();

		try {
			// write out a CSV file
			String filename = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + "wekadata.csv";
			FileWriter fstream = new FileWriter(filename);
			BufferedWriter out = new BufferedWriter(fstream);
	
        	for (int i = 0 ; i < numFunctions ; i++) {
				if (i > 0) {
					out.write(",");
				}
        		out.write(Utility.shortenFunctionName(data.getNames(i)));
        	}
			out.newLine();

        	for (int i = 0 ; i < values.length ; i++) {
        		for (int j = 0 ; j < values[i].length ; j++) {
					if (j > 0) {
						out.write(",");
					}
        			out.write(Float.toString(values[i][j]));
        		}
				out.newLine();
        	}

			out.close();
			ExternalTool tool = ExternalTool.createWekaConfiguration(true);
			ExternalTool.launch(tool);
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}	
    }
}
