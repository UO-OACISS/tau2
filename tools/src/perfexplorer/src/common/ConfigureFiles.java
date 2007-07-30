package common;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.database.ParseConfig;

/**
 * Simple Class to retrieve database configurations stored on the file system.
 * 
 * @author scottb
 *
 */


abstract public class ConfigureFiles {

	private static final String home = System.getProperty("user.home");
	private static final String slash = System.getProperty("file.separator");
	private static final String ppDir = ".ParaProf";
	private static final String cfg = "perfdmf.cfg";

	/* Returns a List of ParseConfigs. */
	public static List getConfigurations()
	{
		List files = getConfigurationFiles();
		List configs = new ArrayList();
		for (int i=0; i<files.size(); i++)
		{
			configs.add(new ParseConfig((String) files.get(i)));
		}
		return files;
	}
	/* Returns a List of Files. */
	public static List getConfigurationFiles()
	{
		List names = getConfigurationNames();
		List files = new ArrayList();
		for (int i=0; i<names.size(); i++)
		{
			files.add(new File(home + slash + ppDir + slash + (String) names.get(i)));
		}
		return files;
	}
	/* Returns a List of Strings. */
	public static List getConfigurationNames()
	{
		File paraprofDirectory = new File(home + slash + ppDir);
		String[] fileNames = paraprofDirectory.list();
		List perfdmfConfigs  = new ArrayList();
		if (fileNames != null) {
			for (int i = 0; i<fileNames.length; i++)
			{
				if (fileNames[i].startsWith(cfg) && !fileNames[i].endsWith("~"))
					perfdmfConfigs.add(fileNames[i]);
			}
		}
		return perfdmfConfigs;
	}
	/* testing only */
	public static void main(String[] args)
	{
		System.out.println("Names: ");
		System.out.println(getConfigurationFiles());	
	}
}