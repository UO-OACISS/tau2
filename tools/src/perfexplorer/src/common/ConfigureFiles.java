package edu.uoregon.tau.perfexplorer.common;
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
	public static List<File> getConfigurations()
	{
		List<File> files = getConfigurationFiles();
		List<ParseConfig> configs = new ArrayList<ParseConfig>();
		for (int i=0; i<files.size(); i++)
		{
			configs.add(new ParseConfig(files.get(i).getName()));
		}
		return files;
	}
	/* Returns a List of Files. */
	public static List<File> getConfigurationFiles()
	{
		List<String> names = getConfigurationNames();
		List<File> files = new ArrayList<File>();
		for (int i=0; i<names.size(); i++)
		{
			files.add(new File(home + slash + ppDir + slash + names.get(i)));
		}
		return files;
	}
	/* Returns a List of Strings. */
	public static List<String> getConfigurationNames()
	{
		File paraprofDirectory = new File(home + slash + ppDir);
		String[] fileNames = paraprofDirectory.list();
		List<String> perfdmfConfigs  = new ArrayList<String>();
		if (fileNames != null) {
			for (int i = 0; i<fileNames.length; i++)
			{
				if (fileNames[i].startsWith(cfg) && !fileNames[i].endsWith("~"))
					perfdmfConfigs.add(home + slash + ppDir + slash + fileNames[i]);
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