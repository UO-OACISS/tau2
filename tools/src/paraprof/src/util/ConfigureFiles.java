package edu.uoregon.tau.paraprof.util;
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

	/* Returns a List of ParseConfigs. */
	static List getConfigurations()
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
	static List getConfigurationFiles()
	{
		List names = getConfigurationNames();
		List files = new ArrayList();
		for (int i=0; i<names.size(); i++)
		{
			files.add(new File(System.getProperty("user.home") + "/.ParaProf/" + (String) names.get(i)));
		}
		return files;
	}
	/* Returns a List of Strings. */
	static List getConfigurationNames()
	{
		File paraprofDirectory = new File(System.getProperty("user.home") + "/.ParaProf");
		String[] fileNames = paraprofDirectory.list();
		List perfdmfConfigs  = new ArrayList();
		for (int i = 0; i<fileNames.length; i++)
		{
			if (fileNames[i].startsWith("perfdmf.cfg") && !fileNames[i].endsWith("~"))
				perfdmfConfigs.add(fileNames[i]);
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