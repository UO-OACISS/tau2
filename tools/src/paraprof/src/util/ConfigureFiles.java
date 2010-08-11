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
	static List<File> getConfigurations()
	{
		List<File> files = getConfigurationFiles();
		List<ParseConfig> configs = new ArrayList<ParseConfig>();
		for (int i=0; i<files.size(); i++)
		{
			configs.add(new ParseConfig(files.get(i).toString()));
		}
		return files;
	}
	/* Returns a List of Files. */
	static List<File> getConfigurationFiles()
	{
		List<String> names = getConfigurationNames();
		List<File> files = new ArrayList<File>();
		for (int i=0; i<names.size(); i++)
		{
			files.add(new File(System.getProperty("user.home") + "/.ParaProf/" + names.get(i)));
		}
		return files;
	}
	/* Returns a List of Strings. */
	static List<String> getConfigurationNames()
	{
		File paraprofDirectory = new File(System.getProperty("user.home") + "/.ParaProf");
		String[] fileNames = paraprofDirectory.list();
		List<String> perfdmfConfigs  = new ArrayList<String>();
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