package edu.uoregon.tau.common;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.StringTokenizer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.HashSet;
import java.lang.Thread;
import java.awt.Component;

import javax.swing.JOptionPane;
import javax.swing.JFileChooser;

/**
 * An External Tool properties management class.  If you want to generate
 * a default external properties file, run the main method, or call the
 * createDefaultTool() method.
 *

Example properties file:
----------------------------------------------------------------

# Properties file for an External Tool
Tool_Name = Paraver

# Program Name
Program_Name = pk2prv

# File type supported
File_Type = Paraver

# each command has an index (one for each request)
Command.0 = timeline.cfg function_name metric_name
Command_Label.0 = View Timeline

# each command has 0 or more parameter names
Parameter_Name.0.0 = function_name
Parameter_Name.0.1 = metric_name

# Another command, no parameters
Command.1 = Do_Something
Command_Label.1 = Something Else

----------------------------------------------------------------

 * 
 * <P>CVS $Id: ExternalTool.java,v 1.4 2009/11/09 18:55:00 khuck Exp $</P>
 * $RCSfile: ExternalTool.java,v $
 * $Date: 2009/11/09 18:55:00 $
 * @author  $Author: khuck $
 * @version $Revision: 1.4 $
 */
public class ExternalTool {
	
	// location of properties files (should be $HOME/.ParaProf)
	private static final String PROPERTIES_LOCATION = System.getProperty("user.home") + 
	  File.separator + ".ParaProf" + File.separator;
	// external tool file prefix
	private static final String PREFIX = "externalTool.";
	// file extention
	private static final String SUFFIX = ".properties";
	
	// Expected properties:
	public static final String TOOL_NAME = "Tool_Name";
	public static final String PROGRAM_NAME = "Program_Name";
	public static final String FILE_TYPE = "File_Type";
	public static final String WORKING_DIRECTORY = "Working_Directory";
	public static final String ENVIRONMENT_VARIABLES = "Environment_Variables";
	public static final String COMMAND = "Command";
	public static final String COMMAND_LABEL = "Command_Label";
	public static final String PARAMETER_NAME = "Parameter_Name";
	public static final String DELIM = ".";
	public static final String FUNCTION_NAME = "function_name";
	public static final String SHORT_FUNCTION_NAME = "short_function_name";
	public static final String METRIC_NAME = "metric_name";
	public static final String METADATA = "metadata_";
	public static final String NODE_ID = "node_ID";
	public static final String THREAD_ID = "thread_ID";
	private static final String HEADER = "Default Properties file for an external tool.";
	
	private static List/*<ExternalTool>*/ loadedTools = null;
	private static String currentTrial = null;
	
	private String propertiesFile = null;  // properties file name
	private Properties properties = null;
	private String toolName = null;
	private String programName = null;
	private String workingDirectory = null;
	private String[] environmentVariables = null;
	private List/*<Command>*/ commands = new ArrayList/*<Command>*/();

	/** 
	 * Constructor.  This function will read the properties from the specified
	 * properties file, and return a new ExternalTool object.
	 * 
	 * @param propertiesFile
	 */
	public ExternalTool (String propertiesFile) {
		this.propertiesFile = propertiesFile;
		this.properties = new Properties();
		File file = new File(propertiesFile);
		if (file.exists()) {
			try {
				this.properties.load(new FileInputStream(this.propertiesFile));
				processProperties();
			} catch (FileNotFoundException e) {
				System.err.println(e.getMessage());
			} catch (IOException e) {
				System.err.println(e.getMessage());
			}
			this.toolName = this.properties.getProperty(TOOL_NAME, null);
			this.programName = this.properties.getProperty(PROGRAM_NAME, null);
			this.workingDirectory = this.properties.getProperty(WORKING_DIRECTORY, null);
			String tmp = this.properties.getProperty(ENVIRONMENT_VARIABLES, null);
			if (tmp != null) {
				StringTokenizer st = new StringTokenizer(tmp, ":");
				this.environmentVariables = new String[st.countTokens()];
				int index = 0;
				while (st.hasMoreTokens()) {
					this.environmentVariables[index++] = st.nextToken();
				}
			}
		}
	}
	
	/**
	 * Iterate over the commands in the properties file, and save the data.
	 */
	private void processProperties() {
		// iterate over the commands
		for (int c = 0 ; true ; c++) {
			String commandString = properties.getProperty(COMMAND + DELIM + c, null);
			// if there are no more commands, exit
			if (commandString == null) {
				break;
			}
			Command command = new Command(this, commandString);
			command.label = properties.getProperty(COMMAND_LABEL + DELIM + c, commandString);
			this.commands.add(command);
			// iterate over the parameters, and for each one get name and type
			for (int p = 0 ; true ; p++) {
				String parameterName = properties.getProperty(PARAMETER_NAME + DELIM + c + DELIM + p, null);				
				if (parameterName == null) {
					break;
				}
				command.parameterNames.add(parameterName);
			}
		}
	}

	/**
	 * Search for all the external tool properties files in the .ParaProf
	 * directory, and create ExternalTool objects for each one.
	 * 
	 * @return List list of external tools
	 */
	public static List/*<ExternalTool>*/ loadAllTools() {
		if (ExternalTool.loadedTools == null) {
			loadedTools = new ArrayList/*<ExternalTool>*/();
			File directory = new File(PROPERTIES_LOCATION);
			if (directory.isDirectory()) {
				File[] files = directory.listFiles();
				for (int i = 0 ; i < files.length ; i++) {
					File file = files[i];
					if (file.getName().startsWith(PREFIX) && file.getName().endsWith(SUFFIX)) {
						ExternalTool tool = new ExternalTool(file.getPath());
						//System.out.println("Loaded properties file: " + tool.getPropertiesFile());
						loadedTools.add(tool);
					}
				}
			}
		}
		return loadedTools;
	}

	/**
	 * Search for all the external tool properties files in the .ParaProf
	 * directory, and create ExternalTool objects for each one.
	 * 
	 * @return List list of external tools
	 */
	public static List/*<ExternalTool>*/ loadAllTools(String trialName) {
		if (ExternalTool.currentTrial != null && !ExternalTool.currentTrial.equals(trialName)) {
			ExternalTool.loadedTools = null;
		}
		ExternalTool.currentTrial = trialName;
		return loadAllTools();
	}
	
	/**
	 * Reload the external tools.
	 * 
	 * @return List list of external tools
	 */
	public static List/*<ExternalTool>*/ reloadTools() {
		ExternalTool.loadedTools = null;
		return ExternalTool.loadAllTools();
	}
	
	/**
	 * Check to see that there is a configured external tool which supports
	 * this profile file type.
	 * 
	 * @param fileType
	 * @return
	 */
	public static List/*<ExternalTool>*/ findMatchingTools(String fileType) {
		return findMatchingTools(fileType, null);
	}


	/**
	 * Check to see that there is a configured external tool which supports
	 * this profile file type.
	 * 
	 * @param fileType
	 * @param trialName
	 * @return
	 */
	public static List/*<ExternalTool>*/ findMatchingTools(String fileType, String trialName) {
		List/*<ExternalTool>*/ tools = new ArrayList/*<ExternalTool>*/();
		ExternalTool.loadAllTools(trialName);
		if (ExternalTool.loadedTools == null)
			return tools;
		for (Iterator iter = ExternalTool.loadedTools.iterator() ; iter.hasNext() ;) {
			ExternalTool tool = (ExternalTool)iter.next();
			String configuredType = tool.properties.getProperty(ExternalTool.FILE_TYPE, "None"); 
			if (configuredType.equalsIgnoreCase(fileType)) {
				tools.add(tool);
			}
		}
		return tools;
	}
	
	/**
	 * Check to see that there is a configured external tool which supports
	 * this profile file type.
	 * 
	 * @param fileType
	 * @return
	 */
	public static boolean matchingToolExists(String fileType, String trialName) {
		if (findMatchingTools(fileType, trialName).isEmpty()) 
			return false;
		return true;
	}

	/**
	 * Check to see that there is a configured external tool which supports
	 * this profile file type.
	 * 
	 * @param fileType
	 * @return
	 */
	public static boolean matchingToolExists(String fileType) {
		if (findMatchingTools(fileType).isEmpty()) 
			return false;
		return true;
	}
	
	public String getProperty(String key) {
		String prop = properties.getProperty(key, "Undefined");
		return prop;
	}
	
	public String dump() {
		StringBuffer buf = new StringBuffer();
		for (Iterator iter2 = properties.keySet().iterator(); iter2.hasNext(); ) {
			String key = (String)iter2.next();
			buf.append("\t" + key + " = " + properties.getProperty(key) + "\n");
		}
		
		return buf.toString();
//		return properties.toString();
	}

	public String getPropertiesFile() {
		return propertiesFile;
	}

	public void setPropertiesFile(String propertiesFile) {
		this.propertiesFile = propertiesFile;
	}

	public static void launch(List/*<ExternalTool>*/ tools, CommandParameters params, Component parentWindow) {
		// the mean and standard deviation IDs from ParaProf will be less than
		// zero.  Fix that, if necessary.  This should not be called from mean
		// or standard deviation thread, but just in case...
    	params.nodeID = (params.nodeID < 0) ? 0 : params.nodeID;
    	params.threadID = (params.threadID < 0) ? 0 : params.threadID;

		List/*<Command>*/ commands = new ArrayList/*<Command>*/();
		for (Iterator iter = tools.iterator() ; iter.hasNext() ; ) {
			ExternalTool tool = (ExternalTool)iter.next();
			commands.addAll(tool.commands);
		}
    	
    	// show a list of commands to run in the external tool
    	Object[] options = commands.toArray();
		Object obj = JOptionPane.showInputDialog (parentWindow,
				"Select a command for the external tool:",
				"External Tool Commands",
				JOptionPane.PLAIN_MESSAGE,
				null,
				options,
				options[0]);
		
		// If the user canceled, do nothing.
		if (obj == null)
			return;
		
		// build the external command
		Command command = (Command) obj;
		String commandString = command.tool.programName + " " + command.name;
		for (Iterator iter = command.parameterNames.iterator() ; iter.hasNext() ; ) {
			String pName = (String)iter.next();
			if (pName.equals(FUNCTION_NAME)) {
				// remove callpath
				String leaf = removeCallpath(params.function);
				commandString = commandString.replaceAll(pName, leaf);
			} else if (pName.equals(SHORT_FUNCTION_NAME)) {
				String leaf = removeCallpath(params.function);
				commandString = commandString.replaceAll(pName, Utility.shortenFunctionName(leaf));
			} else if (pName.equals(METRIC_NAME)) {
				commandString = commandString.replaceAll(pName, params.metric);
			} else if (pName.equals(NODE_ID)) {
				commandString = commandString.replaceAll(pName, Integer.toString(params.nodeID));
			} else if (pName.equals(THREAD_ID)) {
				commandString = commandString.replaceAll(pName, Integer.toString(params.threadID));
			} else if (pName.startsWith(METADATA)) {
				String fieldName = pName.replaceFirst(METADATA, "");
				String tmp = (String)params.metadata.get(fieldName);
				if (tmp != null)
					commandString = commandString.replaceAll(pName, tmp);
			}
		}

		if (command.tool.workingDirectory.equalsIgnoreCase("%PROMPT%")) {
			// prompt the user for the working directory
            JFileChooser fc = new JFileChooser(System.getProperty("user.dir"));
			fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
			fc.setDialogTitle("Choose the gnuplot files directory");
        	int returnVal = fc.showOpenDialog(parentWindow);
        	if (returnVal == JFileChooser.APPROVE_OPTION) {
            	command.tool.workingDirectory = fc.getSelectedFile().getAbsolutePath();
			} else {
				return;
			}
       	}
		ToolRunner tool = new ToolRunner(command.tool.workingDirectory, command.tool.environmentVariables, commandString);
	}
	
	private static String removeCallpath(String function) {
		StringTokenizer st = new StringTokenizer(function, "=>");
		String leafNode = null;
		String tmp = null;
		// get the last one.
		while(st.hasMoreTokens()) {
			tmp = st.nextToken();
			// we want to ignore samples at the end of the callpath
			if (tmp.indexOf(':') < 0 && tmp.indexOf(File.separatorChar) < 0) {
				leafNode = tmp;
			}
		}
		return leafNode.trim();
	}

	public static void createDefaultTool(boolean overwrite) {
		String fileName = PROPERTIES_LOCATION + PREFIX + "default" + SUFFIX;
		
		File dummy = new File(fileName);
		if (!dummy.exists() || overwrite) {
			ExternalTool tool = new ExternalTool(fileName);
			tool.properties.setProperty(TOOL_NAME, "Some Performance Tool");
			tool.properties.setProperty(PROGRAM_NAME, "echo");  // command line application name
			tool.properties.setProperty(FILE_TYPE, "PPK");  // Some file type which matches DataSource file types (String)
			tool.properties.setProperty(WORKING_DIRECTORY, "");  // The working directory for the program
			tool.properties.setProperty(ENVIRONMENT_VARIABLES, "");  // Environment variables which need to be set for the program
			tool.properties.setProperty(COMMAND + DELIM + "0", "my_command");  //The first command available in the program
			tool.properties.setProperty(COMMAND_LABEL + DELIM + "0", "Echo Parameters");  //A user-friendly label for the user to select for this command
			tool.properties.setProperty(PARAMETER_NAME + DELIM + "0" + DELIM + "0", FUNCTION_NAME);
			tool.properties.setProperty(PARAMETER_NAME + DELIM + "0" + DELIM + "1", METRIC_NAME);
			tool.properties.setProperty(PARAMETER_NAME + DELIM + "0" + DELIM + "2", NODE_ID);
			tool.properties.setProperty(PARAMETER_NAME + DELIM + "0" + DELIM + "3", THREAD_ID);
			OutputStream out;
			try {
				out = new FileOutputStream(fileName);
				tool.properties.store(out, HEADER);
			} catch (IOException e) {
				System.err.println(e.getMessage());
				e.printStackTrace();
			}
		}
	}
	
	public static void main(String[] args) {
		ExternalTool.createDefaultTool(false);
		List tools = ExternalTool.loadAllTools();
		for (Iterator iter = tools.iterator(); iter.hasNext(); ) {
			ExternalTool tool = (ExternalTool)iter.next();
			System.out.println("Properties: ");
			System.out.println(tool.dump());
		}
		Component mainWindow = null;
		ExternalTool.CommandParameters params = new ExternalTool.CommandParameters();
		params.function = "function";
		params.metric = "metric";
		params.nodeID = 0;
		params.threadID = 0;
		ExternalTool.launch(tools, params, mainWindow);
	} 
	
	class Command {
		public String name = null;
		public String label = null;
		public List/*<String>*/ parameterNames = new ArrayList/*<String>*/();
		public ExternalTool tool = null;
		Command(ExternalTool tool, String name) {
			this.tool = tool;
			this.name = name;
		}
		public String toString() {
			return tool.toolName + ": " + label;
		}
	}
	
	public static class CommandParameters {
		public String function = null; 
		public String metric = null;
		public int nodeID = 0;
		public int threadID = 0;
		public Map metadata = null;
	}
}

	class ToolRunner extends Thread {
		private String commandString = null;
		private String workingDirectory = null;
		private String[] environmentVariables = null;
		
		ToolRunner(String commandString) {
			this.commandString = commandString;
			this.start();
		}

		ToolRunner(String workingDirectory, String[] environmentVariables, String commandString) {
			this.commandString = commandString;
			this.workingDirectory = workingDirectory;
			this.environmentVariables = environmentVariables;
			this.start();
		}
		
		public void run() {
			// run the command
			Runtime r = Runtime.getRuntime();
			try {
				File dir = null;
				if (workingDirectory != null) {
					dir = new File(workingDirectory);
					System.out.println("Working directory: " + workingDirectory);
				}
				String[] envp = null;
				System.out.println(commandString);
				Process p = r.exec(commandString, envp, dir);
				
				// get the output stream (for some reason named "input" stream
				InputStream in = p.getInputStream();
				BufferedInputStream buf = new BufferedInputStream(in);
				InputStreamReader inread = new InputStreamReader(buf);
				BufferedReader bufferedreader = new BufferedReader(inread);

				// get the error stream
				InputStream err = p.getErrorStream();
				BufferedInputStream eBuf = new BufferedInputStream(err);
				InputStreamReader eInread = new InputStreamReader(eBuf);
				BufferedReader eBufferedreader = new BufferedReader(eInread);

				// Read the output
				String line;
				while ((line = bufferedreader.readLine()) != null) {
					System.out.println("External Tool Output: " + line);
				}
				// Read the errors
				while ((line = eBufferedreader.readLine()) != null) {
					System.err.println("External Tool Error: " + line);
				}
				
				// this will block the TAU tool until the external tool finishes.
				// If we retain this behavior, we should spawn a Java thread to make this call.
				try {
					if (p.waitFor() != 0) {
					    System.err.println("External Tool Exit Value = " + p.exitValue());
					}
				} catch (InterruptedException e) {
				    System.err.println(e.getMessage());
				} finally {
					// Close the InputStream
					eBufferedreader.close();
					eInread.close();
					eBuf.close();
					err.close();
					// Close the InputStream
					bufferedreader.close();
					inread.close();
					buf.close();
					in.close();
				}
				
				if (p.exitValue() == 0) {
					// System.out.println("Program exited normally.");
				} else {
					// what should we do?
				}
			} catch (IOException e) {
				System.err.println(e.getMessage());
				e.printStackTrace();
			}
		}
	}

