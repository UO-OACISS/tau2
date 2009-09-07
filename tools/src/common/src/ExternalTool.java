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
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.lang.Thread;
import java.awt.Component;

import javax.swing.JOptionPane;

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
 * <P>CVS $Id: ExternalTool.java,v 1.2 2009/09/07 09:51:26 khuck Exp $</P>
 * $RCSfile: ExternalTool.java,v $
 * $Date: 2009/09/07 09:51:26 $
 * @author  $Author: khuck $
 * @version $Revision: 1.2 $
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
	public static final String COMMAND = "Command";
	public static final String COMMAND_LABEL = "Command_Label";
	public static final String PARAMETER_NAME = "Parameter_Name";
	public static final String DELIM = ".";
	public static final String FUNCTION_NAME = "function_name";
	public static final String METRIC_NAME = "metric_name";
	public static final String PROCESS_ID = "process_ID";
	public static final String THREAD_ID = "thread_ID";
	private static final String HEADER = "Default Properties file for an external tool.";
	
	
	private static List/*<ExternalTool>*/ loadedTools = null;
	
	private String propertiesFile = null;  // properties file name
	private Properties properties = null;
	private String toolName = null;
	private String programName = null;
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
			this.toolName = new String(this.properties.getProperty(TOOL_NAME, null));
			this.programName = new String(this.properties.getProperty(PROGRAM_NAME, null));
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
						System.out.println("Loaded properties file: " + tool.getPropertiesFile());
						loadedTools.add(tool);
					}
				}
			}
		}
		return loadedTools;
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
		List/*<ExternalTool>*/ tools = new ArrayList/*<ExternalTool>*/();
		ExternalTool.loadAllTools();
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

	public static void launch(List/*<ExternalTool>*/ tools, String function, String metric, int nodeID, int threadID, Component parentWindow) {
    	// the mean and standard deviation IDs from ParaProf will be less than zero.  Fix that, if necessary.
    	nodeID = (nodeID < 0) ? 0 : nodeID;
    	threadID = (threadID < 0) ? 0 : threadID;

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
				commandString = commandString.replaceAll(FUNCTION_NAME, function);
			} else if (pName.equals(METRIC_NAME)) {
				commandString = commandString.replaceAll(METRIC_NAME, metric);
			} else if (pName.equals(PROCESS_ID)) {
				commandString = commandString.replaceAll(PROCESS_ID, Integer.toString(nodeID));
			} else if (pName.equals(THREAD_ID)) {
				commandString = commandString.replaceAll(THREAD_ID, Integer.toString(threadID));
			}
		}
		
		ToolRunner tool = new ToolRunner(commandString);
	}
	
	public static void createDefaultTool(boolean overwrite) {
		String fileName = PROPERTIES_LOCATION + PREFIX + "default" + SUFFIX;
		
		File dummy = new File(fileName);
		if (!dummy.exists() || overwrite) {
			ExternalTool tool = new ExternalTool(fileName);
			tool.properties.setProperty(TOOL_NAME, "Some Performance Tool");
			tool.properties.setProperty(PROGRAM_NAME, "echo");  // command line application name
			tool.properties.setProperty(FILE_TYPE, "PPK");  // Some file type which matches DataSource file types (String)
			tool.properties.setProperty(COMMAND + DELIM + "0", "my_command");  //The first command available in the program
			tool.properties.setProperty(COMMAND_LABEL + DELIM + "0", "Echo Parameters");  //A user-friendly label for the user to select for this command
			tool.properties.setProperty(PARAMETER_NAME + DELIM + "0" + DELIM + "0", FUNCTION_NAME);
			tool.properties.setProperty(PARAMETER_NAME + DELIM + "0" + DELIM + "1", METRIC_NAME);
			tool.properties.setProperty(PARAMETER_NAME + DELIM + "0" + DELIM + "2", PROCESS_ID);
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
		ExternalTool.launch(tools, "function", "metric", 0, 0, mainWindow);
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
			return tool.programName + ": " + label;
		}
	}
	
}

	class ToolRunner extends Thread {
		private String commandString = null;
		
		ToolRunner(String commandString) {
			this.commandString = commandString;
			this.start();
		}
		
		public void run() {
			// run the command
			Runtime r = Runtime.getRuntime();
			try {
				//System.out.println(commandString);
				Process p = r.exec(commandString);
				
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
					System.out.println(line);
				}
				// Read the errors
				while ((line = eBufferedreader.readLine()) != null) {
					System.err.println(line);
				}
				
				// this will block the TAU tool until the external tool finishes.
				// If we retain this behavior, we should spawn a Java thread to make this call.
				try {
					if (p.waitFor() != 0) {
					    System.err.println("exit value = " + p.exitValue());
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
					System.out.println("Program exited normally.");
				} else {
					// what should we do?
				}
			} catch (IOException e) {
				System.err.println(e.getMessage());
				e.printStackTrace();
			}
		}
	}

