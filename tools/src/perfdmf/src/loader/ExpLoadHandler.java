package edu.uoregon.tau.perfdmf.loader;

import edu.uoregon.tau.perfdmf.database.*;

import java.sql.SQLException;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/*** SAX Handler which creates SQL to load a xml document into the database. ***/

public class ExpLoadHandler extends DefaultHandler {

	protected final static String EXP_TABLE = "experiment";
	protected final static String TRIAL_TABLE = "trial";
	
    // w.r.t. applications
    protected String id = "";
    protected String name = "";
    protected String application = "";
    protected String systemName = "";
    protected String systemMachineType = "";
    protected String systemArch = "";
    protected String systemOS = "";
    protected String systemMemorySize = "";
    protected String systemProcessorAmount = "";
    protected String systemL1CacheSize = "";
    protected String systemL2CacheSize = "";
    protected String compilerCppName = "";
    protected String compilerCppVersion = "";
    protected String compilerCcName = "";
    protected String compilerCcVersion = "";
    protected String compilerJavaDirpath = "";
    protected String compilerJavaVersion = "";
    protected String configurePrefix = "";
    protected String configureArch = "";
    protected String configureCpp = "";
    protected String configureCc = "";
    protected String configureJdk = "";
    protected String configureProfile = "";
    protected String systemUserData = "";
    protected String compilerUserData = "";
    protected String configureUserData = "";
    protected String userData = "";

	protected String currentSection = "";
	protected String currentSubSection = "";
	protected String currentSubSubSection = "";
	protected String currentElement = "";
	private DB dbconnector;

public ExpLoadHandler(String application) {
	super();
	this.application = application;
}

public ExpLoadHandler(DB db, String application){
	super();
	this.dbconnector = db;
	this.application = application;
}

public DB getDB() {
	return dbconnector;
}

public String getExpId(){
	return id;
}

/*** Initialize the document table when begining loading a XML document.*/

public void startDocument() throws SAXException{
    // nothing needs to be done here.
}

/*** Handle element, attributes, and the connection from this element to its parent. ***/

public void startElement(String url, String name, String qname, Attributes attrList) throws SAXException {	
	
	if( name.equalsIgnoreCase("system") ) {
		currentSection = name;
		currentSubSection = "";
		currentSubSubSection = "";
	} else if( name.equalsIgnoreCase("hw-spec") ) {
		currentSubSection = name;
		currentSubSubSection = "";
	} else if( name.equalsIgnoreCase("cache-size") ) {
		currentSubSubSection = name;
	} else if( name.equalsIgnoreCase("compiler") ) {
		currentSection = name;
		currentSubSection = "";
		currentSubSubSection = "";
	} else if( name.equalsIgnoreCase("cpp") ) {
		currentSubSection = name;
		currentSubSubSection = "";
		currentElement = name;
	} else if( name.equalsIgnoreCase("cc") ) {
		currentSubSection = name;
		currentSubSubSection = "";
		currentElement = name;
	} else if( name.equalsIgnoreCase("java") ) {
		currentSubSection = name;
		currentSubSubSection = "";
	} else if( name.equalsIgnoreCase("configure") ) {
		currentSection = name;
		currentSubSection = "";
		currentSubSubSection = "";
	} else {
		currentElement = name;
	}       
}

/**
 * Handle character data regions.
 */
public void characters(char[] chars, int start, int length) {
	
	// check if characters is whitespace, if so, return
    
	boolean isWhitespace = true;
	
	for (int i = start; i < start+length; i++) {		
		if (! Character.isWhitespace(chars[i])) {
			isWhitespace = false;
			break;
		}
	}
	if (isWhitespace == true) {
		return;
	}

	String tempstr = new String(chars, start, length);
	
	if (currentElement.equalsIgnoreCase("name")) {
		if (currentSection.equalsIgnoreCase("system")) {
			this.systemName = tempstr;
		} else if (currentSection.equalsIgnoreCase("compiler")) {
			if (currentSubSection.equalsIgnoreCase("cpp")) {
				this.compilerCppName = tempstr;
			} else if (currentSubSection.equalsIgnoreCase("cc")) {
				this.compilerCcName = tempstr;
			}
		} else {
			this.name = tempstr;
		}
	}

	else if (currentElement.equalsIgnoreCase("machine-type"))
		systemMachineType = tempstr;      	
	else if (currentElement.equalsIgnoreCase("arch")) {
		if (currentSection.equalsIgnoreCase("system")) {
			systemArch = tempstr;
		} else if (currentSection.equalsIgnoreCase("configure")) {
			configureArch = tempstr;
		}
	}
	else if (currentElement.equalsIgnoreCase("os")) 
		systemOS = tempstr;
	else if (currentElement.equalsIgnoreCase("memory-size"))
		systemMemorySize = tempstr;
	else if (currentElement.equalsIgnoreCase("processor-amt"))
		systemProcessorAmount = tempstr;
	else if (currentElement.equalsIgnoreCase("L1"))
		systemL1CacheSize = tempstr;
	else if (currentElement.equalsIgnoreCase("L2"))
		systemL2CacheSize = tempstr;

	else if (currentElement.equalsIgnoreCase("version")) {
		if (currentSection.equalsIgnoreCase("compiler")) {
			if (currentSubSection.equalsIgnoreCase("cpp")) {
				this.compilerCppVersion = tempstr;
			} else if (currentSubSection.equalsIgnoreCase("cc")) {
				this.compilerCcVersion = tempstr;
			} else if (currentSubSection.equalsIgnoreCase("java")) {
				this.compilerJavaVersion = tempstr;
			}
		}
	}
	else if (currentElement.equalsIgnoreCase("dirpath"))
		compilerJavaDirpath = tempstr;
	else if (currentElement.equalsIgnoreCase("prefix"))
		configurePrefix = tempstr;
	else if (currentElement.equalsIgnoreCase("cc") && currentSection.equalsIgnoreCase("configure"))
		configureCc = tempstr;
	else if (currentElement.equalsIgnoreCase("cpp") && currentSection.equalsIgnoreCase("configure"))
		configureCpp = tempstr;
	else if (currentElement.equalsIgnoreCase("jdk"))
		configureJdk = tempstr;
	else if (currentElement.equalsIgnoreCase("profile"))
		configureProfile = tempstr;
	else if (currentElement.equalsIgnoreCase("userdata")) {
		if (currentSection.equalsIgnoreCase("system")) {
			this.systemUserData = tempstr;
		} else if (currentSection.equalsIgnoreCase("compiler")) {
			this.compilerUserData = tempstr;
		} else if (currentSection.equalsIgnoreCase("configure")) {
			this.configureUserData = tempstr;
		}
	}
}

public void endElement(String url, String name, String qname) {
	
	if (name.equalsIgnoreCase("experiment")){

	    // check if the application is already stored in DB
	    StringBuffer buf = new StringBuffer();
	    buf.append("select id from  ");
	    buf.append(EXP_TABLE);
	    buf.append("  where name like '" + this.name + "' and application = " + application + "; ");
		// System.out.println(buf.toString());
	    
	    
	    String value = null;
		try {
		    value = getDB().getDataItem(buf.toString());
		} catch (SQLException e) {
		}

	    
	    if (value == null){

	    	buf = new StringBuffer();
	    	buf.append("insert into " + EXP_TABLE);
	    	buf.append(" (name, application, system_name, ");
			buf.append("system_machine_type, system_arch, system_os, ");
			buf.append("system_memory_size, system_processor_amt, ");
			buf.append("system_l1_cache_size, system_l2_cache_size, ");
			buf.append("compiler_cpp_name, compiler_cpp_version, ");
			buf.append("compiler_cc_name, compiler_cc_version, ");
			buf.append("compiler_java_dirpath, compiler_java_version, ");
			buf.append("configure_prefix, configure_arch, configure_cpp, ");
			buf.append("configure_cc, configure_jdk, configure_profile, ");
			buf.append("userdata, system_userdata, ");
			buf.append("compiler_userdata, configure_userdata) values ('");
	    	buf.append(this.name + "', " + application + ", '" 
		       + systemName + "', '" 
			   + systemMachineType + "', '" 
			   + systemArch + "', '" 
			   + systemOS + "', '" 
			   + systemMemorySize + "', '" 
			   + systemProcessorAmount + "', '" 
			   + systemL1CacheSize + "', '" 
			   + systemL2CacheSize + "', '" 
			   + compilerCppName + "', '" 
			   + compilerCppVersion + "', '" 
			   + compilerCcName + "', '" 
			   + compilerCcVersion + "', '" 
			   + compilerJavaDirpath + "', '" 
			   + compilerJavaVersion + "', '" 
			   + configurePrefix + "', '" 
			   + configureArch + "', '" 
			   + configureCpp + "', '" 
			   + configureCc + "', '" 
			   + configureJdk + "', '" 
			   + configureProfile + "', '" 
			   + userData + "', '"
			   + systemUserData + "', '"
			   + compilerUserData + "', '"
			   + configureUserData + "'); ");       
	    	// System.out.println(buf.toString());
	    	try {
		    	getDB().executeUpdate(buf.toString());
		    	buf.delete(0, buf.toString().length());
				if (getDB().getDBType().compareTo("mysql") == 0)
		    		buf.append("select LAST_INSERT_ID();");
				else if (getDB().getDBType().compareTo("db2") == 0)
		    		buf.append("select IDENTITY_VAL_LOCAL() from experiment ");
				else if (getDB().getDBType().compareTo("derby") == 0)
		    		buf.append("select IDENTITY_VAL_LOCAL() from experiment ");
				else
		    		buf.append("select currval('experiment_id_seq');");
		    	id = getDB().getDataItem(buf.toString()); 
		    	System.out.println("The ID for the experiment is: "+ id);
			} catch (SQLException ex) {
		    	ex.printStackTrace();                
			}
	    }
	    else System.out.println("The application has already been loaded.");
	}

}

}

