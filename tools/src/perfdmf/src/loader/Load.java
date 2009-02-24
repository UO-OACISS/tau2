package edu.uoregon.tau.perfdmf.loader;

import java.io.*;
import java.sql.ResultSet;
import java.sql.SQLException;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.DefaultHandler;
import org.xml.sax.helpers.XMLReaderFactory;

import edu.uoregon.tau.perfdmf.database.DB;

/** For applications or experiments, Load checks if there is a duplicate, 
    if so, reject loading. Otherwise, load them into database.  For trials, 
    Load invokes LoadHandler to parse and then load them. **/

public class Load {

    private DB db = null;
    private String parserClass;

    public Load(String parserClassIn) {
	super();
	parserClass = new String(parserClassIn);
    }

    public Load(DB db, String parserClassIn) {
	super();
	parserClass = new String(parserClassIn);
	setDB(db);
    }	

    public DB getDB() {
        return db;
    }

    public void setDB(DB newValue) {
        this.db = newValue;
    }

    public LoadHandler newHandler(String trialId, String problemFile) {
		return new LoadHandler(getDB(), trialId, problemFile); 
    }

    public AppLoadHandler newAppHandler() {
		return new AppLoadHandler(getDB());
    }

    public ExpLoadHandler newExpHandler(String application) {
		return new ExpLoadHandler(getDB(), application);
    }

    
    private XMLReader getXMLReader(String parserClass){
    	XMLReader parser;
        
    	try{//Passed in
    		parser = XMLReaderFactory.createXMLReader(parserClass);
    	}catch(SAXException e0){
    	
    	  try { // Xerces
    	    parser = XMLReaderFactory.createXMLReader(
    	     "org.apache.xerces.parsers.SAXParser"
    	    );
    	  }
    	  catch (SAXException e1) {
    	    try { // Crimson
    	      parser = XMLReaderFactory.createXMLReader(
    	       "org.apache.crimson.parser.XMLReaderImpl"
    	      );
    	    }
    	    catch (SAXException e2) { 
    	      try { // Ælfred
    	        parser = XMLReaderFactory.createXMLReader(
    	         "gnu.xml.aelfred2.XmlReader"
    	        );
    	      }
    	      catch (SAXException e3) {
    	        try { // Piccolo
    	          parser = XMLReaderFactory.createXMLReader(
    	            "com.bluecast.xml.Piccolo"
    	          );
    	        }
    	        catch (SAXException e4) {
    	          try { // Oracle
    	            parser = XMLReaderFactory.createXMLReader(
    	              "oracle.xml.parser.v2.SAXParser"
    	            );
    	          }
    	          catch (SAXException e5) {
    	            try { // default
    	              parser = XMLReaderFactory.createXMLReader();
    	            }
    	            catch (SAXException e6) {
    	              throw new NoClassDefFoundError(
    	                "No SAX parser is available");
    	            }
    	          }
    	        }
    	      }
    	    } 
    	  }
    	}
    	return parser;
    }
    
    
    /*** Parse an XML file related to a trial using a SAX parser
	 Note: the parser in <parserClass> MUST be included in the Java CLASSPATH. ***/

    public String parse(String xmlFile, String trialid, String problemFile) {
	
	try {
	    
		// put the problemFile into a string
		String problemDefinition = getProblemString(problemFile);
	    XMLReader xmlreader = getXMLReader(parserClass);
	    
	    DefaultHandler handler;
	   	handler = this.newHandler(trialid, problemDefinition);
	    xmlreader.setContentHandler(handler);
	    xmlreader.setErrorHandler(handler);
	    try {
			((LoadHandler) handler).setDocumentName(xmlFile);
			File file = new File(xmlFile);
			xmlreader.parse(new InputSource(new FileInputStream(file)));
			return ((LoadHandler) handler).getTrialId();
	    } catch (SAXException saxe) {
			saxe.printStackTrace();
	    } catch (IOException ioe) {
			ioe.printStackTrace();
	    }
	} catch (NullPointerException ex) {
	    ex.printStackTrace();
	}
	return null;
    }            
             
    /*** Parse a xml file related to an application. ***/

    public String parseApp(String appFile) {
	
	try {
	    XMLReader xmlreader = getXMLReader(parserClass);
	    
	    DefaultHandler handler = this.newAppHandler();
	    xmlreader.setContentHandler(handler);
	    xmlreader.setErrorHandler(handler);
	    
	    try {
		File file = new File(appFile);
		xmlreader.parse(new InputSource(new FileInputStream(file)));
		return ((AppLoadHandler) handler).getAppId();	
	    } catch (SAXException saxe) {
		saxe.printStackTrace();
	    } catch (IOException ioe) {
		ioe.printStackTrace();
	    }
	} catch (NullPointerException ex) {
	    ex.printStackTrace();
	}
	return null;
    }       

    /*** Parse a xml file related to an experiment. ***/

    public String parseExp(String expFile, String application) {
	
	try {
	    XMLReader xmlreader = getXMLReader(parserClass);
	    
	    DefaultHandler handler = this.newExpHandler(application);
	    xmlreader.setContentHandler(handler);
	    xmlreader.setErrorHandler(handler);
	    
	    try {
		File file = new File(expFile);
		xmlreader.parse(new InputSource(new FileInputStream(file)));
		return ((ExpLoadHandler) handler).getExpId();	
	    } catch (SAXException saxe) {
		saxe.printStackTrace();
	    } catch (IOException ioe) {
		ioe.printStackTrace();
	    }
	} catch (NullPointerException ex) {
	    ex.printStackTrace();
	}
	return null;
    }       

    /*** look up the record for an appliaction, if there is, return appID ***/

    public String lookupApp(String name, String version){
	StringBuffer buf = new StringBuffer();
	buf.append("select distinct id from ");
	buf.append("application ");
	if (version.trim().length()==0) {
	    buf.append("  where name='" + name.trim() + "'; ");
	}
	else buf.append("  where name='" + name.trim() + "' and version='" + version.trim() + "'; ");

	try {
	    ResultSet appId = getDB().executeQuery(buf.toString());	
	    if (appId.next() == false){
		System.out.println("no such application found");
		appId.close();		
		return null;
	    }
	    else {
		String str = appId.getString(1);
		appId.close(); 
		return str;
	    }
	}catch (Exception ex) {
	    ex.printStackTrace();
	    return null;
	}
    }    

    /*** look up the record for an experiment, if there is, return expID ***/

    public String lookupExp(String exptable, String appid, String sysinfo, String configinfo, String compilerinfo, String instruinfo){
	StringBuffer buf = new StringBuffer();
	
	buf.append("select distinct id from ");
	buf.append(exptable);
	buf.append("  where application = '" + appid.trim() + "' and system_info='" + sysinfo.trim() + "' and configuration_info='" + configinfo.trim() + "' and instrumentation_info='" + instruinfo.trim() +"' and compiler_info='" + compilerinfo.trim() + "'; ");

	try {
	    ResultSet expId = getDB().executeQuery(buf.toString());	
	    if (expId.next() == false){			
		expId.close();
		return null;
	    }
	    else {
		String str = expId.getString(1);
		expId.close(); 
		return str;
	    }
	}catch (Exception ex) {
	    ex.printStackTrace();
	    return null;
	}
    }    

    public String lookupTrial(String trialTable, String trialid) {
		StringBuffer buf = new StringBuffer();
	
		buf.append("select distinct id from ");
		buf.append(trialTable);
		buf.append("  where id = " + trialid.trim() + "; ");

		try {
	    	ResultSet expId = getDB().executeQuery(buf.toString());	
	    	if (expId.next() == false){			
				expId.close();
				return null;
	    	} else {
				String str = expId.getString(1);
				expId.close(); 
				return str;
	    	}
		} catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
    }    

    /*** insert an experiment record into PerfDMF ***/

    public String insertExp(String exptable, String appid, String sys, String config, String compiler, String instru, String defValue){
	String expid;
	StringBuffer buf = new StringBuffer();

	try {
	    buf.append("insert into ");
	    // buf.append(exptable);
	    buf.append("experiment ");
		buf.append(" (application, system_info, configuration_info, instrumentation_info, compiler_info)");
	    buf.append(" values ");
		buf.append("(" + appid  + ", '" + sys + "', '");
		buf.append(config + "', '" + instru + "', '" + compiler + "'); ");  
	    
	    getDB().executeUpdate(buf.toString());	    
	    buf.delete(0, buf.toString().length());
		if (getDB().getDBType().compareTo("mysql") == 0)
	    	buf.append("select LAST_INSERT_ID();");
		else
	    	buf.append("select currval('experiment_id_seq');");
	    expid = getDB().getDataItem(buf.toString());
	    System.out.println("The ID for the experiment is: "+ expid);	
	    return expid;
	} catch (SQLException ex) {
	    ex.printStackTrace();
       	} catch (Exception ex) {
	    ex.printStackTrace();
	}
	return null;
    } 

	public String getProblemString(String problemFile) {
		// if the file wasn't passed in, this is an existing trial.
		if (problemFile == null)
			return new String("");

		// open the file
		BufferedReader reader = null;
		try {
			reader = new BufferedReader (new FileReader (problemFile));
		} catch (Exception e) {
			System.out.println("Problem file not found!  Exiting...");
			System.exit(0);
		}
		// read the file, one line at a time, and do some string
		// substitution to make sure that we don't blow up our
		// SQL statement.  ' characters aren't allowed...
		StringBuffer problemString = new StringBuffer();
		String line;
		while (true) {
			try {
				line = reader.readLine();
			} catch (Exception e) {
				line = null;
			}
			if (line == null) break;
			problemString.append(line.replaceAll("'", "\'"));
		}

		// close the problem file
		try {
			reader.close();
		} catch (Exception e) {
		}

		// return the string
		return problemString.toString();
	}
}




