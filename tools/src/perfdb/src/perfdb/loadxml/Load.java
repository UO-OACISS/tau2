package perfdb.loadxml;

import perfdb.util.dbinterface.*;
import java.sql.*;
import org.xml.sax.*; 
import org.xml.sax.helpers.*; 
import org.w3c.dom.Document;
import java.io.*;

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

    public LoadHandler newHandler() {
	
	return new LoadHandler(getDB()); // no DB partition.
	
    }

    public AppLoadHandler newAppHandler() {
	return new AppLoadHandler(getDB());
    }

    /*** Parse an XML file related to a trial using a SAX parser
	 Note: the parser in <parserClass> MUST be included in the Java CLASSPATH. ***/

    public String parse(String xmlFile) {
	
	try {
	    
	    XMLReader xmlreader = XMLReaderFactory.createXMLReader(parserClass);	    
	    
	    DefaultHandler handler = this.newHandler();
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
	} catch (SAXException ex) {
	    ex.printStackTrace();
	} catch (NullPointerException ex) {
	    ex.printStackTrace();
	}
	return null;
    }            
             
    /*** Parse a xml file related to an application. ***/

    public String parseApp(String appFile) {
	
	try {
	    XMLReader xmlreader = XMLReaderFactory.createXMLReader(parserClass);
	    
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
	} catch (SAXException ex) {
	    ex.printStackTrace();
	} catch (NullPointerException ex) {
	    ex.printStackTrace();
	}
	return null;
    }       

    /*** look up the record for an appliaction, if there is, return appID ***/

    public String lookupApp(String name, String version){
	StringBuffer buf = new StringBuffer();
	buf.append("select distinct appid from ");
	buf.append("applications ");
	if (version.trim().length()==0) {
	    buf.append("  where AppName='" + name.trim() + "'; ");
	}
	else buf.append("  where AppName='" + name.trim() + "' and version='" + version.trim() + "'; ");

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
	
	buf.append("select distinct expid from ");
	buf.append(exptable);
	buf.append("  where appid='" + appid.trim() + "' and sysinfo='" + sysinfo.trim() + "' and configinfo='" + configinfo.trim() + "' and instruinfo='" + instruinfo.trim() +"' and compilerinfo='" + compilerinfo.trim() + "'; ");

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

    /*** insert an experiment record into PerfDB ***/

    public String insertExp(String exptable, String appid, String sys, String config, String compiler, String instru, String defValue){
	String expid;
	StringBuffer buf = new StringBuffer();

	try {
	    buf.append("insert into ");
	    buf.append(exptable);
	    if (defValue==null)
		buf.append(" (appid, sysinfo, configinfo, instruinfo, compilerinfo)");
	    else buf.append(" (appid, sysinfo, configinfo, instruinfo, compilerinfo, trial_table_name)");

	    buf.append(" values ");

	    if (defValue==null)
		buf.append("(" + appid  + ", '" + sys   
		       + "', '" + config + "', '" + instru + "', '" + compiler + "'); ");  
	    else 
		buf.append("(" + appid  + ", '" + sys + "', '" + config + "', '" + instru + "', '" + compiler + "', '" + defValue + "'); "); 

	    
	    getDB().executeUpdate(buf.toString());	    
	    buf.delete(0, buf.toString().length());
	    buf.append("select currval('experiments_expid_seq');");
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
}




