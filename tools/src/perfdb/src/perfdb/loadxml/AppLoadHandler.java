package perfdb.loadxml;

import perfdb.util.dbinterface.*;
import org.xml.sax.*;
import org.xml.sax.helpers.*;
import java.util.*;
import java.sql.*;

/*** SAX Handler which creates SQL to load a xml document into the database. ***/

public class AppLoadHandler extends DefaultHandler {

	protected final static String APP_TABLE = "Applications";
	protected final static String EXP_TABLE = "Experiments";
	
    // w.r.t. applications
        protected String appid = "";
        protected String appname = "";
        protected String version = "";
        protected String desc = "";
        protected String lang = "";
        protected String paradiag = "";
        protected String usage = "";
        protected String exeopt = "";

	protected String currentElement = "";
	private DB dbconnector;

public AppLoadHandler() {
	super();
}

public AppLoadHandler(DB db){
	super();
	this.dbconnector = db;
}

public DB getDB() {
	return dbconnector;
}

public String getAppId(){
	return appid;
}

/*** Initialize the document table when begining loading a XML document.*/

public void startDocument() throws SAXException{
    // nothing needs to be done here.
}

/*** Handle element, attributes, and the connection from this element to its parent. ***/

/*public String metricAttrToString(AttributeList attrList) {
        StringBuffer buf = new StringBuffer();
        int length = attrList.getLength();
        for (int i=0;i<length;i++) {
                buf.append(attrList.getValue(i));
        }
        return buf.toString();
}*/

public void startElement(String url, String name, String qname, Attributes attrList) throws SAXException {	
	
	 if( name.equalsIgnoreCase("name") ) {
	     currentElement = "appname";
	 }
         if( name.equalsIgnoreCase("description") ) {
	     currentElement = "desc";
	     }
         if( name.equalsIgnoreCase("version") ) {
	     currentElement = "version";
	 }
	 if( name.equalsIgnoreCase("language") ) {
	     currentElement = "lang";
	 }
	 if( name.equalsIgnoreCase("para_diag") ) {
	     currentElement = "paradiag";
	 }
	 if( name.equalsIgnoreCase("usage") ) {
	     currentElement = "usage";
	 }
	 if( name.equalsIgnoreCase("exe_opt") ) {
	     currentElement = "exeopt";
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
	
	if (currentElement.equals("appname")) appname = tempstr;
	 
	if (currentElement.equals("desc")) desc = tempstr;      	

	if (currentElement.equals("version")) version = tempstr;
	 
	if (currentElement.equals("lang")) lang = tempstr;
	
	if (currentElement.equals("paradiag")) paradiag = tempstr;
	
	if (currentElement.equals("usage")) usage = tempstr;
	
	if (currentElement.equals("exeopt")) exeopt = tempstr;
	
}

public void endElement(String url, String name, String qname) {
	
	if (name.equalsIgnoreCase("application")){

	    // check if the application is already stored in DB
	    StringBuffer buf = new StringBuffer();
	    buf.append("select appid from  ");
	    buf.append(APP_TABLE);
	    buf.append("  where AppName='" + appname + "' and version='" + version + "'; ");
	    if (getDB().getDataItem(buf.toString()) == null){

	    	buf = new StringBuffer();
	    	buf.append("insert into");
	    	buf.append(" " + APP_TABLE + " ");
	    	buf.append("(AppName, Version, Description, Language, Paradigm, UsageText, Exe_opt)");
	    	buf.append(" values ");
	    	buf.append("('" + appname + "', '" + version + "', '" 
		       + desc + "', '" + lang + "', '" + paradiag + "', '" + usage + "', '" + exeopt + "'); ");       
	    	// System.out.println(buf.toString());
	    	try {
		    getDB().executeUpdate(buf.toString());
		    buf.delete(0, buf.toString().length());
			if (getDB().getDBType().compareTo("mysql") == 0)
		    	buf.append("select LAST_INSERT_ID();");
			else
		    	buf.append("select currval('applications_appid_seq');");
		    appid = getDB().getDataItem(buf.toString()); 
		    System.out.println("The ID for the application is: "+ appid);
		} catch (SQLException ex) {
		    ex.printStackTrace();                
		}


	    }
	    else System.out.println("The application has already been loaded.");
	}

}

}

