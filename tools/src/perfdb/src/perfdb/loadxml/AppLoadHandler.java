package perfdb.loadxml;

import java.sql.SQLException;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import perfdb.util.dbinterface.DB;

/*** SAX Handler which creates SQL to load a xml document into the database. ***/

public class AppLoadHandler extends DefaultHandler {

	protected final static String APP_TABLE = "application";
	protected final static String EXP_TABLE = "experiment";
	
    // w.r.t. applications
    protected String appid = "";
    protected String name = "";
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

public void startElement(String url, String name, String qname, Attributes attrList) throws SAXException {	
	
	if( name.equalsIgnoreCase("name") ) {
		currentElement = "name";
	} else if( name.equalsIgnoreCase("description") ) {
		currentElement = "description";
	} else if( name.equalsIgnoreCase("version") ) {
		currentElement = "version";
	} else if( name.equalsIgnoreCase("language") ) {
		currentElement = "language";
	} else if( name.equalsIgnoreCase("para_diag") ) {
		currentElement = "paradiagm";
	} else if( name.equalsIgnoreCase("usage") ) {
		currentElement = "usage";
	} else if( name.equalsIgnoreCase("exe_opt") ) {
		currentElement = "execution_options";
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
	
	if (currentElement.equals("name")) this.name = tempstr;
	else if (currentElement.equals("description")) desc = tempstr;      	
	else if (currentElement.equals("version")) version = tempstr;
	else if (currentElement.equals("language")) lang = tempstr;
	else if (currentElement.equals("paradiagm")) paradiag = tempstr;
	else if (currentElement.equals("usage")) usage = tempstr;
	else if (currentElement.equals("execution_options")) exeopt = tempstr;
	
}

public void endElement(String url, String name, String qname) {
	
	if (name.equalsIgnoreCase("application")){

	    // check if the application is already stored in DB
	    StringBuffer buf = new StringBuffer();
	    buf.append("select id from  ");
	    buf.append(APP_TABLE);
	    buf.append("  where name='" + this.name + "' and version='" + version + "'; ");
	    if (getDB().getDataItem(buf.toString()) == null){

	    	buf = new StringBuffer();
	    	buf.append("insert into");
	    	buf.append(" " + APP_TABLE + " ");
	    	buf.append("(name, version, description, language, para_diag, usage_text, execution_options)");
	    	buf.append(" values ");
	    	buf.append("('" + this.name + "', '" + version + "', '" 
		       + desc + "', '" + lang + "', '" + paradiag + "', '" + usage + "', '" + exeopt + "'); ");       
	    	// System.out.println(buf.toString());
	    	try {
		    getDB().executeUpdate(buf.toString());
		    buf.delete(0, buf.toString().length());
			if (getDB().getDBType().compareTo("mysql") == 0)
		    	buf.append("select LAST_INSERT_ID();");
			else
		    	buf.append("select currval('application_id_seq');");
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

