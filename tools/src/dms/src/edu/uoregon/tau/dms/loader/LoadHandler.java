package edu.uoregon.tau.dms.loader;

import edu.uoregon.tau.dms.database.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/*** SAX Handler which creates SQL to load a document into the database. ***/

public class LoadHandler extends DefaultHandler {
	
    protected String appid = "";
    protected String expid = "";
    protected String probsize = "";
    protected String trialName = "";
    protected String userData = "";
    protected String problemString = "";

    protected String currentElement = "";
    protected String documentName = "";
    protected String documentId = "";

    protected String metricStr = "";
    protected String metricId = "";
    protected String trialId = "";
    protected String trialTime = "";
	protected boolean newTrial = false;

    protected int funAmt;
    protected int ueAmt;

    protected String nodenum = "";
    protected String contextpnode = "";
    protected String threadpcontext = "";
    protected String nodeid = "";
    protected String threadid = "";
    protected String contextid = "";

    protected String funid = "";
    protected String funname = "";
    protected String fungroup = "";
    protected String funIndex = "";
    protected String locid = "";

    protected String inclperc = "";
    protected String incl = "";
    protected String exclperc = "";
    protected String excl = "";
    protected String callnum = "";
    protected String subrs = "";
    protected String inclpcall = "";
    
    protected String uename = "";
    protected String ueid = "";
    protected String uegroup = "UE";
    protected String numofsamples = "";
    protected String maxvalue = "";
    protected String minvalue = "";
    protected String meanvalue = "";
    protected String standardDeviation = "";
       
    private DB dbconnector; 
    private String[] funArray;
    private String[] ueArray;

    public LoadHandler(DB db, String trialId, String problemString){	
		super();
		this.dbconnector = db;
		this.trialId = trialId;
		this.problemString = problemString;
    }

    public DB getDB() {
	return dbconnector;
    }

    public String getDocumentId(){
	return documentId;  
    }

    public String getTrialId() {
	return trialId;
    }

    public String getDocumentName() {
	if (documentName == null) {
	    setDocumentName("NoName" + getDocumentId());
	}
	return documentName;
    }

    public void setDocumentName(String newValue) {
	this.documentName = newValue;
    }

    /*** Initialize the document table when begining loading a XML document.*/
	// we will set the trial and metric ids to 0 initially, but they will be updated.

    public void startDocument() throws SAXException{
    }

    public String metricAttrToString(Attributes attrList) {
        StringBuffer buf = new StringBuffer();
        int length = attrList.getLength();
        for (int i=0;i<length;i++) {
	    buf.append(attrList.getValue(i));
        }
        return buf.toString();
    }

    public void startElement(String url, String name, String qname, Attributes attrList) throws SAXException {	
	
		if( name.equalsIgnoreCase("Onetrial") ) {
	    	metricStr = metricAttrToString(attrList);	     
		}
		currentElement = new String(name);
    }

    /*** Handle character data regions. ***/

    public void characters(char[] chars, int start, int length) {
	
	// Check if characters is whitespace, if so, return
   
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

	String tempcode = new String(chars, start, length);

	if (currentElement.equals("AppID")) {
	    appid = tempcode;
	    if  (appid.length()==0){
		System.out.println("No valid application ID. Quit loadding.");
		System.exit(-1);
	    }	    
	}
	
	else if (currentElement.equals("ExpID")) {
	    expid = tempcode;
	    if  (expid.length()==0){
		System.out.println("No valid experiment ID. Quit loadding.");
		System.exit(-1);
	    }
	}    

	else if (currentElement.equals("ProblemSize")) probsize = tempcode;

	else if (currentElement.equals("TrialName")) trialName = tempcode;

	else if (currentElement.equals("UserData")) userData = tempcode;

	else if (currentElement.equals("Trialtime")) trialTime = tempcode;
	
	else if (currentElement.equals("FunAmt")) {
	    funAmt = Integer.parseInt(tempcode);	    
		if (funAmt > 0)
		    funArray = new String[funAmt];
	}

	else if (currentElement.equals("UserEventAmt")){
	    ueAmt = Integer.parseInt(tempcode);
		if (ueAmt > 0)
		    ueArray = new String[ueAmt];
		if ((ueAmt+funAmt) == 0) {
		    System.out.println("Cannot get a valid function amount, quit loadding.");
		    System.exit(-1);
	    }	
	}

	else if (currentElement.equals("node")) nodenum = tempcode;
	else if (currentElement.equals("context")) contextpnode = tempcode;
	else if (currentElement.equals("thread")) threadpcontext = tempcode;      	
	else if (currentElement.equals("nodeID")) nodeid = tempcode;
	else if (currentElement.equals("contextID")) contextid = tempcode;
	else if (currentElement.equals("threadID")) threadid = tempcode;
	else if (currentElement.equals("funname")) funname += tempcode; 
	else if (currentElement.equals("fungroup")) fungroup += tempcode;
	else if (currentElement.equals("funID")) funid = tempcode;
	else if (currentElement.equals("inclperc")) inclperc = tempcode;
	else if (currentElement.equals("inclutime")) incl = tempcode;
	else if (currentElement.equals("exclperc")) exclperc = tempcode;
	else if (currentElement.equals("exclutime")) excl = tempcode;
	else if (currentElement.equals("call")) callnum = tempcode;
	else if (currentElement.equals("subrs")) subrs = tempcode;
	else if (currentElement.equals("inclutimePcall")) inclpcall = tempcode;
	else if (currentElement.equals("uename")) uename = tempcode;
	else if (currentElement.equals("ueID")) ueid = tempcode; 
	else if (currentElement.equals("numofsamples")) numofsamples = tempcode;
	else if (currentElement.equals("maxvalue")) maxvalue = tempcode;
	else if (currentElement.equals("minvalue")) minvalue = tempcode;
	else if (currentElement.equals("meanvalue")) meanvalue = tempcode;
	else if (currentElement.equals("stddevvalue")) standardDeviation = tempcode;
    }

    public void endElement(String url, String name, String qname) {
        StringBuffer buf = new StringBuffer();
	
	if (name.equalsIgnoreCase("usereventamt")){
	    	    
		// if this is a new trial, create a new trial
		if (trialId.compareTo("0") == 0) {
			newTrial = true;
	    	buf.append("insert into trial (experiment, name, time, node_count, contexts_per_node, threads_per_context, userdata, problem_definition)");
	    	buf.append(" values ");
	    	buf.append("(" + expid + ", '" + trialName + "', '" + trialTime 
			+ "', "  + nodenum 
			+ ", " + contextpnode
			+ ", " + threadpcontext 
			+ ", '" + userData
			+ "', '" + problemString + "'); ");       
	   		// System.out.println(buf.toString());
	    	try{	
	    		getDB().executeUpdate(buf.toString());
	    		buf.delete(0, buf.toString().length());
				if (getDB().getDBType().compareTo("mysql") == 0)
		    		buf.append("select LAST_INSERT_ID();");
				else
	    			buf.append("select currval('trial_id_seq');");
	    		trialId = getDB().getDataItem(buf.toString());
	    	} catch (SQLException ex){
                	ex.printStackTrace();
					System.out.println(buf.toString());
		System.exit(0);
	    	}		    
		} else {
	    	try{	
				// get the functions from the database
				newTrial = false;
				buf.append("select id, function_number from function where trial = "+trialId+" order by function_number asc;");
	    		ResultSet functions = getDB().executeQuery(buf.toString());	
				String tmpId;
				int tmpInt;
	    		while (functions.next() != false){
					tmpId = functions.getString(1);
					tmpInt = functions.getInt(2);
		    		funArray[tmpInt] = tmpId;
				}
				functions.close();
	    	} catch (SQLException ex){
                	ex.printStackTrace();
					System.out.println(buf.toString());
		System.exit(0);
	    	}		    
		}

		// insert the metric
		try{
	    	buf.delete(0, buf.toString().length());
	    	buf.append("insert into metric (name, trial) values (TRIM('");
			buf.append(metricStr);
	    	buf.append("'), " + trialId + ");");
	   		// System.out.println(buf.toString());
	    	getDB().executeUpdate(buf.toString());
	    	buf.delete(0, buf.toString().length());
			if (getDB().getDBType().compareTo("mysql") == 0)
		    	buf.append("select LAST_INSERT_ID();");
			else
	    		buf.append("select currval('metric_id_seq');");
	    	metricId = getDB().getDataItem(buf.toString());
	    } catch (SQLException ex) {
			ex.printStackTrace();
					System.out.println(buf.toString());
		System.exit(0);
		}

	}

	if (name.equalsIgnoreCase("instrumentedobj")) {
		int tempInt = Integer.parseInt(funid);
		// insert the function into the database
		if (funArray[tempInt] == null){
			// if the function doesn't belong to any group.
		   	if (fungroup.trim().length() == 0) 
				fungroup = "NA";
	    	buf.delete(0, buf.toString().length());
	    	buf.append("insert into function (trial, function_number, name, group_name) values (");
			buf.append(getTrialId() + ", ");
			buf.append(funid + ", '");
			buf.append(funname + "', '");
			buf.append(fungroup + "');");
	    	try{	
	   		// System.out.println(buf.toString());
	    		getDB().executeUpdate(buf.toString());
	    		buf.delete(0, buf.toString().length());
				if (getDB().getDBType().compareTo("mysql") == 0)
		   			buf.append("select LAST_INSERT_ID();");
				else
	    			buf.append("select currval('function_id_seq');");
		   		funArray[tempInt] = getDB().getDataItem(buf.toString());
	    	} catch (SQLException ex){
               		ex.printStackTrace();
					System.out.println(buf.toString());
		System.exit(0);
	    	}		    
		}	

		// insert the interval location into the database
	    buf.delete(0, buf.toString().length());
	    buf.append("insert into interval_location_profile (function, node, context, thread, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call) VALUES ( ");
		buf.append(funArray[tempInt] + ", ");
		buf.append(nodeid + ", ");
		buf.append(contextid + ", ");
		buf.append(threadid + ", ");
		buf.append(metricId + ", ");
		buf.append(inclperc + ", ");
		buf.append(incl + ", ");
		buf.append(exclperc + ", ");
		buf.append(excl + ", ");
		buf.append(callnum + ", ");
		buf.append(subrs + ", ");
		buf.append(inclpcall + ");");
	    try{	
	   		// System.out.println(buf.toString());
	    	getDB().executeUpdate(buf.toString());
	    } catch (SQLException ex){
              		ex.printStackTrace();
					System.out.println(buf.toString());
		System.exit(0);
	    }		    
	    
	    funname = "";
	    fungroup = "";
	}	

	// don't add the user event if this is not a new trial
	if (name.equalsIgnoreCase("userevent") && newTrial){
		int ueidInt= Integer.parseInt(ueid);
		if (ueArray[ueidInt] == null){
	    	buf.delete(0, buf.toString().length());
			buf.append("insert into user_event (trial, name, group_name) VALUES (");
		   	buf.append(getTrialId() + ", '");
			buf.append(uename + "', '");
			buf.append(uegroup + "');");
	    	try{	
	   		// System.out.println(buf.toString());
	    		getDB().executeUpdate(buf.toString());
	    		buf.delete(0, buf.toString().length());
				if (getDB().getDBType().compareTo("mysql") == 0)
		   			buf.append("select LAST_INSERT_ID();");
				else
	    			buf.append("select currval('user_event_id_seq');");
		   		ueArray[ueidInt] = getDB().getDataItem(buf.toString());
	    	} catch (SQLException ex){
               		ex.printStackTrace();
					System.out.println(buf.toString());
		System.exit(0);
	    	}		    
		}	

	    buf.delete(0, buf.toString().length());
		buf.append("insert into atomic_location_profile (user_event, node, context, thread, sample_count, maximum_value, minimum_value, mean_value, standard_deviation) VALUES (");
		buf.append(ueArray[ueidInt] + ", ");
		buf.append(nodeid + ", ");
		buf.append(contextid + ", ");
		buf.append(threadid + ", ");
		buf.append(numofsamples + ", ");
		buf.append(maxvalue + ", ");
		buf.append(minvalue + ", ");
		buf.append(meanvalue + ", ");
		buf.append(standardDeviation + ");");
	    try{	
	   		// System.out.println(buf.toString());
	    	getDB().executeUpdate(buf.toString());
	    } catch (SQLException ex){
			ex.printStackTrace();
					System.out.println(buf.toString());
		System.exit(0);
	    }		    
	    
	    uename = "";	    
	}

	if (name.equalsIgnoreCase("totalfunction")) {
	    buf.delete(0, buf.toString().length());
		buf.append("insert into interval_total_summary (function, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call) VALUES ( ");
		buf.append(funArray[Integer.parseInt(funid)] + ", ");
		buf.append(metricId + ", ");
		buf.append(inclperc + ", ");
		buf.append(incl + ", ");
		buf.append(exclperc + ", ");
		buf.append(excl + ", ");
		buf.append(callnum + ", ");
		buf.append(subrs + ", ");
		buf.append(inclpcall + ");");
	    try{	
	   		// System.out.println(buf.toString());
	    	getDB().executeUpdate(buf.toString());
	    } catch (SQLException ex){
			ex.printStackTrace();
					System.out.println(buf.toString());
		System.exit(0);
	    }		    
	    
	    funname = "";
	    fungroup = "";
	}	

	if (name.equalsIgnoreCase("meanfunction")) {
	    buf.delete(0, buf.toString().length());
		buf.append("insert into interval_mean_summary (function, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call) VALUES ( ");
		buf.append(funArray[Integer.parseInt(funid)] + ", ");
		buf.append(metricId + ", ");
		buf.append(inclperc + ", ");
		buf.append(incl + ", ");
		buf.append(exclperc + ", ");
		buf.append(excl + ", ");
		buf.append(callnum + ", ");
		buf.append(subrs + ", ");
		buf.append(inclpcall + ");");
	    try{	
	    	getDB().executeUpdate(buf.toString());
	    } catch (SQLException ex){
	   		System.out.println(buf.toString());
			ex.printStackTrace();
					System.out.println(buf.toString());
		System.exit(0);
	    }		    

	    funname = "";
	    fungroup = "";
	}	
    }
}

