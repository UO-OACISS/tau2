package perfdb.loadxml;

import perfdb.util.dbinterface.*;
import perfdb.dbmanager.*;
import org.xml.sax.*;
import org.xml.sax.helpers.*;
import java.util.*;
import java.io.*;

import java.lang.*;
import java.sql.*;

/*** SAX Handler which creates SQL to load a document into the database. ***/

public class LoadHandler extends DefaultHandler {
	
    protected String TRIAL_TABLE = "trial";
    protected String XMLFILE_TABLE = "xml_file";
    protected String FUN_TABLE = "function";
    protected String TOTAL_TABLE = "interval_total_summary";
    protected String MEAN_TABLE = "interval_mean_summary";
    protected String INTER_LOC_TABLE = "interval_location_profile";
    protected String UE_TABLE = "user_event";
    protected String ATOMIC_LOC_TABLE = "atomic_location_profile";
    protected int funIndexCounter;
    protected int funTotalCounter;
    protected int funMeanCounter;
    protected int ueIndexCounter;
    protected int interLocCounter;
    protected int atomicLocCounter;
    
    protected String appid = "";
    protected String expid = "";
    protected String probsize = "";

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

    private File ueTempFile;
    private BufferedWriter uewriter;

    private File funTempFile;
    private BufferedWriter fwriter;
 
    private File interLocTempFile;
    private BufferedWriter ilwriter;

    private File atomicLocTempFile;
    private BufferedWriter alwriter;

    private File totalTempFile;
    private BufferedWriter twriter;

    private File meanTempFile;
    private BufferedWriter mwriter;
	

    public LoadHandler(DB db, String trialId){	
	
		super();
		this.dbconnector = db;
		this.trialId = trialId;
	
		try{

		    atomicLocTempFile = new File("atomicLoc.tmp");
		    atomicLocTempFile.createNewFile();	
		    alwriter = new BufferedWriter(new FileWriter(this.atomicLocTempFile));

		    ueTempFile = new File("ue.tmp");
		    ueTempFile.createNewFile();
		    uewriter = new BufferedWriter(new FileWriter(this.ueTempFile));

		    funTempFile = new File("fun.tmp");
		    funTempFile.createNewFile();	
		    fwriter = new BufferedWriter(new FileWriter(this.funTempFile));

		    interLocTempFile = new File("interLoc.tmp");
		    interLocTempFile.createNewFile();	
		    ilwriter = new BufferedWriter(new FileWriter(this.interLocTempFile));

		    totalTempFile = new File("total.tmp");
		    totalTempFile.createNewFile();	
		    twriter = new BufferedWriter(new FileWriter(this.totalTempFile));

		    meanTempFile = new File("mean.tmp");
		    meanTempFile.createNewFile();	
		    mwriter = new BufferedWriter(new FileWriter(this.meanTempFile));
	   
		}catch (IOException ioe){
		    ioe.printStackTrace();
		}

		StringBuffer buf = new StringBuffer();
	
		// get the current max(id) for the function table

		buf.append("select max(id) from function;");
		String tempStr = getDB().getDataItem(buf.toString());
		if (tempStr == null)
	    	funIndexCounter = 0;
		else
	    	funIndexCounter = Integer.parseInt(tempStr);

		// get the current max(id) for the interval_total_summary table

		buf.delete(0, buf.toString().length());
		buf.append("select max(id) from interval_total_summary;");
		tempStr = getDB().getDataItem(buf.toString());
		if (tempStr == null)
	    	funTotalCounter = 0;
		else
	    	funTotalCounter = Integer.parseInt(tempStr);

		// get the current max(id) for the interval_mean_summary table

		buf.delete(0, buf.toString().length());
		buf.append("select max(id) from interval_mean_summary;");
		tempStr = getDB().getDataItem(buf.toString());
		if (tempStr == null)
	    	funMeanCounter = 0;
		else
	    	funMeanCounter = Integer.parseInt(tempStr);

		// get the current max(id) for the user_event table

		buf.delete(0, buf.toString().length());
		buf.append("select max(id) from user_event;");
		tempStr = getDB().getDataItem(buf.toString());
		if (tempStr == null)
	    	ueIndexCounter = 0;
		else
	    	ueIndexCounter = Integer.parseInt(tempStr);

		// get the current max(id) for the interval_location_profile table

		buf.delete(0, buf.toString().length());
		buf.append("select max(id) from interval_location_profile;");
		tempStr = getDB().getDataItem(buf.toString());
		if (tempStr == null)
	    	interLocCounter = 0;
		else 
	    	interLocCounter = Integer.parseInt(tempStr);	

		// get the current max(id) for the atomic_location_profile table

		buf.delete(0, buf.toString().length());
		buf.append("select max(id) from atomic_location_profile;");
		tempStr = getDB().getDataItem(buf.toString());
		if (tempStr == null)
	    	atomicLocCounter = 0;
		else 
	    	atomicLocCounter = Integer.parseInt(tempStr);	
    }

    public String getTrialTable(){ return TRIAL_TABLE; }

    public String getFunTable(){  return FUN_TABLE; }
    
    public String getInterLocTable(){ return INTER_LOC_TABLE; }

    public String getUETable(){ return UE_TABLE; }    

    public String getAtomicLocTable(){ return ATOMIC_LOC_TABLE; }

    public String getTotalTable(){ return TOTAL_TABLE; }

    public String getMeanTable(){ return MEAN_TABLE; }

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
		
	StringBuffer buf = new StringBuffer();
	buf.append("insert into ");
	buf.append(XMLFILE_TABLE);
	buf.append(" (trial, metric, name)");
	buf.append(" values ");
	File ff = new File(getDocumentName());
        String filename = ff.getAbsolutePath();
	buf.append("(0, 0, '" + filename + "'); ");
	try{
	    getDB().executeUpdate(buf.toString());
	    buf.delete(0, buf.toString().length());
		if (getDB().getDBType().compareTo("mysql") == 0)
		   	buf.append("select LAST_INSERT_ID();");
		else
	    	buf.append("select currval('xml_file_id_seq');");
	    documentId = getDB().getDataItem(buf.toString());
	} catch (SQLException ex) {
	    ex.printStackTrace();
        }
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
	
		if( name.equalsIgnoreCase("Trials") ) {
	    	currentElement = "Trials";
		}
		else if( name.equalsIgnoreCase("Onetrial") ) {
	    	currentElement = "Onetrial";
	    	metricStr = metricAttrToString(attrList);	     
		}
		else if( name.equalsIgnoreCase("ComputationModel") ) {
	    	currentElement = "ComputationModel";
		}
		else if( name.equalsIgnoreCase("AppID") ){
	    	currentElement = "AppID";		    
		}
		else if( name.equalsIgnoreCase("ExpID") ){
	    	currentElement = "ExpID";	
		}
		else if( name.equalsIgnoreCase("ProblemSize") ){
	    	currentElement = "ProblemSize";
		}
		else if( name.equalsIgnoreCase("FunAmt") ){
	    	currentElement = "FunAmt";
		}
		else if( name.equalsIgnoreCase("UserEventAmt") ){	    
	    	currentElement = "UEAmt";
		}    
		else if( name.equalsIgnoreCase("Trialtime") ){
	    	currentElement = "Trialtime";
		}
		else if( name.equalsIgnoreCase("node") ) {
	    	currentElement = "node";
		}
		else if( name.equalsIgnoreCase("context") ) {
	    	currentElement = "context";
		}
		else if( name.equalsIgnoreCase("thread") ) {
	    	currentElement = "thread";
		}
		else if( name.equalsIgnoreCase("Pprof") ) {
	    	currentElement = "Pprof";
		}
		else if( name.equalsIgnoreCase("nodeID") ) {
	    	currentElement = "nodeID";
		}
		else if( name.equalsIgnoreCase("contextID") ) {
	    	currentElement = "contextID";
		}
		else if( name.equalsIgnoreCase("threadID") ) {
	    	currentElement = "threadID";
		}
		else if( name.equalsIgnoreCase("instrumentedobj") ) {
	    	currentElement = "instrumentedobj";
		}
		else if( name.equalsIgnoreCase("funname") ) {
	    	currentElement = "funname";
		}
		else if( name.equalsIgnoreCase("funID") ) {
	    	currentElement = "funID";
		}
		else if (name.equalsIgnoreCase("fungroup")){
	    	currentElement = "fungroup";
		}
		else if( name.equalsIgnoreCase("inclperc") ) {
	    	currentElement = "inclperc";
		}
		else if( name.equalsIgnoreCase("inclutime") ) {
	    	currentElement = "inclutime";
		}
		else if( name.equalsIgnoreCase("exclperc") ) {
	    	currentElement = "exclperc";
		}
		else if( name.equalsIgnoreCase("exclutime") ) {
	    	currentElement = "exclutime";
		}
		else if( name.equalsIgnoreCase("call") ) {
	    	currentElement = "call";
		}
		else if( name.equalsIgnoreCase("subrs") ) {
	    	currentElement = "subrs";
		}
		else if( name.equalsIgnoreCase("inclutimePcall") ) {
	    	currentElement = "inclutimePcall";
		}
		else if( name.equalsIgnoreCase("userevent") ) {
	    	currentElement = "userevent";
		}    
		else if( name.equalsIgnoreCase("uename") ) {
	    	currentElement = "uename";
		}
		else if( name.equalsIgnoreCase("ueID") ) {
	    	currentElement = "ueID";
		}    
		else if( name.equalsIgnoreCase("numofsamples") ) {
	    	currentElement = "numofsamples";
		}
		else if( name.equalsIgnoreCase("maxvalue") ) {
	    	currentElement = "maxvalue";
		}    
		else if( name.equalsIgnoreCase("minvalue") ) {
	    	currentElement = "minvalue";
		}
		else if( name.equalsIgnoreCase("meanvalue") ) {
	    	currentElement = "meanvalue";
		}    
		else if( name.equalsIgnoreCase("stddevvalue") ) {
	    	currentElement = "stddevvalue";
		}    
		else if( name.equalsIgnoreCase("totalfunsummary") ) {
	    	currentElement = "totalfunsummary";
		}    
		else if( name.equalsIgnoreCase("meanfunsummary") ) {
	    	currentElement = "meanfunsummary";
		}	 
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

	else if (currentElement.equals("Trialtime")) trialTime = tempcode;
	
	else if (currentElement.equals("FunAmt")) {
	    funAmt = Integer.parseInt(tempcode);	    
		if (funAmt > 0)
		    funArray = new String[funAmt];
	}

	else if (currentElement.equals("UEAmt")){
	    ueAmt = Integer.parseInt(tempcode);
		if (ueAmt > 0)
		    ueArray = new String[ueAmt];
		if ((ueAmt+funAmt) == 0) {
		    System.out.println("Cannot get a valid function amount, quit loadding.");
		    System.exit(-1);
	    }	
	}

	if (currentElement.equals("node")) nodenum = tempcode;
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
	
	if (name.equalsIgnoreCase("ProblemSize")){
	    	    
		// if this is a new trial, create a new trial
		if (trialId.compareTo("0") == 0) {
			newTrial = true;
	    	buf.append("insert into ");
	    	buf.append(getTrialTable());
	    	if (probsize==""){	
	    		buf.append(" (experiment, time, node_count, contexts_per_node, threads_per_context)");
	    		buf.append(" values ");
	    		buf.append("(" + expid + ", '" + trialTime 
			   	+ "', "  + nodenum 
			   	+ ", " + contextpnode
			   	+ ", " + threadpcontext + "); ");       
	    	} else {
	    		buf.append(" (experiment, time, problem_size, node_count, contexts_per_node, threads_per_context)");
	    		buf.append(" values ");
	    		buf.append("(" + expid + ", '" + trialTime 
			   	+ "', " + probsize + ", " + nodenum 
			   	+ ", " + contextpnode
			   	+ ", " + threadpcontext + "); ");       
	    	}
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
	    	}		    
		}

		// insert the metric
		try{
	    	buf.delete(0, buf.toString().length());
	    	buf.append("insert into metric (name) values (TRIM('");
			buf.append(metricStr);
	    	buf.append("'));");
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
		}

	    try{	
			// update the xml_file table to have this trial, metric
	   		buf.delete(0, buf.toString().length());
			buf.append("update " + XMLFILE_TABLE + " set trial = " + trialId);
			buf.append(", metric = " + metricId + " where id = " + getDocumentId() + ";");
	   		// System.out.println(buf.toString());
	    	getDB().executeUpdate(buf.toString());
	    } catch (SQLException ex){
            ex.printStackTrace();
	    }		    
	}

	if (name.equalsIgnoreCase("instrumentedobj")) {
	    try{
			int tempInt = Integer.parseInt(funid);
			if (funArray[tempInt] == null){
		    	funIndexCounter++;
		    	if (fungroup.trim().length() == 0) // the function doesn't belong to any group.
					fungroup = "NA";
		    	String ftempStr = String.valueOf(funIndexCounter)+"\t"+getTrialId()+"\t"+funid+"\t"+funname+"\t"+fungroup;
		    	fwriter.write(ftempStr, 0, ftempStr.length());
		    	fwriter.newLine();
		    	funArray[tempInt] = String.valueOf(funIndexCounter);	
			}	

			interLocCounter++;
			String ltempStr = String.valueOf(interLocCounter)+"\t"+funArray[tempInt]+"\t"+nodeid+"\t"+contextid+"\t"+threadid+"\t"+ metricId + "\t"+ inclperc + "\t" + incl + "\t" + exclperc + "\t" + excl + "\t" + callnum + "\t" + subrs + "\t" + inclpcall;
			ilwriter.write(ltempStr, 0, ltempStr.length());
			ilwriter.newLine();
	    } catch (IOException ex){
			ex.printStackTrace();
	    }
	    
	    funname = "";
	    fungroup = "";
	}	

	// don't add the user event if this is not a new trial
	if (name.equalsIgnoreCase("userevent") && newTrial){
	    try{		
			int ueidInt= Integer.parseInt(ueid);
			if (ueArray[ueidInt] == null){
		    	ueIndexCounter++;
		    	String ftempStr = String.valueOf(ueIndexCounter)+"\t"+getTrialId()+"\t"+uename+"\t"+uegroup;
		    	uewriter.write(ftempStr, 0, ftempStr.length());
		    	uewriter.newLine();
		    	ueArray[ueidInt] = String.valueOf(ueIndexCounter);	
			}	

			atomicLocCounter++;
			String ltempStr = String.valueOf(atomicLocCounter)+"\t"+ueArray[ueidInt]+"\t"+nodeid+"\t"+contextid+"\t"+threadid+"\t" + numofsamples + "\t" + maxvalue + "\t" + minvalue + "\t" + meanvalue + "\t" + standardDeviation;
			alwriter.write(ltempStr, 0, ltempStr.length());
			alwriter.newLine();
	    		     
	    } catch (IOException ex){
			ex.printStackTrace();
	    }
	    
	    uename = "";	    
	}

	if (name.equalsIgnoreCase("totalfunction")) {
	    try{
		funTotalCounter++;
		String ttempStr = String.valueOf(funTotalCounter) + "\t" + funArray[Integer.parseInt(funid)]  
			+ "\t" + metricId
		    + "\t" + inclperc  + "\t" + incl      + "\t" + exclperc
		    + "\t" + excl      + "\t" + callnum   + "\t" + subrs 
		    + "\t" + inclpcall;
	    
		twriter.write(ttempStr, 0, ttempStr.length());
		twriter.newLine();
	    } catch (IOException ex){
		ex.printStackTrace();
	    }
	    
	    funname = "";
	    fungroup = "";
	}	

	if (name.equalsIgnoreCase("meanfunction")) {
	    try{
		funMeanCounter++;
		String mtempStr = String.valueOf(funMeanCounter) + "\t" + funArray[Integer.parseInt(funid)]
			+ "\t" + metricId
		    + "\t" + inclperc  + "\t" + incl      + "\t" + exclperc
		    + "\t" + excl      + "\t" + callnum   + "\t" + subrs 
		    + "\t" + inclpcall;
	    
		mwriter.write(mtempStr, 0, mtempStr.length());
		mwriter.newLine();
	    } catch (IOException ex){
		ex.printStackTrace();
	    }

	    funname = "";
	    fungroup = "";
	}	

	if (name.equalsIgnoreCase("Onetrial")) {
	    try{
		fwriter.write("\\.", 0, ("\\.").length());
		fwriter.newLine();
		fwriter.close();

		ilwriter.write("\\.", 0, ("\\.").length());
		ilwriter.newLine();
		ilwriter.close();

		alwriter.write("\\.", 0, ("\\.").length());
		alwriter.newLine();
		alwriter.close();

		uewriter.write("\\.", 0, ("\\.").length());
		uewriter.newLine();
		uewriter.close();

		twriter.write("\\.", 0, ("\\.").length());
		twriter.newLine();
		twriter.close();

		mwriter.write("\\.", 0, ("\\.").length());
		mwriter.newLine();
		mwriter.close();
	    } catch (IOException ex){
		ex.printStackTrace();
	    }

		if (getDB().getDBType().compareTo("mysql") == 0) {
	    	buf.append("load data infile '");
	    	buf.append(funTempFile.getAbsolutePath());
	    	buf.append("' into table ");
	    	buf.append(getFunTable() + ";");
		} else {
	    	buf.append("copy ");
	    	buf.append(getFunTable());
	    	buf.append(" from ");
	    	buf.append("'" + funTempFile.getAbsolutePath() + "';");
		}

	    // System.out.println(buf.toString());

	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }	

	    buf.delete(0, buf.toString().length());
	    	
		if (getDB().getDBType().compareTo("mysql") == 0) {
	    	buf.append("load data infile '");
	    	buf.append(interLocTempFile.getAbsolutePath());
	    	buf.append("' into table ");
	    	buf.append(getInterLocTable() + ";");
		} else {
	    	buf.append("copy ");
	    	buf.append(getInterLocTable());
	    	buf.append(" from ");
	    	buf.append("'" + interLocTempFile.getAbsolutePath() + "';");
		}

	    // System.out.println(buf.toString());

	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }	

	    if (ueAmt > 0){

			buf.delete(0, buf.toString().length());

			if (getDB().getDBType().compareTo("mysql") == 0) {
	    		buf.append("load data infile '");
	    		buf.append(ueTempFile.getAbsolutePath());
	    		buf.append("' into table ");
	    		buf.append(getUETable() + ";");
			} else {
				buf.append("copy ");
				buf.append(getUETable());
				buf.append(" from ");
				buf.append("'" + ueTempFile.getAbsolutePath() + "';");
			}

			// System.out.println(buf.toString());

			try{	
		    	getDB().executeUpdate(buf.toString());
	    		
			} catch (SQLException ex){
		    	ex.printStackTrace();
			}

	    	buf.delete(0, buf.toString().length());
	    	
			if (getDB().getDBType().compareTo("mysql") == 0) {
	    		buf.append("load data infile '");
	    		buf.append(atomicLocTempFile.getAbsolutePath());
	    		buf.append("' into table ");
	    		buf.append(getAtomicLocTable() + ";");
			} else {
	    		buf.append("copy ");
	    		buf.append(getAtomicLocTable());
	    		buf.append(" from ");
	    		buf.append("'" + atomicLocTempFile.getAbsolutePath() + "';");
			}

	    	// System.out.println(buf.toString());

	    	try{	
	    		getDB().executeUpdate(buf.toString());
	    		
	    	} catch (SQLException ex){
                	ex.printStackTrace();
	    	}	
	    }

	    buf.delete(0, buf.toString().length());

		if (getDB().getDBType().compareTo("mysql") == 0) {
	    	buf.append("load data infile '");
	    	buf.append(totalTempFile.getAbsolutePath());
	    	buf.append("' into table ");
	    	buf.append(getTotalTable() + ";");
		} else {
	    	buf.append("copy ");
	    	buf.append(getTotalTable());
	    	buf.append(" from ");
	    	buf.append("'" + totalTempFile.getAbsolutePath() + "';");
		}

	    // System.out.println(buf.toString());

	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }

	    buf.delete(0, buf.toString().length());

		if (getDB().getDBType().compareTo("mysql") == 0) {
	    	buf.append("load data infile '");
	    	buf.append(meanTempFile.getAbsolutePath());
	    	buf.append("' into table ");
	    	buf.append(getMeanTable() + ";");
		} else {
	    	buf.append("copy ");
	    	buf.append(getMeanTable());
	    	buf.append(" from ");
	    	buf.append("'" + meanTempFile.getAbsolutePath() + "';");
		}

	    // System.out.println(buf.toString());

	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }
	       
	    funTempFile.delete();   
	    interLocTempFile.delete();
	    // ueTempFile.delete();
	    atomicLocTempFile.delete();
	    totalTempFile.delete();
	    meanTempFile.delete();
	    	    
	}
    }

}

