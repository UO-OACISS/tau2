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
	
    protected String TRIAL_TABLE = "Trials";
    protected String PPROF_TABLE = "Pprof";   
    protected String UE_TABLE = "UserEvent";

    protected String XMLFILE_TABLE = "XMLfiles";

    protected String FUN_TABLE = "Funindex";
    protected int funIndexCounter;

    protected String LOC_TABLE = "Locationindex";
    protected int locCounter;

    protected String TOTAL_TABLE = "Totalsummary";
    
    protected String MEAN_TABLE = "Meansummary";
    
    protected String appid = "";
    protected String expid = "";
    protected String probsize = "";

    protected String currentElement = "";
    protected String documentName = "";
    protected String documentId = "";

    protected String metricStr = "";
    protected String trialId = "";
    protected String trialTime = "";

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
    protected String numofsamples = "";
    protected String maxvalue = "";
    protected String minvalue = "";
    protected String meanvalue = "";
       
    private DB dbconnector; 
    private int partitionFlag = 0;
    private String[] funArray;

    private File pprofTempFile;
    private BufferedWriter pwriter;

    private File ueTempFile;
    private BufferedWriter uewriter;

    private File funTempFile;
    private BufferedWriter fwriter;
 
    private File locTempFile;
    private BufferedWriter lwriter;

    private File totalTempFile;
    private BufferedWriter twriter;

    private File meanTempFile;
    private BufferedWriter mwriter;
	

    public LoadHandler(DB db){	
	
	super();
	this.dbconnector = db;
	
	try{

	    pprofTempFile = new File("pprof.tmp");
	    pprofTempFile.createNewFile();	
	    pwriter = new BufferedWriter(new FileWriter(this.pprofTempFile));

	    ueTempFile = new File("ue.tmp");
	    ueTempFile.createNewFile();
	    uewriter = new BufferedWriter(new FileWriter(this.ueTempFile));

	    funTempFile = new File("fun.tmp");
	    funTempFile.createNewFile();	
	    fwriter = new BufferedWriter(new FileWriter(this.funTempFile));

	    locTempFile = new File("loc.tmp");
	    locTempFile.createNewFile();	
	    lwriter = new BufferedWriter(new FileWriter(this.locTempFile));

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
	
	buf.append("select max(funindexid) from funindex;");
	
	String tempStr = getDB().getDataItem(buf.toString());

	if (tempStr == null)
	    funIndexCounter = 0;
	else
	    funIndexCounter = Integer.parseInt(tempStr);

	buf.delete(0, buf.toString().length());

	buf.append("select max(locid) from locationindex;");

	tempStr = getDB().getDataItem(buf.toString());

	if (tempStr == null)
	    locCounter = 0;
	else 
	    locCounter = Integer.parseInt(tempStr);	
 		
    }

    public void setTableName(String appid, String expid){
	if (partitionFlag!=0){
	    if (partitionFlag==1){ // i.e., partition according to applications only.
		TRIAL_TABLE = "trial_for_app"+appid;
		FUN_TABLE = "fun_for_app"+appid;
		PPROF_TABLE = "pprof_for_app"+appid;
		UE_TABLE = "ue_for_app"+appid;
		LOC_TABLE = "loc_for_app"+appid;
		TOTAL_TABLE = "total_for_app"+appid;
		MEAN_TABLE = "mean_for_app"+appid;
	    }
	    else if (partitionFlag == 2){ //i.e., partition according to experiments only.
		TRIAL_TABLE = "trial_for_exp"+expid;
		FUN_TABLE = "fun_for_exp"+expid;
		PPROF_TABLE = "pprof_for_exp"+expid;
		UE_TABLE = "ue_for_exp"+expid;
		LOC_TABLE = "loc_for_exp"+expid;
		TOTAL_TABLE = "total_for_exp"+expid;
		MEAN_TABLE = "mean_for_exp"+expid;
	    }
	    else if (partitionFlag == 3){ //i.e., partition according to experiments and applications.
		TRIAL_TABLE = "trial_for_app"+appid+"_exp"+expid;
		FUN_TABLE = "fun_for_app"+appid+"_exp"+expid;
		PPROF_TABLE = "pprof_for_app"+appid+"_exp"+expid;
		UE_TABLE = "ue_for_app"+appid+"_exp"+expid;
		LOC_TABLE = "loc_for_app"+appid+"_exp"+expid;
		TOTAL_TABLE = "total_for_app"+appid+"_exp"+expid;
		MEAN_TABLE = "mean_for_app"+appid+"_exp"+expid;
	    }
	    else {
		System.out.println("Wrong partition flag: 1 for application partition, 2 for experiment partition, and 3 for both.");
		System.exit(-1);
	    }
	}
    }

    public String getTrialTable(){ return TRIAL_TABLE; }

    public String getFunTable(){  return FUN_TABLE; }
    
    public String getPprofTable(){ return PPROF_TABLE; }

    public String getUETable(){ return UE_TABLE; }    

    public String getLocTable(){ return LOC_TABLE; }

    public String getTotalTable(){ return TOTAL_TABLE; }

    public String getMeanTable(){ return MEAN_TABLE; }

    public int getParFlag(){ return partitionFlag; }

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

    public void startDocument() throws SAXException{
		
	StringBuffer buf = new StringBuffer();
	buf.append("insert into");
	buf.append(" " + XMLFILE_TABLE + " ");
	buf.append("(xmlfilename)");
	buf.append(" values ");
	File ff = new File(getDocumentName());
        String filename = ff.getAbsolutePath();
	buf.append("('" + filename + "'); ");
	try{
	    getDB().executeUpdate(buf.toString());
	    buf.delete(0, buf.toString().length());
	    buf.append("select currval('xmlfiles_xmlfileid_seq');");
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
	if( name.equalsIgnoreCase("Onetrial") ) {
	    currentElement = "Onetrial";
	    metricStr = metricAttrToString(attrList);	     
	}
	if( name.equalsIgnoreCase("ComputationModel") ) {
	    currentElement = "ComputationModel";
	}
	if( name.equalsIgnoreCase("AppID") ){
	    currentElement = "AppID";		    
	}
	if( name.equalsIgnoreCase("ExpID") ){
	    currentElement = "ExpID";	
	}
	if( name.equalsIgnoreCase("ProblemSize") ){
	    currentElement = "ProblemSize";
	}
	if( name.equalsIgnoreCase("FunAmt") ){
	    currentElement = "FunAmt";
	}
	if( name.equalsIgnoreCase("UserEventAmt") ){	    
	    currentElement = "UEAmt";
	}    
	if( name.equalsIgnoreCase("Trialtime") ){
	    currentElement = "Trialtime";
	}
	if( name.equalsIgnoreCase("node") ) {
	    currentElement = "node";
	}
	if( name.equalsIgnoreCase("context") ) {
	    currentElement = "context";
	}
	if( name.equalsIgnoreCase("thread") ) {
	    currentElement = "thread";
	}
	if( name.equalsIgnoreCase("Pprof") ) {
	    currentElement = "Pprof";
	}
	if( name.equalsIgnoreCase("nodeID") ) {
	    currentElement = "nodeID";
	}
	if( name.equalsIgnoreCase("contextID") ) {
	    currentElement = "contextID";
	}
	if( name.equalsIgnoreCase("threadID") ) {
	    currentElement = "threadID";
	}
	if( name.equalsIgnoreCase("instrumentedobj") ) {
	    currentElement = "instrumentedobj";
	}
	if( name.equalsIgnoreCase("funname") ) {
	    currentElement = "funname";
	}
	if( name.equalsIgnoreCase("funID") ) {
	    currentElement = "funID";
	}
	if( name.equalsIgnoreCase("inclperc") ) {
	    currentElement = "inclperc";
	}
	if( name.equalsIgnoreCase("inclutime") ) {
	    currentElement = "inclutime";
	}
	if( name.equalsIgnoreCase("exclperc") ) {
	    currentElement = "exclperc";
	}
	if( name.equalsIgnoreCase("exclutime") ) {
	    currentElement = "exclutime";
	}
	if( name.equalsIgnoreCase("call") ) {
	    currentElement = "call";
	}
	if( name.equalsIgnoreCase("subrs") ) {
	    currentElement = "subrs";
	}
	if( name.equalsIgnoreCase("inclutimePcall") ) {
	    currentElement = "inclutimePcall";
	}
	if( name.equalsIgnoreCase("userevent") ) {
	    currentElement = "userevent";
	}    
	if( name.equalsIgnoreCase("uename") ) {
	    currentElement = "uename";
	}
	if( name.equalsIgnoreCase("ueID") ) {
	    currentElement = "ueID";
	}    
	if( name.equalsIgnoreCase("numofsamples") ) {
	    currentElement = "numofsamples";
	}
	if( name.equalsIgnoreCase("maxvalue") ) {
	    currentElement = "maxvalue";
	}    
	if( name.equalsIgnoreCase("minvalue") ) {
	    currentElement = "minvalue";
	}
	if( name.equalsIgnoreCase("meanvalue") ) {
	    currentElement = "meanvalue";
	}    
	if( name.equalsIgnoreCase("totalfunsummary") ) {
	    currentElement = "totalfunsummary";
	}    
	if( name.equalsIgnoreCase("meanfunsummary") ) {
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
	
	if (currentElement.equals("ExpID")) {
	    expid = tempcode;
	    if  (expid.length()==0){
		System.out.println("No valid experiment ID. Quit loadding.");
		System.exit(-1);
	    }

	    setParFlag(expid);
		
	    setTableName(appid, expid);	    	    
	}    

	if (currentElement.equals("ProblemSize")) probsize = tempcode;

	if (currentElement.equals("Trialtime")) trialTime = tempcode;
	
	if (currentElement.equals("FunAmt")) {
	    funAmt = Integer.parseInt(tempcode);	    
	}

	if (currentElement.equals("UEAmt")){
	    
	    ueAmt = Integer.parseInt(tempcode);
	    
	    System.out.println(getParFlag());

	    if (ueAmt > 0){
		
		ResultSet rs;
		StringBuffer buf = new StringBuffer();
		UEPartitionManager ueTable;

		try {
		    DatabaseMetaData dbMetaData = ((DBConnector)getDB()).getConnection().getMetaData();

		    if (getParFlag()==3){		    
			rs = dbMetaData.getTables(null, null, "ue_for_app"+appid, new String[]{ "TABLE" });

			if (rs.next() == false){// no user event tables generated for this application yet.
			    ueTable = new UEPartitionManager("LOC_FOR_APP" + appid);			
			    buf.append(ueTable.tableCreation("UE_FOR_APP"+appid));
			}		    
		    }	
	    
		    rs = dbMetaData.getTables(null, null, getUETable(), new String[]{ "TABLE" });
		    if (rs.next() == false){
			ueTable = new UEPartitionManager(getLocTable());
			if (getParFlag()==3) ueTable.setInheritTable("UE_FOR_APP"+appid);
			buf.append(ueTable.tableCreation(getUETable()));		       	       
		    }
		    
		    System.out.println(buf.toString());
		    if (buf.toString().length()>0)
			getDB().executeUpdate(buf.toString());			
			
		    rs.close();

		} catch (SQLException ex) {
		    ex.printStackTrace();
		}
	    }
	 	    
	    try{
		if ((ueAmt+funAmt) > 0)
		    funArray = new String[funAmt+ueAmt];
		else {
		    System.out.println("Cannot get a valid function amount, quit loadding.");
		    System.exit(-1);
		}
	    } catch (Exception ex){
		ex.printStackTrace();
	    }	
	}

	if (currentElement.equals("node")) nodenum = tempcode;
	 
	if (currentElement.equals("context")) contextpnode = tempcode;
	
	if (currentElement.equals("thread")) threadpcontext = tempcode;      	

	if (currentElement.equals("nodeID")) nodeid = tempcode;
	 
	if (currentElement.equals("contextID")) contextid = tempcode;
	
	if (currentElement.equals("threadID")) threadid = tempcode;
	
	if (currentElement.equals("funname")) funname += tempcode; 
      
	if (currentElement.equals("funID")) funid = tempcode;
	
	if (currentElement.equals("inclperc")) inclperc = tempcode;
	
	if (currentElement.equals("inclutime")) incl = tempcode;
	
	if (currentElement.equals("exclperc")) exclperc = tempcode;
	
	if (currentElement.equals("exclutime")) excl = tempcode;
	
	if (currentElement.equals("call")) callnum = tempcode;
	
	if (currentElement.equals("subrs")) subrs = tempcode;
	
	if (currentElement.equals("inclutimePcall")) inclpcall = tempcode;

	if (currentElement.equals("uename")) uename = tempcode;

	if (currentElement.equals("ueID")) ueid = tempcode; 

	if (currentElement.equals("numofsamples")) numofsamples = tempcode;

	if (currentElement.equals("maxvalue")) maxvalue = tempcode;

	if (currentElement.equals("minvalue")) minvalue = tempcode;

	if (currentElement.equals("meanvalue")) meanvalue = tempcode;
    }

    public void endElement(String url, String name, String qname) {
        StringBuffer buf = new StringBuffer();
	
	if (name.equalsIgnoreCase("ProblemSize")){
	    	    
	    buf.append("insert into");
	    buf.append(" " + getTrialTable() + " ");
	    if (probsize==""){	
	    	buf.append("(expid, time, metric, nodenum, contextpnode, threadpcontext, xmlfileid)");
	    	buf.append(" values ");
	    	buf.append("(" + expid + ", '" + trialTime 
			   + "', '" + metricStr + "', "  + nodenum 
			   + ", " + contextpnode
			   + ", " + threadpcontext + ", " + getDocumentId() + "); ");       
	    	System.out.println(buf.toString());
	    }
	    else {
	    	buf.append("(expid, time, metric, problemsize, nodenum, contextpnode, threadpcontext, xmlfileid)");
	    	buf.append(" values ");
	    	buf.append("(" + expid + ", '" + trialTime 
			   + "', '" + metricStr + "', " + probsize + ", " + nodenum 
			   + ", " + contextpnode
			   + ", " + threadpcontext + ", " + getDocumentId() + "); ");       
	    	System.out.println(buf.toString());
	    }
	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	buf.delete(0, buf.toString().length());
	    	buf.append("select currval('trials_trialid_seq');");
	    	trialId = getDB().getDataItem(buf.toString());
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }		    
	}

	if (name.equalsIgnoreCase("instrumentedobj")) {
	    try{
		int tempInt = Integer.parseInt(funid);
		if (funArray[tempInt] == null){

		    funIndexCounter++;
		    String ftempStr = String.valueOf(funIndexCounter)+"\t"+funid+"\t"+funname+"\t"+getTrialId();
		    fwriter.write(ftempStr, 0, ftempStr.length());
		    fwriter.newLine();
								
		    funArray[tempInt] = String.valueOf(funIndexCounter);	
		}	

		locCounter++;
		String ltempStr = String.valueOf(locCounter)+"\t"+nodeid+"\t"+contextid+"\t"+threadid+"\t"+funArray[tempInt];
		lwriter.write(ltempStr, 0, ltempStr.length());
		lwriter.newLine();
	    		     
		String ptempStr = String.valueOf(locCounter)+ "\t" + inclperc + "\t" + incl + "\t" + exclperc
		    + "\t" + excl + "\t" + callnum + "\t" + subrs + "\t" + inclpcall;
		pwriter.write(ptempStr, 0, ptempStr.length());
		pwriter.newLine();
	    } catch (IOException ex){
		ex.printStackTrace();
	    }
	    
	    funname = "";
	}	

	if (name.equalsIgnoreCase("userevent")){
	    try{		
		
		int ueidInt= Integer.parseInt(ueid);
		if (funArray[ueidInt+funAmt] == null){
		    
		    funIndexCounter++;
		    String ftempStr = String.valueOf(funIndexCounter)+"\t"+ueid+"\t"+uename+"\t"+getTrialId();
		    fwriter.write(ftempStr, 0, ftempStr.length());
		    fwriter.newLine();
					    
		    funArray[ueidInt+funAmt] = String.valueOf(funIndexCounter);	
		}	

		locCounter++;
		String ltempStr = String.valueOf(locCounter)+"\t"+nodeid+"\t"+contextid+"\t"+threadid+"\t"+funArray[ueidInt+funAmt];
		lwriter.write(ltempStr, 0, ltempStr.length());
		lwriter.newLine();
	    		     
		String uetempStr = String.valueOf(locCounter)+ "\t" + numofsamples + "\t" + maxvalue + "\t" + minvalue
		    + "\t" + meanvalue;
		uewriter.write(uetempStr, 0, uetempStr.length());
		uewriter.newLine();
	    } catch (IOException ex){
		ex.printStackTrace();
	    }
	    
	    uename = "";	    
	}

	if (name.equalsIgnoreCase("totalfunction")) {
	    try{
		String ttempStr = funArray[Integer.parseInt(funid)]  
		    + "\t" + inclperc  + "\t" + incl      + "\t" + exclperc
		    + "\t" + excl      + "\t" + callnum   + "\t" + subrs 
		    + "\t" + inclpcall;
	    
		twriter.write(ttempStr, 0, ttempStr.length());
		twriter.newLine();
	    } catch (IOException ex){
		ex.printStackTrace();
	    }
	    
	    funname = "";
	}	

	if (name.equalsIgnoreCase("meanfunction")) {
	    try{
		String mtempStr = funArray[Integer.parseInt(funid)]
		    + "\t" + inclperc  + "\t" + incl      + "\t" + exclperc
		    + "\t" + excl      + "\t" + callnum   + "\t" + subrs 
		    + "\t" + inclpcall;
	    
		mwriter.write(mtempStr, 0, mtempStr.length());
		mwriter.newLine();
	    } catch (IOException ex){
		ex.printStackTrace();
	    }

	    funname = "";
	}	

	if (name.equalsIgnoreCase("Onetrial")) {
	    try{
		fwriter.write("\\.", 0, ("\\.").length());
		fwriter.newLine();
		fwriter.close();

		lwriter.write("\\.", 0, ("\\.").length());
		lwriter.newLine();
		lwriter.close();

		pwriter.write("\\.", 0, ("\\.").length());
		pwriter.newLine();
		pwriter.close();

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

	    buf.append("copy ");
	    buf.append(getFunTable());
	    buf.append(" from ");
	    buf.append("'" + funTempFile.getAbsolutePath() + "';");

	    System.out.println(buf.toString());

	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }	

	    buf.delete(0, buf.toString().length());
	    	
	    buf.append("copy ");
	    buf.append(getLocTable());
	    buf.append(" from ");
	    buf.append("'" + locTempFile.getAbsolutePath() + "';");

	    System.out.println(buf.toString());

	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }	

	    buf.delete(0, buf.toString().length());

	    buf.append("copy ");
	    buf.append(getPprofTable());
	    buf.append(" from ");
	    buf.append("'" + pprofTempFile.getAbsolutePath() + "';");

	    System.out.println(buf.toString());

	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }

	    if (ueAmt > 0){

		buf.delete(0, buf.toString().length());

		buf.append("copy ");
		buf.append(getUETable());
		buf.append(" from ");
		buf.append("'" + ueTempFile.getAbsolutePath() + "';");

		System.out.println(buf.toString());

		try{	
		    getDB().executeUpdate(buf.toString());
	    	
		} catch (SQLException ex){
		    ex.printStackTrace();
		}
	    }

	    buf.delete(0, buf.toString().length());

	    buf.append("copy ");
	    buf.append(getTotalTable());
	    buf.append(" from ");
	    buf.append("'" + totalTempFile.getAbsolutePath() + "';");

	    System.out.println(buf.toString());

	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }

	    buf.delete(0, buf.toString().length());

	    buf.append("copy ");
	    buf.append(getMeanTable());
	    buf.append(" from ");
	    buf.append("'" + meanTempFile.getAbsolutePath() + "';");

	    System.out.println(buf.toString());

	    try{	
	    	getDB().executeUpdate(buf.toString());
	    	
	    } catch (SQLException ex){
                ex.printStackTrace();
	    }
	       
	    funTempFile.delete();   
	    pprofTempFile.delete();
	    ueTempFile.delete();
	    locTempFile.delete();
	    totalTempFile.delete();
	    meanTempFile.delete();
	    	    
	}
    }

    public void setParFlag(String expid){
	StringBuffer buf = new StringBuffer();
	buf.append("select trial_table_name from experiments where expid = "+expid+";");
	String tableName = getDB().getDataItem(buf.toString()).toUpperCase();
	//System.out.println(tableName);
	if (tableName.indexOf("_APP")>0)
	    if (tableName.indexOf("_EXP")>0)
		this.partitionFlag = 3;
	    else 
		this.partitionFlag = 1;
	else 
	    if (tableName.indexOf("EXP")>0)
		this.partitionFlag = 2;
	    else 
		this.partitionFlag = 0;
    }

}

