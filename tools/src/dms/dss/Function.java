package com.perfdb.api;

import com.perfdb.util.dbinterface.*;
import com.perfdb.util.io.*;
import com.perfdb.dbmanager.*;
import java.io.*;
import java.net.*;
import java.sql.*;


public class Function {
	private int functionIndexID;
	private int functionID;
	private String name;
	private int trialID;
	private int experimentID;
	private int applicationID;
	private FunctionDataObject meanSummary;
	private FunctionDataObject totalSummary;

	public void setIndexID (int id) {
		this.functionIndexID = id;
	}

	public void setFunctionID (int id) {
		this.functionID = id;
	}

	public void setName (String name) {
		this.name = name;
	}

	public void setTrialID (int id) {
		this.trialID = id;
	}

	public void setExperimentID (int id) {
		this.experimentID = id;
	}

	public void setApplicationID (int id) {
		this.applicationID = id;
	}

	public void setMeanSummary (FunctionDataObject meanSummary) {
		this.meanSummary = meanSummary;
	}

	public void setTotalSummary (FunctionDataObject totalSummary) {
		this.totalSummary = totalSummary;
	}

	public int getIndexID () {
		return this.functionIndexID;
	}

	public int getFunctionID () {
		return this.functionID;
	}

	public String getName () {
		return this.name;
	}

	public int getTrialID () {
		return this.trialID;
	}

	public int getExperimentID () {
		return this.experimentID;
	}

	public int getApplicationID () {
		return this.applicationID;
	}

	public FunctionDataObject getMeanSummary () {
		return this.meanSummary;
	}

	public FunctionDataObject getTotalSummary () {
		return this.totalSummary;
	}
}

