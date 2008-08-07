/*
 * SavedPreferences.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Font;
import java.awt.Point;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Map;
import java.util.Vector;

public class Preferences implements Serializable {

    private String paraProfFont = "SansSerif";
    private int fontStyle = Font.PLAIN;
    private int fontSize = 12;

    private Vector colors = null;
    private Vector groupColors = null;
    private Color highlightColor = null;
    private Color groupHighlightColor = null;
    private Color userEventHighlightColor = null;
    private Color miscFunctionColor = null;
    private String databasePassword = null;
    private String databaseConfigurationFile = null;
    private Point managerWindowPosition;
    private Map assignedColors;

    private boolean showValuesAsPercent = false;
    private boolean showPathTitleInReverse = false;
    private int units = 2;
    private boolean reversedCallPaths = false;
    private boolean computeMeanWithoutNulls = false;
    private boolean generateIntermediateCallPathData = false;
    
    private ArrayList sourceLocations;
    private boolean showSourceLocation = true;
    private boolean autoLabels = true;
    
    static final long serialVersionUID = 183442743456314793L;


    public void setColors(Vector vector) {
        colors = vector;
    }

    public Vector getColors() {
        return colors;
    }

    public void setGroupColors(Vector vector) {
        groupColors = vector;
    }

    public Vector getGroupColors() {
        return groupColors;
    }

    public void setHighlightColor(Color highlightColor) {
        this.highlightColor = highlightColor;
    }

    public Color getHighlightColor() {
        return highlightColor;
    }

    public void setGroupHighlightColor(Color grouphighlightColor) {
        this.groupHighlightColor = grouphighlightColor;
    }

    public Color getGroupHighlightColor() {
        return groupHighlightColor;
    }

    public void setUserEventHighlightColor(Color userEventHighlightColor) {
        this.userEventHighlightColor = userEventHighlightColor;
    }

    public Color getUserEventHighlightColor() {
        return userEventHighlightColor;
    }

    public void setMiscFunctionColor(Color miscFunctionColor) {
        this.miscFunctionColor = miscFunctionColor;
    }

    public Color getMiscFunctionColor() {
        return miscFunctionColor;
    }

    public String getFontName() {
        return paraProfFont;
    }

    public void setFontName(String paraProfFont) {
        this.paraProfFont = paraProfFont;
    }

    public void setFontStyle(int fontStyle) {
        this.fontStyle = fontStyle;
    }

    public int getFontStyle() {
        return fontStyle;
    }

    public void setFontSize(int fontSize) {
        this.fontSize = fontSize;
    }

    public int getFontSize() {
        return this.fontSize;
    }

    public void setDatabasePassword(String databasePassword) {
        this.databasePassword = databasePassword;
    }

    public String getDatabasePassword() {
        return databasePassword;
    }

    public void setDatabaseConfigurationFile(String databaseConfigurationFile) {
        this.databaseConfigurationFile = databaseConfigurationFile;
    }

    public String getDatabaseConfigurationFile() {
        return databaseConfigurationFile;
    }

    public void setManagerWindowPosition(Point position) {
        this.managerWindowPosition = position;
    }

    public Point getManagerWindowPosition() {
        return this.managerWindowPosition;
    }

    public void setAssignedColors(Map assignedColors) {
        this.assignedColors = assignedColors;
    }

    public Map getAssignedColors() {
        return assignedColors;
    }

    public boolean getShowPathTitleInReverse() {
        return showPathTitleInReverse;
    }

    public void setShowPathTitleInReverse(boolean showPathTitleInReverse) {
        this.showPathTitleInReverse = showPathTitleInReverse;
    }

    public boolean getShowValuesAsPercent() {
        return showValuesAsPercent;
    }

    public void setShowValuesAsPercent(boolean showValuesAsPercent) {
        this.showValuesAsPercent = showValuesAsPercent;
    }

    public int getUnits() {
        return units;
    }

    public void setUnits(int units) {
        this.units = units;
    }

    public boolean getReversedCallPaths() {
        return reversedCallPaths;
    }

    public void setReversedCallPaths(boolean reversedCallPaths) {
        this.reversedCallPaths = reversedCallPaths;
    }

    // this is reverse from everything else so that it can start out false
    // this way people that already had preferences files won't get the new 
    // behavior (they go to false if not found)
    public boolean getComputeMeanWithoutNulls() {
        return computeMeanWithoutNulls;
    }

    // this is reverse from everything else so that it can start out false
    // this way people that already had preferences files won't get the new 
    // behavior (they go to false if not found)
    public void setComputeMeanWithoutNulls(boolean computeMeanWithoutNulls) {
        this.computeMeanWithoutNulls = computeMeanWithoutNulls;
    }

    public boolean getGenerateIntermediateCallPathData() {
        return generateIntermediateCallPathData;
    }

    public void setGenerateIntermediateCallPathData(boolean generateIntermediateCallPathData) {
        this.generateIntermediateCallPathData = generateIntermediateCallPathData;
    }

	public ArrayList getSourceLocations() {
		return sourceLocations;
	}

	public void setSourceLocations(ArrayList sourceLocations) {
		this.sourceLocations = sourceLocations;
	}

    public boolean getShowSourceLocation() {
        return showSourceLocation;
    }

    public void setShowSourceLocation(boolean showSourceLocation) {
        this.showSourceLocation = showSourceLocation;
    }

    public boolean getAutoLabels() {
        return autoLabels;
    }

    public void setAutoLabels(boolean autoLabels) {
        this.autoLabels = autoLabels;
    }

}