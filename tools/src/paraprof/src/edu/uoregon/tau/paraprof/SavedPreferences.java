/*
 * SavedPreferences.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.io.*;
import java.awt.*;

public class SavedPreferences implements Serializable {
    
    boolean loadedFromFile = false;

    public void setLoaded(boolean b) {
        loadedFromFile = true;
    }
    
    public boolean getLoaded() {
        return loadedFromFile;
    }
    
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

    public Color getUserEventHightlightColor() {
        return userEventHighlightColor;
    }

    public void setMiscFunctionColor(Color miscFunctionColor) {
        this.miscFunctionColor = miscFunctionColor;
    }

    public Color getMiscFunctionColor() {
        return miscFunctionColor;
    }

    public String getParaProfFont() {
        return paraProfFont;
    }

    public void setParaProfFont(String paraProfFont) {
        this.paraProfFont = paraProfFont;
    }

    public void setBarSpacing(int barSpacing) {
        this.barSpacing = barSpacing;
    }

    public int getBarSpacing() {
        return barSpacing;
    }

    public void setBarHeight(int barHeight) {
        this.barHeight = barHeight;
    }

    public int getBarHeight() {
        return barHeight;
    }

    public void setBarDetailsSet(boolean barDetailsSet) {
        this.barDetailsSet = barDetailsSet;
    }

    public boolean getBarDetailsSet() {
        return barDetailsSet;
    }

    public void setInclusiveOrExclusive(String inclusiveOrExclusive) {
        this.inclusiveOrExclusive = inclusiveOrExclusive;
    }

    public String getInclusiveOrExclusive() {
        return inclusiveOrExclusive;
    }

    public void setFontStyle(int fontStyle) {
        this.fontStyle = fontStyle;
    }

    public int getFontStyle() {
        return fontStyle;
    }

    public void setSortBy(String sortBy) {
        this.sortBy = sortBy;
    }

    public String getSortBy() {
        return sortBy;
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

    //####################################
    //Instance data.
    //####################################
    private Vector colors = null;
    private Vector groupColors = null;
    private Color highlightColor = null;
    private Color groupHighlightColor = null;
    private Color userEventHighlightColor = null;
    private Color miscFunctionColor = null;
    private int barSpacing = -1;
    private int barHeight = -1;
    private boolean barDetailsSet = false;
    private String paraProfFont;
    private String inclusiveOrExclusive = null;
    private String sortBy = null;
    private int fontStyle = -1;
    private String databasePassword = null;
    private String databaseConfigurationFile = null;
    //####################################
    //End - Instance data.
    //####################################
}