/* 
  SavedPreferences.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package ParaProf;

import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;

public class SavedPreferences implements Serializable
{ 
  //******************************
  //Instance data.
  //******************************
  Vector globalColors = null;
  Vector mappingGroupColors = null;
  
  private Color highlightColor = null;
  private Color groupHighlightColor = null;
  private Color uEHC = null;
  private Color miscMappingsColor = null;
  private int barSpacing = -1;
  private int barHeight = -1;
  boolean barDetailsSet = false;
  
  String ParaProfFont;
  
  String inExValue = null;
  String sortBy = null;
  
  int fontStyle = -1;
  //******************************
  //End - Instance data.
  //******************************
  
  public void SavedPreferences()
  {
  }
  
  public void setGlobalColors(Vector inVector)
  {
    globalColors = inVector;
  }
  
  public Vector getGlobalColors()
  {
    return globalColors;
  }
  
  public void setMappingGroupColors(Vector inVector)
  {
    mappingGroupColors = inVector;
  }
  
  public Vector getMappingGroupColors()
  {
    return mappingGroupColors;
  }
  
  public void setHighlightColor(Color inhighlightColor)
  {
    highlightColor = inhighlightColor;
  }
  
  public Color getHighlightColor()
  {
    return highlightColor;
  }
  
  public void setGroupHighlightColor(Color inGrouphighlightColor)
  {
    groupHighlightColor = inGrouphighlightColor;
  }
  
  public Color getGroupHighlightColor()
  {
    return groupHighlightColor;
  }
  
  public void setUEHC(Color inUEHC)
  {
    uEHC = inUEHC;
  }
  
  public Color getUEHC()
  {
    return uEHC;
  }
  
  public void setMiscMappingsColor(Color inMiscMappingsColor)
  {
    miscMappingsColor = inMiscMappingsColor;
  }
  
  public Color getMiscMappingsColor()
  {
    return miscMappingsColor;
  }
  
  public String getJRacyFont()
  {
    return ParaProfFont;
  } 
  
  public void setJRacyFont(String inString)
  {
    ParaProfFont = inString;
  }
  
  public void setBarSpacing(int inInt)
  {
    barSpacing = inInt;
  }
  
  public void setBarHeight(int inInt)
  {
    barHeight = inInt;
  }
  
  public int getBarSpacing()
  {
    return barSpacing;
  }
  
  public int getBarHeight()
  {
    return barHeight;
  }
  
  public void setBarDetailsSet(boolean inBool)
  {
    barDetailsSet = inBool;
  }
  
  public boolean getBarDetailsSet()
  {
    return barDetailsSet;
  }
  
  public void setInExValue(String inString)
  {
    inExValue = inString;
  }
  
  public String getInExValue()
  {
    return inExValue;
  }
  
  public void setFontStyle(int inInt)
  {
    fontStyle = inInt;
  }
  
  public int getFontStyle()
  {
    return fontStyle;
  }
  
  //Setting and returning sortBy.
  public void setSortBy(String inString)
  {
    sortBy = inString;
  }
  
  public String getSortBy()
  {
    return sortBy;
  }
  
}
