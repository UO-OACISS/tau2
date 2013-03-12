/**
 * 
 */
package edu.uoregon.tau.perfdmf;

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

import com.google.gson.*;

/**
 * @author khuck
 *
 */
public class MetaDataParserJSON {

	private MetaDataParserJSON () {
	}
	
	public static void parse (Map<String, String> metadataMap, File fileName) {
		try {
			parse(metadataMap, DataSource.readFileAsString(fileName), true);
		} catch (IOException e) {
			System.err.println("Error reading metadata:");
			System.err.println(e.getMessage());
//			e.printStackTrace();
		}
	}
	
	public static boolean isJSON(String string) {
		try {
			Gson gson = new Gson();
			Object obj = gson.fromJson(string, Object.class);
		} catch (JsonSyntaxException e) {
			return false;
		}
		return true;
	}
	
	public static void parse (Map<String, String> metadataMap, String string, boolean full) throws JsonSyntaxException {
		Gson gson = new Gson();
		System.out.println(string);
		Object obj = gson.fromJson(string, Object.class);
		System.out.println(obj.getClass().toString());
		if (obj.getClass() == LinkedHashMap.class) {
			Map<String, Object> map = (LinkedHashMap<String,Object>)obj;
			for (Map.Entry<String, Object> entry : map.entrySet()) {
				processElement(metadataMap, entry, "", full);
			}
		}
	}

	private static void processElement(Map<String, String> metadataMap,
			Map.Entry<String, Object> entry, String prefix, boolean full) {
		String key = entry.getKey();
		Object value = entry.getValue();
		if (prefix.length() > 0)
			key = prefix + ":" + key;
		if (value == null) {
			System.out.println("null");
			metadataMap.put(key, null);
		} else {
			if (value.getClass() == LinkedHashMap.class) {
				if (full) {
					Map<String, Object> map = (LinkedHashMap<String,Object>)value;
					for (Map.Entry<String, Object> innerEntry : map.entrySet()) {
						processElement(metadataMap, innerEntry, key, full);
					}
				}
			} else {
				System.out.println(value.getClass().toString());
				metadataMap.put(key, value.toString());
			}
		}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Map<String, String> map = new TreeMap<String, String>();
		MetaDataParserJSON.parse(map, new File(args[0]));
		for (Map.Entry<String, String> entry : map.entrySet()) {
		    String key = entry.getKey();
		    String value = entry.getValue();
		    System.out.println(key + " = " + value);
		}
	}

}
