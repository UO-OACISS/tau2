package edu.uoregon.tau.common;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class Zip {

    public static void unzip(String input) throws FileNotFoundException, IOException {
        ZipInputStream zin = new ZipInputStream(new BufferedInputStream(new FileInputStream(input)));
				int BUFFER = 2048;
        ZipEntry entry;
				byte data[] = new byte[BUFFER];
				int count;
				while((entry = zin.getNextEntry()) != null)
				{
					if (entry.isDirectory())
					{
						//System.out.println("Creating: " + entry);
						new File(entry.getName()).mkdirs();		
					}
					else
					{
						//System.out.println("Extracting: " + entry);
						FileOutputStream fos = new FileOutputStream(entry.getName());
						BufferedOutputStream bos = new BufferedOutputStream(fos, BUFFER);
						while ((count = zin.read(data, 0, BUFFER)) != -1) {
							bos.write(data, 0, count);
						}
						bos.close();
					}
				}
				zin.close();
    }
		public static void main(String[] args) {
			if (args.length == 1) {
						try {
					unzip(args[0]);
				} catch (IOException e) {
					System.out.println("Failed unzipping " + e);
				}
			} else {
				System.out.println("Usage: Zip <local filename>");
				System.out.println("Failed unzipping " + args[0]);
			}
		}

}
