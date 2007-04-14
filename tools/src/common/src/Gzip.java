package edu.uoregon.tau.common;

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class Gzip {

    public static void gunzip(String input, String output) throws FileNotFoundException, IOException {
        GZIPInputStream gis = new GZIPInputStream(new BufferedInputStream(new FileInputStream(input)));
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(output));
        int c;
        while ((c = gis.read()) != -1) {
            bos.write((byte) c);
        }
        gis.close();
        bos.close();
    }

	public static final byte[] compress(String str) {
		byte[] compressed = null;
		try {
			ByteArrayOutputStream out = new ByteArrayOutputStream();
			GZIPOutputStream zout = new GZIPOutputStream(out);
			//zout.putNextEntry(new ZipEntry("0"));
			zout.write(str.getBytes());
			//zout.closeEntry();
			zout.finish();
			compressed = out.toByteArray();
			zout.close();
		} catch (IOException e) {
			System.err.println("Couldn't compress string!");
		}
		return compressed;
	}

	public static final String decompress(byte[] compressed) {
		String decompressed = null;
		try {
			ByteArrayOutputStream out = new ByteArrayOutputStream();
			ByteArrayInputStream in = new ByteArrayInputStream(compressed);
			GZIPInputStream zin = new GZIPInputStream(in);
			byte[] buffer = new byte[1024];
			int offset = -1;
			while((offset = zin.read(buffer)) != -1) {
				out.write(buffer, 0, offset);
			}
			decompressed = out.toString();
			out.close();
			zin.close();
		} catch (NullPointerException e) {
			// nothing to decompress
		} catch (IOException e) {
			System.err.println("Couldn't decompress string!");
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
		return decompressed;
	}

	public static final String decompress(InputStream in) {
		String decompressed = null;
		try {
			ByteArrayOutputStream out = new ByteArrayOutputStream();
			GZIPInputStream zin = new GZIPInputStream(in);
			byte[] buffer = new byte[1024];
			int offset = -1;
			while((offset = zin.read(buffer)) != -1) {
				out.write(buffer, 0, offset);
			}
			decompressed = out.toString();
			out.close();
			zin.close();
		} catch (NullPointerException e) {
			// nothing to decompress
		} catch (IOException e) {
			System.err.println("Couldn't decompress string!");
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
		return decompressed;
	}

    
    
    
}
