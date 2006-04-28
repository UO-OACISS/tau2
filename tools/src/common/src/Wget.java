package edu.uoregon.tau.common;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;

public class Wget {

    public static void wget(String URL, String file) throws IOException {

        URLConnection url = new URL(URL).openConnection();
        DataInputStream in = new DataInputStream(url.getInputStream());
        OutputStream out = new FileOutputStream(file);

        try {
            while (true) {
                out.write(in.readUnsignedByte());
            }
        } catch (EOFException e) {
            out.close();
            return;
        }
    }

	public static void main(String[] args) {
		if (args.length != 2)
			System.out.println("Usage: Wget <url> <local filename>");
        try {
			Wget.wget(args[0], args[1]);
		} catch (IOException e) {
			System.out.println("Failed getting " + args[0]);
		}
	}
}
