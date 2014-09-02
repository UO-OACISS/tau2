package edu.uoregon.tau.perfdmf.database;

import java.io.FileInputStream;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.security.KeyStore;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;
import javax.net.ssl.KeyManager;
import javax.net.ssl.SSLContext;
///* Production code should include its own implementation of this: */import org.apache.commons.lang.exception.ExceptionUtils;

import org.postgresql.ssl.WrappedFactory;

public class CustomSSLSocketFactory extends WrappedFactory {

    public CustomSSLSocketFactory(String arg) throws GeneralSecurityException {
        // If JSSE debugging is on, print information about our setup
        String jsseDebug = System.getProperty("javax.net.debug");
        boolean debug = jsseDebug != null && (jsseDebug.startsWith("all") || jsseDebug.startsWith("ssl"));

        KeyManager[] key_managers = null;

        // Find out where to get our key data
        String ksType = System.getProperty("javax.net.ssl.keyStoreType", KeyStore.getDefaultType());
        String ksPath = System.getProperty("javax.net.ssl.keyStore");
        String ksPass = System.getProperty("javax.net.ssl.keyStorePassword");

        if (debug) {
            System.out.println("CustomSSLSocketFactory creation beginning.");
            System.out.println("Using keystore " + ksPath + " of type " + ksType + " pass " + ksPass);
        }

        if (ksPath != null) {
            // Load the key store the user's keys should be in
            KeyStore ks = KeyStore.getInstance(ksType);
            try {
                FileInputStream is = new FileInputStream(ksPath);
                ks.load(is, ksPass.toCharArray());
                is.close();
            } catch (CertificateException ex) {
                throw new GeneralSecurityException(ex);
            } catch (IOException ex) {
                throw new GeneralSecurityException(ex);
            } catch (NoSuchAlgorithmException ex) {
                throw new GeneralSecurityException(ex);
            }

            KeyManager km = new CustomX509KeyManager(ks, ksPass.toCharArray());
            key_managers = new KeyManager[] { km };
        }

        // Set up a custom SSL context using our loaded key store and custom
        // key manager. We'll perform all operations with this custom context.
        SSLContext ctx = SSLContext.getInstance("TLS");
        ctx.init(key_managers, null, null);
        _factory = ctx.getSocketFactory();
        if (debug) {
            System.out.println("CustomSSLSocketFactory creation successful");
        }
    }

}
