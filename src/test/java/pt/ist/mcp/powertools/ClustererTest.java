package pt.ist.mcp.powertools;

import org.apache.commons.io.FilenameUtils;
import org.junit.Test;
import weka.core.Instances;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.util.Arrays;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class ClustererTest {

    @Test
    public void testMain() throws Exception {
        System.out.println("main");
        try {
            Clusterer.main(new String[]{"src/test/resources/1011.csv"});
        } catch (Exception e) {
            fail(e.getMessage());
        }
    }

    @Test
    public void testCluster() throws Exception {
        System.out.println("cluster");

        String input = "src/test/resources/1011.csv";
        String output = FilenameUtils.removeExtension(input) + "_clusered.csv";
        String test = "src/test/resources/1011c.csv";

        try {
            Instances data = Clusterer.loadFile(input);
            Instances newData = Clusterer.cluster(data);
            Clusterer.saveFile(output, newData);
        } catch (Exception e) {
            fail(e.getMessage());
        }

        // let's compare the md5 of test and clustered file
        MessageDigest md = MessageDigest.getInstance("MD5");
        try (InputStream is = Files.newInputStream(Paths.get(output))) {
            DigestInputStream dis = new DigestInputStream(is, md);
        }
        byte[] output_digest = md.digest();

        try (InputStream is = Files.newInputStream(Paths.get(test))) {
            DigestInputStream dis = new DigestInputStream(is, md);
        }
        byte[] test_digest = md.digest();

        assertTrue(Arrays.equals(output_digest, test_digest));
    }

    @Test
    public void testLoadFile() throws Exception {
        System.out.println("testLoadFile");
        String input = "src/test/resources/1011.csv";
        Instances data = Clusterer.loadFile(input);
    }

    @Test
    public void testSaveFile() throws Exception {
        System.out.println("testSaveFile");
        String input = "src/test/resources/1011.csv";
        String output = FilenameUtils.removeExtension(input) + "_clusered.csv";
        Instances data = Clusterer.loadFile(input);
        Instances newData = Clusterer.cluster(data);
        Clusterer.saveFile(output, newData);
    }
}