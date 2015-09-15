/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package pt.ist.mcp.powertools;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.StringUtils;
import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddCluster;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Gabriel
 */
public class Clusterer {

    /**
     * @param args the command line arguments
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            throw new IllegalArgumentException("Input file missing.");
        }

        String input = args[0];
        String output;

        if (args.length >= 2 && args[1] != null)
            output = args[1];
        else {
            output = FilenameUtils.removeExtension(input) + "_clusered.csv";
        }

        Instances data = loadFile(input);
        Instances newData = cluster(data);
        saveFile(output, newData);
    }

    /**
     * Loads a CSV file into a dataset.
     * @param name The path to the file.
     * @return The dataset.
     * @throws IOException
     */
    public static Instances loadFile(String name) throws IOException {
        CSVLoader csv = new CSVLoader();
        csv.setFile(new File(name));
        return csv.getDataSet();
    }

    /**
     * Saves a dataset into a CSV file.
     * @param name The path to the file.
     * @param data The dataset.
     * @throws IOException
     */
    public static void saveFile(String name, Instances data) throws IOException {
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File(name));
        saver.writeBatch();
    }

    /**
     * Clusters student data. Check the papers in http://web.ist.utl.pt/gabriel.barata/ for mode details.
     * @param data The dataset.
     * @return The dataset with a columns respecting the student clusters.
     * @throws Exception
     */
    public static Instances cluster(Instances data) throws Exception {
        EM clustererEM = new EM();

        // compute the indexes to exclude and the one respecting to the last day.
        Pattern regex = Pattern.compile("XP on day \\d+");
        ArrayList<Integer> indexesToIgnore = new ArrayList<>();
        Enumeration attributes = data.enumerateAttributes();
        int lastAtt = 0;
        while (attributes.hasMoreElements()) {
            Attribute att = (Attribute) attributes.nextElement();
            Matcher m = regex.matcher(att.name());
            if (!m.matches())
                indexesToIgnore.add(att.index() + 1);
            else
                lastAtt = att.index();
        }

        // convert the array to a string
        String indexesToIgnoreStr = StringUtils.join(indexesToIgnore, ",");

        // perform EM
        // source: http://weka.8497.n7.nabble.com/Clustering-with-AddCluster-using-Java-code-td32371.html
        clustererEM.setOptions(weka.core.Utils.splitOptions("-I 100 -N -1 -M 1.0E-6 -S 100"));
        AddCluster addCluster = new AddCluster();
        addCluster.setClusterer(clustererEM);
        addCluster.setIgnoredAttributeIndices(indexesToIgnoreStr);
        addCluster.setInputFormat(data);
        Instances newData = Filter.useFilter(data, addCluster);

        // let's find some stats for each cluster, for the last day
        newData.renameAttribute(newData.attribute("cluster"), "Cluster");
        newData.setClass(newData.attribute("Cluster"));
        ArrayList<String> classes = (ArrayList<String>) Collections.list(newData.attribute("Cluster").enumerateValues());

        ArrayList<Integer> counts = new ArrayList<>(Collections.nCopies(classes.size(), 0));
        ArrayList<Integer> sums = new ArrayList<>(Collections.nCopies(classes.size(), 0));
        ArrayList<Double> averages = new ArrayList<>(Collections.nCopies(classes.size(), 0.0));

        Enumeration instances = newData.enumerateInstances();
        Instance instance;
        String classAtt;
        Integer lastAttVal, count, sum, classIndex;
        while (instances.hasMoreElements()) {
            instance = (Instance) instances.nextElement();
            classAtt = instance.stringValue(newData.classIndex());
            classIndex = classes.indexOf(classAtt);
            lastAttVal = (int) instance.value(lastAtt);

            counts.set(classIndex, counts.get(classIndex) + 1);
            sums.set(classIndex, sums.get(classIndex) + lastAttVal);
        }

        // make the averages
        for (int i = 0; i < classes.size(); i++) {
            averages.set(i, (double) sums.get(i) / (double) counts.get(i));
        }

        // let's rename them to a A,B,C,D,etc., based on their average performance on the last day
        class ClusterComparator implements Comparator<String> {
            @Override
            public int compare(String s1, String s2) {
                return averages.get(classes.indexOf(s2)).compareTo(averages.get(classes.indexOf(s1)));
            }
        }

        ArrayList<String> newClasses = (ArrayList<String>) classes.clone();
        Collections.sort(newClasses, new ClusterComparator());

        String newval;
        for (int i = 0; i < newClasses.size(); i++) {
            newval = String.valueOf((char) (((int) 'A') + i));
            newData.renameAttributeValue(newData.attribute("Cluster"), newClasses.get(i), newval);
            newClasses.set(i, newval);
        }

        return newData;
    }

}